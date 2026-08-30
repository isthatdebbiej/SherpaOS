import json
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import snapshot_download
from torch import nn

MODEL_REPO = "iteratehack/sherpaos-risk-tcn-balanced-v2"


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.pad = 4 * dilation
        self.conv = nn.Conv1d(inputs, outputs, 5, padding=self.pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(outputs)
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, values):
        result = self.conv(values)[..., : values.shape[-1]]
        return nn.functional.gelu(self.norm(result)) + self.skip(values)


class TCN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers, width = [], 103
        for index, output in enumerate(channels):
            layers += [ResidualBlock(width, output, 2**index), nn.Dropout(0.1)]
            width = output
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, 2)

    def forward(self, values):
        return self.head(self.net(values)[..., -1])


root = Path(snapshot_download(MODEL_REPO, token=os.environ["HF_TOKEN"]))
checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
model = TCN(checkpoint["config"]["channels"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
normalization = json.loads((root / "normalization.json").read_text())
metrics = json.loads((root / "validation_metrics.json").read_text())
mean = np.asarray(normalization["mean"], dtype=np.float32)
std = np.asarray(normalization["std"], dtype=np.float32)
thresholds = np.asarray(
    [metrics["mobility"]["threshold"], metrics["dynamics"]["threshold"]],
    dtype=np.float32,
)

api = FastAPI(title="SherpaOS Shadow Risk Inference", version="1")


@api.get("/health")
def health():
    return {
        "status": "ok",
        "model_repo": MODEL_REPO,
        "shadow_only": True,
        "input_shape": [100, 103],
    }


@api.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    values = np.asarray(payload.get("observations"), dtype=np.float32)
    if values.shape != (100, 103) or not np.isfinite(values).all():
        raise HTTPException(422, "observations must be one finite 100x103 window")
    normalized = (values - mean) / std
    tensor = torch.from_numpy(normalized[None]).transpose(1, 2)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0].numpy()
    active = probability >= thresholds
    return {
        "shadow_only": True,
        "mobility_probability": float(probability[0]),
        "dynamics_probability": float(probability[1]),
        "mobility_threshold": float(thresholds[0]),
        "dynamics_threshold": float(thresholds[1]),
        "mobility_risk": bool(active[0]),
        "dynamics_risk": bool(active[1]),
        "actuation_authorized": False,
        "model_repo": MODEL_REPO,
    }
