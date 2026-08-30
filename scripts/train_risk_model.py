"""Train the two-head SherpaOS temporal risk model without opening test data."""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
        layers = []
        width = 103
        for i, out in enumerate(channels):
            layers += [ResidualBlock(width, out, 2**i), nn.Dropout(0.1)]
            width = out
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, 2)

    def forward(self, x):
        return self.head(self.net(x)[..., -1])


def load(root, split):
    rows = [json.loads(x) for x in (root / f"{split}_index.jsonl").read_text().splitlines()]
    shards = defaultdict(dict)
    for r in rows:
        shards[(r["observations"], r["labels"])][r["episode_id"]] = r["global_episode_id"]
    xs = []
    ys = []
    ids = []
    cohorts = []
    for (op, lp), identity in sorted(shards.items()):
        with (
            np.load(root / op, allow_pickle=False) as o,
            np.load(root / lp, allow_pickle=False) as labels,
        ):
            assert np.array_equal(o["episode_ids"], labels["episode_ids"])
            m = np.isin(o["episode_ids"], list(identity))
            xs += [o["observations"][m]]
            ys += [np.c_[labels["mobility_targets"][m], labels["dynamics_targets"][m]]]
            global_ids = np.asarray([identity[value] for value in o["episode_ids"][m]])
            ids += [global_ids]
            cohorts += [np.asarray([value.split("/", 1)[0] for value in global_ids])]
    x = np.concatenate(xs).astype("float32")
    y = np.concatenate(ys).astype("float32")
    e = np.concatenate(ids)
    assert x.shape[1:] == (100, 103) and np.isfinite(x).all()
    cohort = np.concatenate(cohorts)
    assert set(e) == {r["global_episode_id"] for r in rows}
    return x, y, e, cohort


def metrics(y, p):
    out = {}
    for j, n in enumerate(("mobility", "dynamics")):
        order = np.argsort(-p[:, j])
        t = y[order, j]
        tp = np.cumsum(t)
        fp = np.cumsum(1 - t)
        rec = tp / max(t.sum(), 1)
        prec = tp / np.maximum(tp + fp, 1)
        ap = np.sum((rec - np.r_[0, rec[:-1]]) * prec)
        auc = np.trapezoid(np.r_[0, rec, 1], np.r_[0, fp / max(len(t) - t.sum(), 1), 1])
        candidates = []
        for q in np.linspace(0.01, 0.99, 197):
            predicted = p[:, j] >= q
            recall = np.sum(predicted & (y[:, j] == 1)) / max(np.sum(y[:, j] == 1), 1)
            false_positive = np.sum(predicted & (y[:, j] == 0)) / max(np.sum(y[:, j] == 0), 1)
            if recall >= 0.90:
                candidates.append((false_positive, -q, q))
        best = min(candidates)[2] if candidates else 0.01
        out[n] = {
            "average_precision": float(ap),
            "auroc": float(auc),
            "brier": float(np.mean((p[:, j] - y[:, j]) ** 2)),
            "threshold": float(best),
            "recall": float(
                np.sum((p[:, j] >= best) & (y[:, j] == 1)) / max(np.sum(y[:, j] == 1), 1)
            ),
            "false_positive_rate": float(
                np.sum((p[:, j] >= best) & (y[:, j] == 0)) / max(np.sum(y[:, j] == 0), 1)
            ),
            "precision": float(
                np.sum((p[:, j] >= best) & (y[:, j] == 1)) / max(np.sum(p[:, j] >= best), 1)
            ),
            "threshold_policy": "minimum_fpr_at_recall_gte_0.90",
            "recall_at_0_5": float(
                np.sum((p[:, j] >= 0.5) & (y[:, j] == 1)) / max(np.sum(y[:, j] == 1), 1)
            ),
            "false_positive_rate_at_0_5": float(
                np.sum((p[:, j] >= 0.5) & (y[:, j] == 0)) / max(np.sum(y[:, j] == 0), 1)
            ),
        }
    return out


a = argparse.ArgumentParser()
a.add_argument("--dataset", type=Path, required=True)
a.add_argument("--output", type=Path, required=True)
a.add_argument("--push-to-hub")
z = a.parse_args()
c = yaml.safe_load(Path("configs/training.yaml").read_text())
random.seed(c["seed"])
np.random.seed(c["seed"])
torch.manual_seed(c["seed"])
xt, yt, it, ct = load(z.dataset, "train")
xv, yv, iv, cv = load(z.dataset, "validation")
assert not (set(it) & set(iv))
mean = xt.mean((0, 1))
std = xt.std((0, 1))
std[std < 1e-6] = 1
xt = (xt - mean) / std
xv = (xv - mean) / std
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCN(c["channels"]).to(dev)
w = (len(yt) - yt.sum(0)) / np.maximum(yt.sum(0), 1)
lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=dev))
opt = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"])
tl = DataLoader(
    TensorDataset(torch.from_numpy(xt), torch.from_numpy(yt)),
    batch_size=c["batch_size"],
    shuffle=True,
    generator=torch.Generator().manual_seed(c["seed"]),
)
best = 1e9
state = None
stale = 0
history = []
for epoch in range(c["max_epochs"]):
    model.train()
    for x, y in tl:
        opt.zero_grad()
        loss = lossfn(model(x.to(dev).transpose(1, 2)), y.to(dev))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(xv).to(dev).transpose(1, 2))
        vl = nn.functional.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(yv).to(dev)
        ).item()
    history += [{"epoch": epoch + 1, "validation_loss": vl}]
    print(history[-1], flush=True)
    if vl < best:
        best = vl
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        stale = 0
    else:
        stale += 1
        if stale >= c["patience"]:
            break
model.load_state_dict(state)
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(xv).to(dev).transpose(1, 2))
p = torch.sigmoid(logits).cpu().numpy()
validation = metrics(yv, p)
validation["by_cohort"] = {
    cohort: metrics(yv[cv == cohort], p[cv == cohort]) for cohort in sorted(set(cv))
}
z.output.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": state, "config": c}, z.output / "model.pt")
for n, v in {
    "normalization.json": {"mean": mean.tolist(), "std": std.tolist()},
    "validation_metrics.json": validation,
    "training_manifest.json": {
        "device": str(dev),
        "train_episodes": len(set(it)),
        "validation_episodes": len(set(iv)),
        "history": history,
        "sampler": "natural_training_distribution",
        "threshold_policy": "minimum_fpr_at_recall_gte_0.90",
        "test_split_opened": False,
    },
}.items():
    (z.output / n).write_text(json.dumps(v, indent=2))
if z.push_to_hub:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(z.push_to_hub, private=True, exist_ok=True)
    api.upload_folder(repo_id=z.push_to_hub, folder_path=z.output)
