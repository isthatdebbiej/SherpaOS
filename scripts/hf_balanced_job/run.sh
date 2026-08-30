#!/usr/bin/env bash
set -euo pipefail
cd /workspace
python - <<'PY'
import urllib.request
import zipfile
urllib.request.urlretrieve(
    "https://github.com/isthatdebbiej/SherpaOS/archive/52a8ed6.zip",
    "code.zip",
)
zipfile.ZipFile("code.zip").extractall(".")
PY
cd /workspace/SherpaOS-*
pip install --quiet huggingface_hub pyyaml
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "isthatdebbiej/sherpaos-himalayan-risk-400-balanced-v2",
    repo_type="dataset",
    local_dir="/workspace/data",
)
PY
PYTHONPATH=. python scripts/train_risk_model.py   --dataset /workspace/data   --output /workspace/output   --push-to-hub iteratehack/sherpaos-risk-tcn-balanced-v2
