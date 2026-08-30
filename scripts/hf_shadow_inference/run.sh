#!/usr/bin/env bash
set -euo pipefail
pip install --quiet fastapi uvicorn huggingface_hub numpy
uvicorn --app-dir /job app:api --host 0.0.0.0 --port 7860
