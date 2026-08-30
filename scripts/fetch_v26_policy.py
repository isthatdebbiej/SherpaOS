"""Download and checksum-verify the frozen G1 v26 ONNX policy."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from sherpaos.sim.v26_playground import POLICY_FILENAME, POLICY_SHA256, POLICY_URL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("var/policies/v26") / POLICY_FILENAME)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    urllib.request.urlretrieve(POLICY_URL, temporary)  # noqa: S310 - pinned HTTPS artifact
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != POLICY_SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"policy checksum mismatch: expected {POLICY_SHA256}, got {digest}")
    temporary.replace(args.output)
    print(f"verified {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
