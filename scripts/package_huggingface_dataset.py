"""Package immutable SherpaOS cohorts for upload to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from sherpaos.datasets.package import package_huggingface_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", nargs=2, metavar=("ID", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = [(cohort_id, Path(path)) for cohort_id, path in args.cohort]
    print(package_huggingface_collection(cohorts, args.output))


if __name__ == "__main__":
    main()
