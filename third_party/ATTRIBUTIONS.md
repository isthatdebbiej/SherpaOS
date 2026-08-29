# ATTRIBUTIONS.md — dependency, model, dataset, commit, license

| Dependency | Purpose | Commit/version | License |
|---|---|---|---|
| MuJoCo (`mujoco` PyPI package) | Physics simulation engine | see `pyproject.toml` pin | Apache-2.0 |
| google-deepmind/mujoco_menagerie | Unitree G1 MJCF model | `da76818e269b82289eba39808e2fb91d679d6994` (sparse checkout: `unitree_g1/` only, via `--filter=blob:none --sparse`, ~56MB) | Apache-2.0 (per-model license in menagerie repo) |
| NumPy / SciPy | Numerics | see `pyproject.toml` | BSD-3-Clause |
| Typer | CLI framework | see `pyproject.toml` | MIT |
| pytest / pytest-xdist / pytest-cov / Hypothesis | Test harness | see `pyproject.toml` | MIT / MPL-2.0 (Hypothesis) |
| Ruff | Lint/format | see `pyproject.toml` | MIT |
| ONNX / ONNX Runtime | Model export/parity (only if a learned model ships) | see `pyproject.toml` | Apache-2.0 / MIT |
| English Wikipedia (waypoint coordinates/elevations) | `configs/terrain/ebc_route.json` geographic-risk guard artifact | fetched 2026-08-29, see `configs/terrain/PROVENANCE.md` for per-waypoint URLs | CC BY-SA 4.0 (raw geographic facts — coordinates/elevations — are not independently copyrightable, but the source is credited per Wikipedia's terms) |

Update this table whenever a new dependency, pretrained model, or dataset is added, with
the exact pinned commit/version and license. This file must be accurate before
submission per `docs/plan.md` acceptance gates.
