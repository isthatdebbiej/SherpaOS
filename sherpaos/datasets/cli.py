"""Typer commands for the dataset pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from sherpaos.datasets.generate import generate_dataset
from sherpaos.datasets.manifest import write_checksums
from sherpaos.datasets.split import build_split_manifest
from sherpaos.datasets.validate import DatasetValidationError, validate_dataset

data_app = typer.Typer(no_args_is_help=True, help="Generate and validate risk datasets.")


@data_app.command("generate")
def data_generate(
    matrix: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    episodes: Annotated[int, typer.Option()] = 200,
    output: Annotated[Path, typer.Option()] = Path("artifacts/datasets/latest"),
) -> None:
    """Generate deterministic controller-only dataset shards."""
    try:
        report = generate_dataset(matrix, episodes, output)
    except (OSError, ValueError) as exc:
        typer.echo(json.dumps({"status": "RED", "error": str(exc)}, indent=2))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"status": "GENERATED", **report}, indent=2))


@data_app.command("validate")
def data_validate(
    dataset: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Validate dataset integrity, separation, shape, rates, and splits."""
    try:
        report = validate_dataset(dataset)
    except (OSError, DatasetValidationError, ValueError) as exc:
        typer.echo(json.dumps({"status": "RED", "error": str(exc)}, indent=2))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2))


@data_app.command("split")
def data_split(
    dataset: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/splits.yaml"
    ),
) -> None:
    """Write deterministic scenario-group split membership."""
    try:
        manifest = build_split_manifest(dataset, config)
        write_checksums(dataset)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(json.dumps({"status": "RED", "error": str(exc)}, indent=2))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"status": "SPLIT", **manifest}, indent=2))
