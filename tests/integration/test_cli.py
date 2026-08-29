from __future__ import annotations

from typer.testing import CliRunner

from sherpaos.cli.main import app


def test_preflight_command_is_green():
    result = CliRunner().invoke(app, ["preflight"])
    assert result.exit_code == 0, result.output
    assert '"status": "GREEN"' in result.output
