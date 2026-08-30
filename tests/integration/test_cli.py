from __future__ import annotations

from typer.testing import CliRunner

from sherpaos.cli.main import app


def test_preflight_command_is_green():
    result = CliRunner().invoke(app, ["preflight"])
    assert result.exit_code == 0, result.output
    assert '"status": "GREEN"' in result.output


def test_simulate_viewer_option_synchronizes_native_viewer(monkeypatch, tmp_path):
    class FakeViewer:
        def __init__(self) -> None:
            self.sync_count = 0
            self.closed = False
            self.is_running_calls = 0

        def sync(self) -> None:
            self.sync_count += 1

        def is_running(self) -> bool:
            self.is_running_calls += 1
            return self.is_running_calls == 1

        def close(self) -> None:
            self.closed = True

    viewer = FakeViewer()
    monkeypatch.setattr(
        "sherpaos.sim.runner.mujoco.viewer.launch_passive", lambda model, data: viewer
    )
    monkeypatch.setattr("sherpaos.sim.runner.time.sleep", lambda seconds: None)

    result = CliRunner().invoke(
        app,
        ["simulate", "--viewer", "--max-steps", "2", "--output", str(tmp_path / "run")],
    )

    assert result.exit_code == 0, result.output
    assert viewer.sync_count > 0
    assert viewer.is_running_calls == 2
    assert viewer.closed is True
