from __future__ import annotations

from pathlib import Path

from sherpaos.datasets.package import package_huggingface_collection


def test_package_rejects_non_green_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "quality_report.json").write_text('{"status":"RED"}')
    try:
        package_huggingface_collection([("bad", source)], tmp_path / "output")
    except ValueError as exc:
        assert "not GREEN" in str(exc)
    else:
        raise AssertionError("RED source was packaged")
