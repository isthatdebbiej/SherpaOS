"""The critical sentinel test: no simulator ground truth in estimator/policy.

Per AGENTS.md safety constraint #1 and docs/CONTRACTS.md's "Leakage rule",
`sherpaos/estimator/` and `sherpaos/policy/` may only ever read fields that
exist on `sherpaos.contracts.RobotTelemetry`. They must never import
MuJoCo, `sherpaos.sim`, or `sherpaos.evaluation.ground_truth`, and must
never reference identifiers that look like simulator ground truth (e.g.
`true_friction`, `ground_truth`, `injected_fault`).

This is a static AST scan (not a substring grep) so that comments and
docstrings that legitimately *discuss* these forbidden concepts (this
module's own docstring, for instance) do not trip false positives.

If `sherpaos/estimator/` and `sherpaos/policy/` are empty or contain only
stub `__init__.py` files (because those lanes haven't landed yet), the
scan simply finds zero violations and passes -- it does not skip or error.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sherpaos

# ---------------------------------------------------------------------------
# Forbidden import / identifier rules
# ---------------------------------------------------------------------------

# Any of these exact dotted module names, or any submodule of them
# (`name == prefix` or `name.startswith(prefix + ".")`), are forbidden.
FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "mujoco",
    "sherpaos.sim",
    "sherpaos.evaluation.ground_truth",
)

# Identifiers that are forbidden regardless of casing: anything starting
# with "true_", plus these exact names.
_FORBIDDEN_IDENTIFIER_PREFIX = "true_"
FORBIDDEN_EXACT_IDENTIFIERS: frozenset[str] = frozenset(
    {"ground_truth", "injected_fault", "true_fall"}
)

LEAKAGE_SCAN_SUBDIRS: tuple[str, ...] = ("estimator", "policy")


def _sherpaos_root() -> Path:
    return Path(sherpaos.__file__).resolve().parent


def _is_forbidden_module(dotted_name: str) -> bool:
    if not dotted_name:
        return False
    return any(
        dotted_name == prefix or dotted_name.startswith(prefix + ".")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def _is_forbidden_identifier(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith(_FORBIDDEN_IDENTIFIER_PREFIX):
        return True
    return lowered in FORBIDDEN_EXACT_IDENTIFIERS


def _module_name_for_file(path: Path) -> tuple[str, bool]:
    """Return (dotted module name, is_package) for a .py file under sherpaos/."""
    repo_root = _sherpaos_root().parent
    rel_parts = list(path.relative_to(repo_root).with_suffix("").parts)
    is_package = rel_parts[-1] == "__init__"
    if is_package:
        rel_parts = rel_parts[:-1]
    return ".".join(rel_parts), is_package


def _resolve_import_from_module(
    current_module: str, is_package: bool, node: ast.ImportFrom
) -> str:
    """Resolve the dotted module an `ast.ImportFrom` node refers to.

    Mirrors importlib's relative-import resolution (PEP 328): `node.level`
    dots walk up from the *importing package* (the module itself if it is
    an `__init__.py`, otherwise its parent package).
    """
    if node.level == 0:
        return node.module or ""

    package = current_module if is_package else current_module.rsplit(".", 1)[0]
    bits = package.rsplit(".", node.level - 1) if package else [""]
    base = bits[0]
    if node.module:
        return f"{base}.{node.module}" if base else node.module
    return base


def scan_file_for_leakage(path: Path) -> list[str]:
    """Static-scan one .py file for forbidden imports/identifiers.

    Returns a list of human-readable violation descriptions (empty if clean).
    """
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: could not parse as Python ({exc})"]

    current_module, is_package = _module_name_for_file(path)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    violations.append(
                        f"{path}:{node.lineno}: forbidden import 'import {alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from_module(current_module, is_package, node)
            if resolved and _is_forbidden_module(resolved):
                src = f"{'.' * node.level}{node.module or ''}"
                violations.append(
                    f"{path}:{node.lineno}: forbidden import 'from {src} import ...' "
                    f"(resolves to module '{resolved}')"
                )
            for alias in node.names:
                full = f"{resolved}.{alias.name}" if resolved else alias.name
                if _is_forbidden_module(full):
                    violations.append(
                        f"{path}:{node.lineno}: forbidden import of '{full}' "
                        f"via 'from ... import {alias.name}'"
                    )
        elif isinstance(node, ast.Name):
            if _is_forbidden_identifier(node.id):
                violations.append(
                    f"{path}:{node.lineno}: forbidden identifier reference '{node.id}'"
                )
        elif isinstance(node, ast.Attribute):
            if _is_forbidden_identifier(node.attr):
                violations.append(
                    f"{path}:{node.lineno}: forbidden attribute access '.{node.attr}'"
                )
        elif isinstance(node, ast.arg):
            # Defense in depth beyond Name/Attribute: catches a forbidden
            # field smuggled in only as a function parameter name.
            if _is_forbidden_identifier(node.arg):
                violations.append(
                    f"{path}:{node.lineno}: forbidden parameter name '{node.arg}'"
                )
        elif isinstance(node, ast.keyword) and node.arg is not None:
            if _is_forbidden_identifier(node.arg):
                violations.append(
                    f"{path}:{node.lineno}: forbidden keyword argument name '{node.arg}'"
                )

    return violations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_no_ground_truth_leakage_in_estimator_and_policy():
    """AST-scan every .py file under sherpaos/estimator and sherpaos/policy.

    Passes trivially (zero violations) if those lanes haven't landed yet.
    """
    root = _sherpaos_root()
    all_violations: list[str] = []
    files_scanned: list[Path] = []

    for subdir in LEAKAGE_SCAN_SUBDIRS:
        target = root / subdir
        if not target.exists():
            continue
        for py_file in sorted(target.rglob("*.py")):
            files_scanned.append(py_file)
            all_violations.extend(scan_file_for_leakage(py_file))

    if all_violations:
        details = "\n".join(f"  - {v}" for v in all_violations)
        pytest.fail(
            f"Simulator ground-truth leakage detected in estimator/policy code "
            f"({len(all_violations)} violation(s) across {len(files_scanned)} file(s) scanned):\n"
            f"{details}"
        )


@pytest.mark.leakage
def test_dynamic_risk_estimator_avoids_ground_truth(make_telemetry):
    """Best-effort dynamic companion to the static scan above.

    If `sherpaos.estimator.risk` exists, construct its estimator entry
    point using only contract telemetry and confirm it runs and returns
    something. This never touches any evaluation/ground-truth object --
    only `make_telemetry`-built `RobotTelemetry` instances are passed in.

    Skips cleanly (does not fail) if the module or a matching API doesn't
    exist yet, so this test carries no weight until the estimator lane lands.
    """
    risk_module = pytest.importorskip("sherpaos.estimator.risk")

    estimator_cls = getattr(risk_module, "RiskEstimator", None)
    if estimator_cls is None:
        # Fall back to searching for anything estimator-shaped in the module.
        for attr_name in dir(risk_module):
            if attr_name.startswith("_"):
                continue
            candidate = getattr(risk_module, attr_name)
            if isinstance(candidate, type) and "estimator" in attr_name.lower():
                estimator_cls = candidate
                break

    if estimator_cls is None:
        pytest.skip("No RiskEstimator-like class found in sherpaos.estimator.risk yet")

    telemetry = make_telemetry()

    try:
        estimator = estimator_cls()
    except TypeError as exc:
        pytest.skip(f"{estimator_cls.__name__}() constructor doesn't match expected shape: {exc}")
        return

    result = None
    found_method = False
    for method_name in ("update", "estimate", "score", "__call__"):
        method = getattr(estimator, method_name, None)
        if not callable(method):
            continue
        found_method = True
        try:
            result = method(telemetry)
        except TypeError:
            continue
        else:
            break

    if not found_method:
        pytest.skip(
            f"Could not find an update/estimate/score/__call__ method on {estimator_cls.__name__}"
        )

    assert result is not None, (
        f"{estimator_cls.__name__} produced no output from a valid RobotTelemetry sample"
    )
