"""Static AST Independence Test.

Ensures that no module inside app/ imports from phase20_standalone, experiments,
or external parent paths.
"""
from __future__ import annotations

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]

FORBIDDEN_KEYWORDS = [
    "phase20_standalone",
    "phase19",
    "phase18",
    "experiments",
    "preprocessed_",
    "speech_segments",
    "vad_aug_temp",
]


def test_zero_outside_imports():
    """Verify no python script in app/ imports from outside directories."""
    violations = []
    py_files = list(APP_DIR.rglob("*.py"))
    assert len(py_files) > 10, "Found too few python files to verify."

    for py_file in py_files:
        if ".venv" in py_file.parts:
            continue
        code = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(code, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.lower()
                    for f in FORBIDDEN_KEYWORDS:
                        if f in name:
                            violations.append((py_file.name, name))
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").lower()
                for f in FORBIDDEN_KEYWORDS:
                    if f in mod:
                        violations.append((py_file.name, mod))

    assert not violations, f"Forbidden external imports detected: {violations}"


def test_zero_escaping_relative_paths():
    """Verify no path literals escape the app/ directory via parent traversal."""
    violations = []
    escape_target = ".." + "/" + ".."
    target_mod = "phase20" + "_standalone"

    for py_file in APP_DIR.rglob("*.py"):
        if ".venv" in py_file.parts or py_file.name == "test_independence.py":
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            if escape_target in line or target_mod in line:
                violations.append((py_file.name, line_no, line.strip()))

    assert not violations, f"Escaping relative paths detected: {violations}"
