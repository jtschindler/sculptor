"""Convention (review §4): every public function, method and class in the
maintained packages has a reST docstring without placeholder fields."""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    [
        *(REPO_ROOT / "src" / "sculptor").glob("*.py"),
        *(REPO_ROOT / "sculptor_extensions" / "src" / "sculptor_extensions").glob(
            "*.py"
        ),
    ]
)


def _problems(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node)
        where = f"{path.name}:{node.lineno} {node.name}"
        if not doc:
            yield f"{where}: no docstring"
        elif re.search(r"^\s*:param \w+:\s*$", doc, re.MULTILINE):
            yield f"{where}: empty :param:"
        elif ":param type:" in doc or ":return type:" in doc:
            yield f"{where}: use :type <name>: / :rtype:"
        elif re.search(r"sculptor\.speconed\.|specfit\b|SpecFit", doc):
            yield f"{where}: refers to removed v1 API"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_public_docstrings(path):
    assert list(_problems(path)) == []
