"""Guard the standalone repository boundary."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "hybrid_detector"
TEXT_EXTENSIONS = {".py", ".json", ".md", ".toml", ".cff", ".html"}


def test_runtime_imports_are_package_local_or_third_party():
    forbidden_roots = {
        "lero" + "bot",
        "metro" + "logy",
        "hikon_cube_" + "tracking_offline",
        "cube_" + "tracking",
    }
    violations = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                root = module.split(".", 1)[0]
                if root in forbidden_roots:
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")
    assert not violations, "\n".join(violations)


def test_packaged_text_has_no_parent_repository_paths():
    forbidden = (
        "third_party/open" + "cv_kalibr",
        "hikon_cube_" + "tracking_offline",
        "/home/hanyu/Codes/" + "lero" + "bot",
    )
    violations = []
    paths = (
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS and ".git" not in path.parts
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)} contains {token!r}")
    assert not violations, "\n".join(violations)
