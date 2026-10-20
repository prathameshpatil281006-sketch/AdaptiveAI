"""Step 69 (I-19) static regression test: no duplicated import blocks.

Guards app/monitoring/monitoring_service.py against the exact
duplicate-import regression. No runtime behavior involved.
"""

from pathlib import Path

from app.core.config import get_project_root


def test_no_duplicated_import_statements():
    source = (
        Path(get_project_root())
        / "app"
        / "monitoring"
        / "monitoring_service.py"
    ).read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("#")
    ]
    imports = [line for line in lines if line.startswith("import ") or line.startswith("from ")]
    assert len(imports) == len(set(imports)), f"Duplicated imports found: {imports}"
