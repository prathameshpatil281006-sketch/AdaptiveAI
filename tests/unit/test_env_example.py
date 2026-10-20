"""Step 72 (I-22) focused tests: .env.example matches supported config.

Guards that every process-environment variable the application reads
is documented in .env.example (and vice versa), so the reference
cannot drift from the code. No runtime behavior involved.
"""

import re
from pathlib import Path

import pytest

from app.core.config import get_project_root
from app.api import http_client as api_client


def _example_assignments():
    text = (Path(get_project_root()) / ".env.example").read_text(encoding="utf-8")
    active, commented = {}, set()
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.*)$", stripped)
        if m:
            active[m.group(1)] = m.group(2).strip()
            continue
        m = re.match(r"^#\s*([A-Z][A-Z0-9_]*)\s*=", stripped)
        if m:
            commented.add(m.group(1))
    return active, commented


def _code_env_reads():
    """Process-env variables the app reads (getenv/environ.get only).

    Resolves module-level ``NAME = "LITERAL"`` aliases (e.g.
    ``API_URL_ENV_VAR = "ADAPTIVE_AI_API_URL"``) so indirect reads are
    attributed to the real variable. Writes such as
    os.environ["X"] = ... are intentionally excluded: they are not
    developer-configurable inputs.
    """
    root = Path(get_project_root())
    reads = {}
    call_re = re.compile(
        r"os\.(?:getenv|environ\.get)\(\s*(?:[\"']([A-Z][A-Z0-9_]*)[\"']|([A-Za-z_][A-Za-z0-9_]*))\s*(,\s*(.*?))?\)"
    )
    alias_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[\"']([A-Z][A-Z0-9_]*)[\"']\s*$")
    for base in ("app", "frontend"):
        for path in (root / base).rglob("*.py"):
            aliases = {}
            text_lines = path.read_text(encoding="utf-8").splitlines()
            for line in text_lines:
                m = alias_re.match(line.strip())
                if m:
                    aliases[m.group(1)] = m.group(2)
            for line in text_lines:
                for m in call_re.finditer(line):
                    name = m.group(1) or aliases.get(m.group(2), m.group(2))
                    if name and re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                        reads.setdefault(name, line.strip())
    return reads


def test_documented_variables_match_code():
    active, commented = _example_assignments()
    reads = _code_env_reads()
    assert set(reads) == {"HOUSE_PRICE_MODEL_NAME", "HOUSE_PRICE_MODELS_DIR", "ADAPTIVE_AI_API_URL"}
    assert set(active) | commented == set(reads)


def test_documented_defaults_match_code():
    active, _ = _example_assignments()
    reads = _code_env_reads()
    # Direct getenv defaults are visible on the scanned call line.
    assert active["HOUSE_PRICE_MODEL_NAME"] == "LinearRegression"
    assert '"LinearRegression"' in reads["HOUSE_PRICE_MODEL_NAME"]
    # MODELS_DIR is optional-unset: code default is None (no literal default).
    assert "HOUSE_PRICE_MODELS_DIR" not in active
    assert re.search(r'os\.getenv\("HOUSE_PRICE_MODELS_DIR"\)\s*$', reads["HOUSE_PRICE_MODELS_DIR"])
    # ADAPTIVE_AI_API_URL is read via alias with `or DEFAULT_API_BASE_URL`
    # fallback; its default is asserted against the module constant.
    assert active["ADAPTIVE_AI_API_URL"] == "http://127.0.0.1:8000"
    assert api_client.DEFAULT_API_BASE_URL == "http://127.0.0.1:8000"


def test_dashboard_default_url_without_env(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_AI_API_URL", raising=False)
    assert api_client.get_api_base_url() == "http://127.0.0.1:8000"
    assert api_client.DEFAULT_API_BASE_URL == "http://127.0.0.1:8000"


def test_example_has_no_secrets_or_absolute_paths():
    active, _ = _example_assignments()
    for name, value in active.items():
        lowered = f"{name}={value}".lower()
        for forbidden in ("password", "secret", "token", "api_key", "apikey"):
            assert forbidden not in lowered, f"suspicious entry: {name}"
        assert not re.match(r"^[A-Za-z]:[\\/]", value), f"machine path: {name}"
        assert not value.startswith("/"), f"absolute path: {name}"
