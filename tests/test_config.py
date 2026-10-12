"""
test_config.py - validation tests for config.py

Verifies the module-level names still exported by config.py (post live-fallback
sunset, #77) and that a missing JWT_SECRET raises at import time rather than
silently proceeding with an empty string.
"""

import importlib
import sys

import pytest

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _reload_config(monkeypatch, env_overrides: dict) -> object:
    """
    Re-import config with a controlled environment.

    `env_overrides` is applied on top of an env that already has
    JWT_SECRET set (so the base case does not fail). Pass `None` as
    a value to delete a variable.
    """
    base = {"JWT_SECRET": "test-secret-for-config-tests"}
    for k, v in env_overrides.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)

    for k, v in base.items():
        if k not in env_overrides:
            monkeypatch.setenv(k, v)

    sys.modules.pop("config", None)
    return importlib.import_module("config")


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------


class TestConfigExports:
    """The remaining module-level names must be accessible."""

    REQUIRED_NAMES = [
        "JWT_SECRET",
        "JWT_ALGORITHM",
        "JWT_EXPIRE_MINUTES",
        "API_SCORE_CACHE_TTL",
        "EXCLUDED_SECTORS",
    ]

    def test_all_required_names_exist(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        for name in self.REQUIRED_NAMES:
            assert hasattr(cfg, name), f"config.{name} must exist"

    def test_excluded_sectors_is_list(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert isinstance(cfg.EXCLUDED_SECTORS, list)

    def test_excluded_sectors_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"MERIDIAN_EXCLUDED_SECTORS": "Healthcare,Real Estate"})
        assert cfg.EXCLUDED_SECTORS == ["Healthcare", "Real Estate"]

    def test_jwt_algorithm_is_hs256(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.JWT_ALGORITHM == "HS256"

    def test_api_score_cache_ttl_positive(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {})
        assert cfg.API_SCORE_CACHE_TTL > 0


class TestConfigMissingJwtSecret:
    """Missing JWT_SECRET must raise at import time (fail-fast)."""

    def test_missing_jwt_secret_raises(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        # dotenv may supply a value; force empty string to trigger the guard
        monkeypatch.setenv("JWT_SECRET", "")
        sys.modules.pop("config", None)
        with pytest.raises((ValueError, SystemExit)):
            importlib.import_module("config")

    def test_present_jwt_secret_does_not_raise(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"JWT_SECRET": "some-secret"})
        assert cfg.JWT_SECRET == "some-secret"


class TestConfigEnvOverrides:
    """Environment variables must override the defaults."""

    def test_api_score_cache_ttl_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"API_SCORE_CACHE_TTL": "7200"})
        assert cfg.API_SCORE_CACHE_TTL == 7200
