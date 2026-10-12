"""
tests/test_api_admin_auth.py -- API-004: cache-mutating admin routes need an
admin scope/role claim or a subject listed in MERIDIAN_ADMIN_SUBJECTS.

Drives the real router with real signed JWTs over FastAPI's TestClient (the
unit handlers alone would not catch a dependency that never registered). No
database: only the in-process ScoreCache is touched.

Backward compatibility is part of the contract: with MERIDIAN_ADMIN_SUBJECTS
unset every authenticated caller is allowed (the Dagster
composite_cache_warm_sensor holds an ordinary token), with one WARNING per
process.
"""

import logging

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

import api.auth as auth
import config
from api.cache import ScoreCache
from api.routers import admin

ROUTES = [
    ("/cache/invalidate", {}),
    ("/admin/cache/warm", {"date": "2026-09-09"}),
]


def _token(**claims) -> str:
    payload = {"sub": "someone", **claims}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def _hdr(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv(auth.ADMIN_SUBJECTS_ENV, raising=False)
    monkeypatch.setattr(auth, "_open_admin_warned", False)
    app = FastAPI()
    app.include_router(admin.router)
    app.state.cache = ScoreCache(ttl_seconds=3600)
    app.state.cache.set("AAPL|2026-09-09|x", pd.DataFrame({"a": [1]}))
    return TestClient(app)


@pytest.mark.parametrize("path,params", ROUTES)
class TestEnforced:
    def test_non_admin_gets_403_when_subjects_configured(self, client, monkeypatch, path, params):
        monkeypatch.setenv(auth.ADMIN_SUBJECTS_ENV, "ops,dagster-sensor")
        r = client.post(path, params=params, headers=_hdr(_token(sub="qc-consumer")))
        assert r.status_code == 403
        # nothing was flushed
        assert client.app.state.cache.stats()["entries"] == 1

    def test_listed_subject_allowed(self, client, monkeypatch, path, params):
        monkeypatch.setenv(auth.ADMIN_SUBJECTS_ENV, " ops , dagster-sensor ")
        r = client.post(path, params=params, headers=_hdr(_token(sub="dagster-sensor")))
        assert r.status_code == 200

    @pytest.mark.parametrize(
        "claims",
        [
            {"scope": "read admin"},
            {"scope": ["admin"]},
            {"scopes": ["admin"]},
            {"role": "admin"},
            {"roles": ["viewer", "admin"]},
        ],
    )
    def test_admin_claim_allowed_even_when_not_listed(
        self, client, monkeypatch, path, params, claims
    ):
        monkeypatch.setenv(auth.ADMIN_SUBJECTS_ENV, "ops")
        r = client.post(path, params=params, headers=_hdr(_token(sub="other", **claims)))
        assert r.status_code == 200

    @pytest.mark.parametrize(
        "claims",
        [
            {"scope": "administrator"},
            {"role": "readonly"},
            {"role": "not admin"},
            {"role": "site admin"},
            {"roles": ["site admin"]},
        ],
    )
    def test_near_miss_claims_are_not_admin(self, client, monkeypatch, path, params, claims):
        monkeypatch.setenv(auth.ADMIN_SUBJECTS_ENV, "ops")
        r = client.post(path, params=params, headers=_hdr(_token(**claims)))
        assert r.status_code == 403

    def test_successful_admin_call_is_logged(self, client, monkeypatch, caplog, path, params):
        monkeypatch.setenv(auth.ADMIN_SUBJECTS_ENV, "ops")
        with caplog.at_level("INFO", logger=auth.logger.name):
            r = client.post(path, params=params, headers=_hdr(_token(sub="ops")))
        assert r.status_code == 200
        assert any("admin route allowed" in m and "ops" in m for m in caplog.messages)

    def test_bad_token_is_still_401(self, client, monkeypatch, path, params):
        monkeypatch.setenv(auth.ADMIN_SUBJECTS_ENV, "ops")
        r = client.post(path, params=params, headers=_hdr("not-a-jwt"))
        assert r.status_code == 401


@pytest.mark.parametrize("path,params", ROUTES)
class TestBackwardCompatible:
    @pytest.mark.parametrize("raw", [None, "", " , "])
    def test_ordinary_token_allowed_when_unconfigured(
        self, client, monkeypatch, caplog, path, params, raw
    ):
        if raw is not None:
            monkeypatch.setenv(auth.ADMIN_SUBJECTS_ENV, raw)
        with caplog.at_level(logging.WARNING, logger="api.auth"):
            r = client.post(path, params=params, headers=_hdr(_token(sub="dagster-sensor")))
        assert r.status_code == 200
        assert any(auth.ADMIN_SUBJECTS_ENV in rec.getMessage() for rec in caplog.records)


def test_open_mode_warns_once_per_process(client, caplog):
    with caplog.at_level(logging.WARNING, logger="api.auth"):
        for _ in range(3):
            assert client.post("/cache/invalidate", headers=_hdr(_token())).status_code == 200
        assert client.post("/admin/cache/warm", headers=_hdr(_token())).status_code == 200
    warnings = [r for r in caplog.records if r.name == "api.auth" and r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_admin_claim_does_not_trigger_open_mode_warning(client, caplog):
    with caplog.at_level(logging.WARNING, logger="api.auth"):
        r = client.post("/cache/invalidate", headers=_hdr(_token(scope="admin")))
    assert r.status_code == 200
    assert not [rec for rec in caplog.records if rec.name == "api.auth"]


def test_read_routes_are_not_admin_gated(client, monkeypatch):
    monkeypatch.setenv(auth.ADMIN_SUBJECTS_ENV, "ops")
    assert client.get("/health").status_code == 200


def test_create_token_scope_round_trips():
    tok = auth.create_token("ops", scope="admin")
    claims = jwt.decode(tok, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
    assert claims["sub"] == "ops" and claims["scope"] == "admin"
    assert "scope" not in jwt.decode(
        auth.create_token("plain"), config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM]
    )
