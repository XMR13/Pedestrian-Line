from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class UiAuthConfig:
    username: str = "admin"
    password: str = ""
    cookie_name: str = "edge_ui_session"
    cookie_max_age_s: int = 12 * 60 * 60

    def enabled(self) -> bool:
        return bool(str(self.password or "").strip())


def issue_session_token(*, cfg: UiAuthConfig, username: str) -> str:
    if not cfg.enabled():
        raise ValueError("Cannot issue UI session token without configured password")
    now = int(time.time())
    payload = {"u": username, "exp": now + int(cfg.cookie_max_age_s)}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(str(cfg.password).encode("utf-8"), raw, hashlib.sha256).digest()
    return f"{_b64url_encode(raw)}.{_b64url_encode(sig)}"


def validate_session_token(*, cfg: UiAuthConfig, token: str) -> bool:
    if not cfg.enabled():
        return True
    if not token or "." not in token:
        return False
    left, right = token.split(".", 1)
    try:
        raw = _b64url_decode(left)
        sig = _b64url_decode(right)
    except Exception:
        return False
    expected = hmac.new(str(cfg.password).encode("utf-8"), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, sig):
        return False
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("u") != cfg.username:
        return False
    exp = payload.get("exp")
    if not isinstance(exp, int):
        return False
    return int(time.time()) <= exp


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + pad).encode("ascii"))


__all__ = ["UiAuthConfig", "issue_session_token", "validate_session_token"]
