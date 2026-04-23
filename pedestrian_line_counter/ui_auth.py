from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class UiAuthConfig:
    username: str = "admin"
    password: str = ""
    cookie_name: str = "edge_ui_session"
    cookie_max_age_s: int = 12 * 60 * 60
    login_rate_limit_max_failures: int = 5
    login_rate_limit_window_s: int = 5 * 60
    login_rate_limit_lockout_s: int = 10 * 60

    def enabled(self) -> bool:
        return bool(str(self.password or "").strip())

    def login_rate_limit_enabled(self) -> bool:
        return (
            int(self.login_rate_limit_max_failures) > 0
            and int(self.login_rate_limit_window_s) > 0
            and int(self.login_rate_limit_lockout_s) > 0
        )


@dataclass
class _LoginAttempt:
    failed_attempts: int = 0
    first_failed_at: float = 0.0
    locked_until: float = 0.0
    last_seen_at: float = 0.0


class LoginRateLimiter:
    def __init__(self, *, cfg: Optional[UiAuthConfig] = None) -> None:
        self._cfg = cfg if cfg is not None else UiAuthConfig()
        self._attempts: Dict[str, _LoginAttempt] = {}
        self._lock = Lock()

    def check_allowed(self, key: str) -> Tuple[bool, int]:
        if not self._cfg.login_rate_limit_enabled():
            return True, 0
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            attempt = self._attempts.get(key)
            if attempt is None:
                return True, 0
            if attempt.locked_until > now:
                return False, max(1, int(math.ceil(attempt.locked_until - now)))
            if attempt.first_failed_at and (now - attempt.first_failed_at) > int(self._cfg.login_rate_limit_window_s):
                self._attempts.pop(key, None)
            return True, 0
    
    def register_failure(self, key: str) -> Tuple[bool, int]:
        if not self._cfg.login_rate_limit_enabled():
            return False, 0
        now = time.time()
        window_s = int(self._cfg.login_rate_limit_window_s)
        lockout_s = int(self._cfg.login_rate_limit_lockout_s)
        max_failures = int(self._cfg.login_rate_limit_max_failures)
        with self._lock:
            self._prune_locked(now)
            attempt = self._attempts.get(key)
            if attempt is None or (attempt.first_failed_at and (now - attempt.first_failed_at) > window_s):
                attempt = _LoginAttempt(failed_attempts=1, first_failed_at=now, last_seen_at=now)
                self._attempts[key] = attempt
                return False, 0
            attempt.failed_attempts += 1
            attempt.last_seen_at = now
            if attempt.failed_attempts >= max_failures:
                attempt.locked_until = now + lockout_s
                return True, lockout_s          
            return False, 0
        
    def register_success(self, key: str) -> None:
        if not self._cfg.login_rate_limit_enabled():
            return
        with self._lock:
            self._attempts.pop(key, None)

    def _prune_locked(self, now: float) -> None:
        window_s = int(self._cfg.login_rate_limit_window_s)
        stale_keys = [
            key 
            for key, attempt in self._attempts.items()
            if (
                attempt.locked_until <= now
                and attempt.first_failed_at
                and (now - attempt.first_failed_at) > window_s
            )
        ]
        for key in stale_keys:
            self._attempts.pop(key, None)

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


__all__ = ["LoginRateLimiter", "UiAuthConfig", "issue_session_token", "validate_session_token"]
