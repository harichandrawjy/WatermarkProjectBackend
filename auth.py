"""
Supabase-backed auth helpers for the Watermark API.

The Supabase Python SDK ships with `auth.sign_up`, `auth.sign_in_with_password`,
and `auth.get_user(jwt)`. We don't manage password hashes or JWT signing
ourselves — Supabase does all of that. Our only job is:

  1. Forward register/login requests to Supabase.
  2. On protected routes, read the Bearer token, ask Supabase who it belongs
     to, and pass the resolved user into the endpoint.
"""

from fastapi import HTTPException, Header

from db import supabase_auth


def get_current_user(authorization: str | None = Header(None)) -> dict:
    """FastAPI dependency: resolve the caller from the `Authorization: Bearer <jwt>` header."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Empty bearer token")

    try:
        res = supabase_auth.auth.get_user(token)
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")

    user = getattr(res, "user", None)
    if not user or not getattr(user, "id", None):
        raise HTTPException(401, "Token did not resolve to a user")

    return {
        "id":    user.id,
        "email": user.email or "",
        "token": token,
    }


def get_optional_user(authorization: str | None = Header(None)) -> dict | None:
    """Like `get_current_user`, but returns None instead of raising.

    For endpoints that are public by design yet want to recognise a caller when
    one is present.  /verify is the case this exists for: anyone holding the
    metadata can verify a file, with or without an account — but when they are
    signed in we record the run to their verification history.

    A bad or expired token is treated as "not signed in" rather than an error,
    so a stale token in localStorage can never break verification for someone
    who did not need to be signed in anyway.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        return get_current_user(authorization)
    except Exception:
        return None
