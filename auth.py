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

from db import supabase


def get_current_user(authorization: str | None = Header(None)) -> dict:
    """FastAPI dependency: resolve the caller from the `Authorization: Bearer <jwt>` header."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Empty bearer token")

    try:
        res = supabase.auth.get_user(token)
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
