"""
auth.py — Authentication & Authorisation
=========================================

This module handles everything related to proving who a user is and what
they are allowed to do.

Architecture:
    1. Password hashing   — we never store plain-text passwords.
    2. JWT token creation — after login, the server issues a signed token.
    3. Token verification — every protected route decodes the token to find
                            out who is making the request.
    4. FastAPI dependency — get_current_user() plugs into FastAPI's
                            Depends() system and is reused across routes.

Flow:
    User sends username + password → /auth/token
    → verify_password()
    → create_access_token()
    → client receives JWT
    → client sends JWT in every subsequent request header
    → get_current_user() decodes and validates the JWT
    → route handler receives the authenticated User object

Why JWT over sessions?
    JWT is stateless: the server does not need to store session state.
    Each token is self-contained and cryptographically signed.
    This scales to multiple server instances without shared session storage.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings
from app.schemas import TokenData, UserInDB

# ─────────────────────────────────────────────────────────────────────────────
# Password Hashing
# ─────────────────────────────────────────────────────────────────────────────

# CryptContext manages the hashing algorithm.
# "bcrypt" is the industry standard for password hashing:
#   - Intentionally slow (adaptive cost factor)
#   - Salted automatically
#   - Never the same hash twice for the same password
# deprecated="auto" means old hashes are automatically upgraded on next login.
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
)


def hash_password(plain_password: str) -> str:
    """
    What does this function do?
    Converts a plain-text password into a bcrypt hash.
    The hash is what gets stored in Supabase — never the original password.

    Why bcrypt?
    Bcrypt is a one-way function. Even if someone reads the database,
    they cannot reverse the hash back into the original password.
    The cost factor (work factor) makes brute-force attacks impractical.
    """
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    What does this function do?
    Checks whether a plain-text password matches a stored bcrypt hash.

    How does it work without reversing the hash?
    bcrypt hashing is deterministic given the same salt (embedded in the hash).
    verify() re-hashes the plain password with the embedded salt and compares.
    """
    return pwd_context.verify(plain_password, hashed_password)


# ─────────────────────────────────────────────────────────────────────────────
# JWT Token Management
# ─────────────────────────────────────────────────────────────────────────────

# OAuth2PasswordBearer tells FastAPI:
#   "Clients authenticate by sending a Bearer token in the Authorization header.
#    The token is obtained by POSTing credentials to /auth/token."
# FastAPI uses this to generate the correct OpenAPI security schema automatically.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    What does this function do?
    Creates a signed JWT token containing the provided payload data.

    The token has three parts (separated by dots):
      Header   — algorithm type (HS256)
      Payload  — claims: username, user_id, role, expiry
      Signature — HMAC-SHA256 of header+payload using our secret key

    Why sign the token?
    The signature prevents tampering. If a user changes their role to
    "admin" in the token payload, the signature check will fail.

    Parameters:
        data:          Dict to embed in the token (username, user_id, role)
        expires_delta: How long until the token expires. Defaults to
                       jwt_expire_minutes from settings.
    """
    to_encode = data.copy()

    # Calculate the expiry timestamp
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_expire_minutes
        )

    # "exp" is the standard JWT claim for expiry — the jose library checks it
    to_encode["exp"] = expire

    # Sign and encode the token
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """
    What does this function do?
    Decodes a JWT token and returns the embedded claims as a TokenData object.

    Raises HTTPException if:
      - The token has been tampered with (invalid signature)
      - The token has expired
      - The token is missing required claims

    This is called on every protected request.
    """
    # This exception is raised if decoding fails for any reason
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # decode() verifies: signature, expiry, and algorithm
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )

        # Extract the username claim ("sub" is the standard JWT subject field)
        username: Optional[str] = payload.get("sub")

        if username is None:
            raise credentials_exception

        return TokenData(
            username=username,
            user_id=payload.get("user_id"),
            role=payload.get("role"),
        )

    except JWTError:
        raise credentials_exception


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Dependency: get_current_user
# ─────────────────────────────────────────────────────────────────────────────


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
    What does this function do?
    FastAPI dependency that decodes the Bearer token and returns the
    corresponding user from Supabase.

    Usage in a route:
        @app.post("/query")
        async def query(request: QueryRequest, user: UserInDB = Depends(get_current_user)):
            ...

    If the token is missing, expired, or invalid, FastAPI automatically
    returns a 401 Unauthorized response — the route handler is never called.

    This function is the security gatekeeper for all protected endpoints.
    """
    # Decode and validate the token
    token_data = decode_token(token)

    # Look up the user in Supabase by username
    # We import here (not at top) to avoid circular imports between
    # auth.py and database.py
    from app.database import get_user_by_username

    user = await get_user_by_username(token_data.username)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    return user


async def get_admin_user(
    current_user: UserInDB = Depends(get_current_user),
) -> UserInDB:
    """
    What does this function do?
    A stricter dependency that requires the user to have the 'admin' role.

    Usage in a route that only admins can access:
        @app.post("/ingest")
        async def ingest(user: UserInDB = Depends(get_admin_user)):
            ...

    If the user's role is not 'admin', returns 403 Forbidden.
    """
    if current_user.role not in ("admin", "auditor"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions — admin or auditor role required",
        )
    return current_user
