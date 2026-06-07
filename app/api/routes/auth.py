from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import auth_store, bearer_scheme, current_user
from app.schemas import AuthUser, LoginRequest, LoginResponse


router = APIRouter()


@router.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest) -> dict:
    user = auth_store.authenticate(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    session = auth_store.create_session(user["id"])
    return {"token": session["token"], "expiresAt": session["expiresAt"], "user": user}


@router.get("/auth/me", response_model=AuthUser)
def me(user: dict = Depends(current_user)) -> dict:
    return user


@router.post("/auth/logout")
def logout(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> dict:
    if credentials is not None:
        auth_store.revoke_session(credentials.credentials)
    return {"ok": True}
