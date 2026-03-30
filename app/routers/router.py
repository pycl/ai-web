from __future__ import annotations

import logging
import secrets
from typing import Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Security,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import desc, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import app.main as main_app
from app.config import get_settings
from app.database.database import get_db
from app.ml_model.ml_model import MockLLM
from app.models.models import APIKey, ChatHistory, User,ChatSession
from app.schemas.schemas import (
    APIKeyCreatedResponse,
    APIKeyCreateRequest,
    APIKeyResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    UserCreateRequest,
    UserResponse,
    ChatSessionCreateRequest,
    ChatSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER_NAME, auto_error=False)
bearer_security = HTTPBearer(auto_error=False)


def get_llm() -> MockLLM:
    return main_app.ml_model_state["ml_model"]


async def get_current_api_key(
    db: AsyncSession = Depends(get_db),
    header_api_key: Optional[str] = Security(api_key_header),
    bearer_credentials: Optional[HTTPAuthorizationCredentials] = Security(
        bearer_security
    ),
) -> APIKey:
    token = header_api_key
    if token is None and bearer_credentials is not None:
        token = bearer_credentials.credentials

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Provide `{settings.API_KEY_HEADER_NAME}` header or Bearer token.",
        )

    stmt = (
        select(APIKey).options(selectinload(APIKey.owner)).where(APIKey.token == token)
    )
    api_key = (await db.execute(stmt)).scalar_one_or_none()
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not valid auth key.",
        )

    return api_key


async def get_user_or_404(
    user_id: UUID,
    db: AsyncSession,
    *,
    with_api_keys: bool = False,
) -> User:
    stmt = select(User).where(User.id == user_id)
    if with_api_keys:
        stmt = stmt.options(selectinload(User.api_keys))

    user = (await db.execute(stmt)).scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User was not found.",
        )
    return user


def ensure_user_access(user_id: UUID, api_key: APIKey) -> None:
    if api_key.owner_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not belong to requested user.",
        )


def schedule_chat_audit(
    chat_id: int,
    user_id: UUID,
    *,
    streamed: bool,
) -> None:
    logger.info(
        "Chat `%s` for user `%s` was stored. Streamed=%s",
        chat_id,
        user_id,
        streamed,
    )


def build_chat_metadata(
    request: ChatRequest, model: MockLLM, *, streamed: bool
) -> dict[str, object]:
    return {
        "model_name": model.model_name,
        "message_count": request.message_count,
        "streamed": streamed,
    }


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    await db.execute(text("SELECT 1"))
    return HealthResponse(
        status="ok",
        model_loaded="ml_model" in main_app.ml_model_state,
        database="ok",
    )


@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["users"],
)
async def create_user(
    request: UserCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> User:
    user = User(username=request.username, email=request.email)
    db.add(user)

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        logger.warning("User creation failed because of unique constraint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this username or email already exists.",
        ) from exc

    await db.refresh(user)
    return user


@router.get("/users/{user_id}", response_model=UserResponse, tags=["users"])
async def get_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> User:
    return await get_user_or_404(user_id, db)


@router.post(
    "/users/{user_id}/api-keys",
    response_model=APIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["api-keys"],
)
async def create_api_key(
    user_id: UUID,
    request: APIKeyCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> APIKey:
    user = await get_user_or_404(user_id, db)
    api_key = APIKey(
        name=request.name,
        token=secrets.token_urlsafe(32),
        owner_id=user.id,
    )
    db.add(api_key)

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        logger.warning("API key creation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not create API key. Please retry.",
        ) from exc

    await db.refresh(api_key)
    return api_key


@router.post(
    "/users/{user_id}/sessions",
    response_model = ChatSessionResponse,
    status_code = status.HTTP_201_CREATED,
    tags = ["chat"],
)
async def create_chat_session(
    user_id: UUID,
    request: ChatSessionCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatSession:
    user = await get_user_or_404(user_id, db)
    chat_session = ChatSession(
        title=request.title,
        user_id=user.id,
    )
    db.add(chat_session)

    await db.commit()
    await db.refresh(chat_session)
    return chat_session


@router.get(
    "/users/{user_id}/sessions",
    response_model=list[ChatSessionResponse],
    tags=["chat"],
)
async def list_chat_sessions(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key),
) -> list[ChatSession]:
    ensure_user_access(user_id, api_key)
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(desc(ChatSession.created_at))
    )
    return list((await db.execute(stmt)).scalars().all())

@router.get(
        "/users/{user_id}/sessions/{session_id}/chat-history",
        response_model=list[ChatHistoryResponse],
        tags=["chat"],
)

async def list_chat_session_history(
    user_id: UUID,
    session_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key),
) -> list[ChatHistory]:
    ensure_user_access(user_id, api_key)

    session_stmt = select(ChatSession).where(
        ChatSession.id == session_id,
        ChatSession.user_id == user_id,
    )
    chat_session = (await db.execute(session_stmt)).scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Chat session not found for this user.",
        )

    stmt = (
        select(ChatHistory)
        .where(
            ChatHistory.session_id == session_id,
            ChatHistory.user_id == user_id,
            )
        .order_by(desc(ChatHistory.created_at))
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


@router.get(
    "/users/{user_id}/api-keys",
    response_model=list[APIKeyResponse],
    tags=["api-keys"],
)
async def list_api_keys(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[APIKey]:
    await get_user_or_404(user_id, db)
    stmt = (
        select(APIKey)
        .where(APIKey.owner_id == user_id)
        .order_by(desc(APIKey.created_at))
    )
    return list((await db.execute(stmt)).scalars().all())


@router.get(
    "/users/{user_id}/chat-history",
    response_model=list[ChatHistoryResponse],
    tags=["chat"],
)
async def list_chat_history(
    user_id: UUID,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key),
) -> list[ChatHistory]:
    ensure_user_access(user_id, api_key)
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.user_id == user_id)
        .order_by(desc(ChatHistory.created_at))
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["chat"],
)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key),
    model: MockLLM = Depends(get_llm),
) -> ChatResponse:
    session_stmt = select(ChatSession).where(
        ChatSession.id == request.session_id,
        ChatSession.user_id == api_key.owner_id,
    )
    chat_session = (await db.execute(session_stmt)).scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Chat session not found for this user.",
        )

    user_prompt = request.messages[-1].message

    if len(user_prompt) > settings.MAX_PROMPT_LENGTH:
        raise main_app.ContextLengthExceeded(settings.MAX_PROMPT_LENGTH)

    response_text = await model.generate(
        prompt=user_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    chat_entry = ChatHistory(
        user_id=api_key.owner_id,
        api_key_id=api_key.id,
        session_id = chat_session.id,
        messages=[message.model_dump() for message in request.messages],
        user_prompt=user_prompt,
        assistant_prompt=response_text,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        streamed=False,
        response_metadata=build_chat_metadata(request, model, streamed=False),
    )
    db.add(chat_entry)
    await db.commit()
    await db.refresh(chat_entry)

    background_tasks.add_task(
        schedule_chat_audit,
        chat_entry.id,
        api_key.owner_id,
        streamed=False,
    )

    return ChatResponse(
        id=chat_entry.id,
        user_id=api_key.owner_id,
        response=response_text,
        temperature=chat_entry.temperature,
        max_tokens=chat_entry.max_tokens,
        model_name=model.model_name,
        created_at=chat_entry.created_at,
    )


@router.post("/chat/stream", tags=["chat"])
async def chat_streaming(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key),
    model: MockLLM = Depends(get_llm),
) -> StreamingResponse:
    session_stmt = select(ChatSession).where(
        ChatSession.id == request.session_id,
        ChatSession.user_id == api_key.owner_id,
    )
    chat_session = (await db.execute(session_stmt)).scalar_one_or_none()
    if chat_session is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "Chat session not found for this user.",
        )

    user_prompt = request.messages[-1].message

    if len(user_prompt) > settings.MAX_PROMPT_LENGTH:
        raise main_app.ContextLengthExceeded(settings.MAX_PROMPT_LENGTH)

    async def stream_response():
        collected_tokens: list[str] = []
        async for token in model.generate_stream(
            prompt=user_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            collected_tokens.append(token)
            yield token

        chat_entry = ChatHistory(
            user_id=api_key.owner_id,
            session_id = chat_session.id,
            api_key_id=api_key.id,
            messages=[message.model_dump() for message in request.messages],
            user_prompt=user_prompt,
            assistant_prompt="".join(collected_tokens).strip(),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            streamed=True,
            response_metadata=build_chat_metadata(request, model, streamed=True),
        )
        db.add(chat_entry)
        await db.commit()
        await db.refresh(chat_entry)
        schedule_chat_audit(chat_entry.id, api_key.owner_id, streamed=True)

    return StreamingResponse(stream_response(), media_type="text/plain")
