from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


class User(Base):
    __tablename__ = "user"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(), primary_key=True, default=uuid.uuid4, comment="Primary key."
    )

    username: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, comment="User`s name."
    )

    email: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, comment="User`s email."
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        comment="Account creation date.",
    )

    api_keys: Mapped[list["APIKey"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    chat_history: Mapped[list["ChatHistory"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    chat_sessions:Mapped[list["ChatSession"]] = relationship(
        back_populates = "user",
        cascade = "all, delete-orphan",
        lazy = "selectin",
    )

class ChatSession(Base):
    __tablename__ = "chat_session"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="Primary key."
    )

    title: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Chat session title."
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(), 
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        index=True,
        comment="Chat session creation date.",
    )

    user: Mapped["User"] = relationship(back_populates="chat_sessions")

    chat_history: Mapped[list["ChatHistory"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="Primary key."
    )

    user_prompt: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Request to the API."
    )

    assistant_prompt: Mapped[str] = mapped_column(
        Text, nullable=False, comment="LLM model response."
    )

    messages: Mapped[list[dict[str, str]]] = mapped_column(
        JSONB, nullable=False, default=list, comment="Full chat messages payload."
    )

    temperature: Mapped[float] = mapped_column(
        Float, nullable=False, comment="LLM model creativity param."
    )

    max_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=100, comment="LLM response length limit."
    )

    streamed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Was response streamed."
    )

    response_metadata: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, default=dict, comment="Extra generation metadata."
    )

    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(),
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    api_key_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("api_key.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chat_session.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        index=True,
        comment="Chat completion creation date.",
    )

    user: Mapped[Optional["User"]] = relationship(back_populates="chat_history")
    api_key: Mapped[Optional["APIKey"]] = relationship(back_populates="chat_history")
    #add new orm relationship to ChatSession,once you get records of chat history, you can easily get the session info as well
    session: Mapped["ChatSession"] = relationship(back_populates="chat_history")



class APIKey(Base):
    __tablename__ = "api_key"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="Primary key."
    )

    name: Mapped[str] = mapped_column(Text, nullable=False, comment="Key name.")

    token: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
        index=True,
        comment="Opaque API key token.",
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(), ForeignKey("user.id", ondelete="CASCADE")
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        comment="Account creation date.",
    )

    owner: Mapped["User"] = relationship(back_populates="api_keys")

    chat_history: Mapped[list["ChatHistory"]] = relationship(
        back_populates="api_key",
        lazy="selectin",
    )
