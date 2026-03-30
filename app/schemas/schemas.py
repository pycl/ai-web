from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class APIResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Message(BaseModel):
    message: str = Field(min_length=1)
    role: Literal["system", "user", "assistant"]

    @field_validator("message")
    @classmethod
    def check_message_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message should not be empty.")
        return cleaned


class ChatRequest(BaseModel):
    #required session_id to link chat history to a specific chat session
    session_id: int
    messages: list[Message]
    temperature: float = Field(default=0.8, le=2.0, ge=0.0)
    max_tokens: int = Field(default=100, le=10000, ge=10)

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatRequest":
        if not self.messages:
            raise ValueError("Messages should not be empty.")
        if self.messages[-1].role != "user":
            raise ValueError(
                "User message is required to be the last message in chat sequence."
            )
        return self

    @computed_field
    @property
    def message_count(self) -> int:
        return len(self.messages)


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(min_length=5, max_length=50)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Username should not be empty.")
        return cleaned

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if "@" not in cleaned or "." not in cleaned.split("@")[-1]:
            raise ValueError("Email should look like a valid address.")
        return cleaned


class UserResponse(APIResponseModel):
    id: uuid.UUID
    username: str
    email: str
    created_at: datetime


class ChatSessionCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Chat session title should not be empty.")
        return cleaned
    

class ChatSessionResponse(APIResponseModel):
    id: int
    title: str
    user_id: uuid.UUID
    created_at: datetime


class APIKeyCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=100)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("API key name should not be empty.")
        return cleaned


class APIKeyResponse(APIResponseModel):
    id: int
    name: str
    owner_id: uuid.UUID
    created_at: datetime


class APIKeyCreatedResponse(APIKeyResponse):
    token: str


class ChatHistoryResponse(APIResponseModel):
    id: int
    user_prompt: str
    assistant_prompt: str
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int
    streamed: bool
    response_metadata: dict[str, object]
    user_id: Optional[uuid.UUID]
    api_key_id: Optional[int]
    session_id: int
    created_at: datetime


class ChatResponse(BaseModel):
    id: int
    user_id: uuid.UUID
    response: str
    temperature: float
    max_tokens: int
    model_name: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    database: Literal["ok"]
