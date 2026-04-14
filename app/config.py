from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main app settings declaration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = Field(description="Async DB connection string.")
    APP_TITLE: str = Field(default="DEMO API")
    MAX_PROMPT_LENGTH: int = Field(default=5000, ge=1)
    API_KEY_HEADER_NAME: str = Field(default="X-API-Key")
    LLM_MODE: str = Field(default="mock", description="LLM runtime mode: mock or real.")
    LLM_PROVIDER: str = Field(
        default="groq", description="External LLM provider identifier."
    )
    LLM_API_KEY: str = Field(default="", description="External LLM provider API key.")
    LLM_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Configured LLM model name.",
    )
    LLM_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Optional base URL for compatible LLM APIs.",
    )
    CORS_ALLOW_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )


@lru_cache
def get_settings() -> Settings:
    """Returns app settings instance."""
    return Settings()
