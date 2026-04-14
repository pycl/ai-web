import asyncio
import logging
import time
from typing import Any, AsyncIterator, Protocol

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    def __init__(
        self,
        *,
        provider: str,
        message: str,
        status_code: int = 502,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class BaseLLM(Protocol):
    provider_name: str
    model_name: str

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        ...

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        ...


class MockLLM:
    provider_name = "mock"
    model_name = "MockLLM"

    def __init__(self) -> None:
        logger.info("[Weights are loading].")
        time.sleep(0.2)
        logger.info("[Weights are loaded].")
        self.semaphore = asyncio.Semaphore(2)

    @staticmethod
    def _build_response_tokens(
        prompt: str, temperature: float, max_tokens: int
    ) -> list[str]:
        prompt_tokens = prompt.split() or ["<empty>"]
        generated_tokens = [
            f"{MockLLM.model_name}[temp={temperature:.2f}]",
            "=>",
            *prompt_tokens,
        ]
        return generated_tokens[: max_tokens or 1]

    @staticmethod
    def _extract_prompt(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("message", "")
        return ""

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        prompt = self._extract_prompt(messages)
        tokens = self._build_response_tokens(prompt, temperature, max_tokens)

        async with self.semaphore:
            logger.info(
                f"Generating response on `prompt`: {prompt[:10]}... with `temperature`:{temperature}"
            )
            logger.info(f"Slots available: {2 - self.semaphore._value}/2.")
            await asyncio.sleep(0.2)
            return " ".join(tokens)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        prompt = self._extract_prompt(messages)
        tokens = self._build_response_tokens(prompt, temperature, max_tokens)

        async with self.semaphore:
            await asyncio.sleep(0.1)
            for token in tokens:
                await asyncio.sleep(0.05)
                yield f"{token} "
