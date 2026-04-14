import json
import logging
from typing import AsyncIterator

import httpx

from app.ml_model.ml_model import LLMProviderError

logger = logging.getLogger(__name__)


class GroqLLM:
    provider_name = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text or "Upstream provider returned an invalid error response."

        error_data = payload.get("error")
        if isinstance(error_data, dict):
            message = error_data.get("message")
            if isinstance(message, str) and message.strip():
                return message

        if isinstance(error_data, str) and error_data.strip():
            return error_data

        return "Upstream provider request failed."

    def _build_status_error(
        self, exc: httpx.HTTPStatusError, *, action: str
    ) -> LLMProviderError:
        response = exc.response
        status_code = response.status_code
        message = self._extract_error_message(response)
        details = {
            "model_name": self.model_name,
            "upstream_status_code": status_code,
            "action": action,
        }
        return LLMProviderError(
            provider=self.provider_name,
            message=message,
            status_code=status_code,
            details=details,
        )

    def _build_request_error(
        self, exc: httpx.RequestError, *, action: str
    ) -> LLMProviderError:
        return LLMProviderError(
            provider=self.provider_name,
            message="Could not reach the upstream LLM provider.",
            status_code=502,
            details={
                "model_name": self.model_name,
                "action": action,
                "reason": str(exc),
            },
        )

    def _build_invalid_response_error(self, *, action: str) -> LLMProviderError:
        return LLMProviderError(
            provider=self.provider_name,
            message="Upstream provider returned an invalid response format.",
            status_code=502,
            details={
                "model_name": self.model_name,
                "action": action,
            },
        )

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, object]:
        safe_temperature = max(temperature, 1e-8)
        return {
            "model": self.model_name,
            "messages": [
                {"role": message["role"], "content": message["message"]}
                for message in messages
            ],
            "temperature": safe_temperature,
            "max_completion_tokens": max_tokens,
            "stream": stream,
        }

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self._build_status_error(exc, action="generate") from exc
        except httpx.RequestError as exc:
            raise self._build_request_error(exc, action="generate") from exc

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise self._build_invalid_response_error(action="generate") from exc

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data = line[6:].strip()
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content")
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                            raise self._build_invalid_response_error(
                                action="generate_stream"
                            ) from exc

                        if delta:
                            yield delta
        except httpx.HTTPStatusError as exc:
            raise self._build_status_error(exc, action="generate_stream") from exc
        except httpx.RequestError as exc:
            raise self._build_request_error(exc, action="generate_stream") from exc
