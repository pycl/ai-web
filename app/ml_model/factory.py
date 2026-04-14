import logging

from app.config import Settings
from app.ml_model.groq_llm import GroqLLM
from app.ml_model.ml_model import BaseLLM, MockLLM

logger = logging.getLogger(__name__)


def create_llm(settings: Settings) -> BaseLLM:
    mode = settings.LLM_MODE.strip().lower()

    if mode == "mock":
        logger.info("LLM mode is `mock`; loading MockLLM.")
        return MockLLM()

    if mode == "real":
        provider = settings.LLM_PROVIDER.strip().lower()
        if provider != "groq":
            raise ValueError(
                f"Unsupported LLM_PROVIDER `{settings.LLM_PROVIDER}`. Expected `groq`."
            )
        if not settings.LLM_API_KEY.strip():
            raise ValueError("LLM_API_KEY must be set when LLM_MODE is `real`.")

        logger.info(
            "LLM mode is `real`; loading provider `%s` with model `%s`.",
            provider,
            settings.LLM_MODEL,
        )
        return GroqLLM(
            api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
        )

    raise ValueError(
        f"Unsupported LLM_MODE `{settings.LLM_MODE}`. Expected `mock` or `real`."
    )
