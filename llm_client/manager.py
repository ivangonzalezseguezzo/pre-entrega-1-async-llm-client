"""`AsyncLLMManager`: elige el proveedor según la configuración y expone una interfaz única."""

import logging
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient, MessagesInput, normalize_messages
from .fallback import FallbackClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Role

logger = logging.getLogger(__name__)

CLIENTS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


def create_client(config: ModelConfig) -> BaseLLMClient:
    """Factory: crea el cliente concreto que corresponde al proveedor de la configuración."""
    return CLIENTS[config.provider](config)


def config_from_env(provider: str | None = None, **overrides: Any) -> ModelConfig:
    """Arma un ModelConfig a partir de variables de entorno (ver .env.example).

    Pydantic convierte y valida los strings del entorno (p. ej. LLM_TEMPERATURE="0.7").
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    client_cls = CLIENTS.get(provider)
    data: dict[str, Any] = {
        "provider": provider,
        "model": os.getenv(f"{provider.upper()}_MODEL"),
        "api_key": os.getenv(client_cls.api_key_env) if client_cls else None,
        "temperature": os.getenv("LLM_TEMPERATURE"),
        "max_tokens": os.getenv("LLM_MAX_TOKENS"),
        "timeout": os.getenv("LLM_TIMEOUT"),
        "max_retries": os.getenv("LLM_MAX_RETRIES"),
    }
    data = {k: v for k, v in data.items() if v not in (None, "")}
    data.update(overrides)
    return ModelConfig.model_validate(data)


class AsyncLLMManager:
    """Fachada sobre los clientes concretos: el código que lo usa no sabe qué proveedor hay detrás.

    Con `fallbacks`, si el proveedor principal falla se prueba con los siguientes, en orden.
    """

    def __init__(self, config: ModelConfig, fallbacks: Sequence[ModelConfig] = ()) -> None:
        self.config = config
        primary = create_client(config)
        self.client: BaseLLMClient | FallbackClient = (
            FallbackClient([primary, *(create_client(c) for c in fallbacks)]) if fallbacks else primary
        )

    @classmethod
    def from_env(
        cls, provider: str | None = None, *, fallbacks: Sequence[str] | None = None, **overrides: Any
    ) -> "AsyncLLMManager":
        """Crea el manager desde el entorno.

        `fallbacks=None` lee LLM_FALLBACKS (p. ej. "anthropic,gemini"); `fallbacks=[]` los desactiva.
        Los `overrides` solo se aplican al proveedor principal. Los fallbacks sin API key se omiten.
        """
        config = config_from_env(provider, **overrides)
        if fallbacks is None:
            names = [p.strip().lower() for p in os.getenv("LLM_FALLBACKS", "").split(",") if p.strip()]
            fallbacks = [name for name in names if name != config.provider]
        fallback_configs = []
        for name in fallbacks:
            fallback_config = config_from_env(name)
            if fallback_config.api_key is None:
                logger.warning("Fallback '%s' omitido: falta %s", name, CLIENTS[name].api_key_env)
                continue
            fallback_configs.append(fallback_config)
        return cls(config, fallback_configs)

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.client.model

    async def generate(self, messages: MessagesInput, *, system: str | None = None) -> ModelResponse:
        return await self.client.generate(self._build(messages, system))

    def stream(self, messages: MessagesInput, *, system: str | None = None) -> AsyncIterator[str]:
        return self.client.stream(self._build(messages, system))

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> "AsyncLLMManager":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @staticmethod
    def _build(messages: MessagesInput, system: str | None) -> list[ChatMessage]:
        chat = normalize_messages(messages)
        if system:
            chat.insert(0, ChatMessage(role=Role.SYSTEM, content=system))
        return chat
