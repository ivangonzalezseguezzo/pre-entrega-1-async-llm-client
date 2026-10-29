"""Cliente asíncrono para OpenAI (Chat Completions) usando `AsyncOpenAI`."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient, classify_sdk_exception
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Usage


class OpenAIClient(BaseLLMClient):
    provider = "openai"
    api_key_env = "OPENAI_API_KEY"
    # Otros proveedores con API compatible con OpenAI (p. ej. Gemini) solo cambian esta URL.
    base_url: ClassVar[str | None] = None

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        assert config.api_key is not None
        # max_retries=0: los reintentos los maneja BaseLLMClient para tener control y logs propios.
        self._client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=self.base_url,
            timeout=config.timeout,
            max_retries=0,
        )

    def _request_kwargs(self, messages: list[ChatMessage]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump(mode="json") for m in messages],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        return kwargs

    async def _generate_once(self, messages: list[ChatMessage]) -> ModelResponse:
        response = await self._client.chat.completions.create(**self._request_kwargs(messages))
        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return ModelResponse(
            provider=self.config.provider,
            model=response.model or self.model,
            content=choice.message.content or "",
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    async def _stream_once(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(**self._request_kwargs(messages), stream=True)
        async for chunk in stream:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                yield delta

    def _classify_error(self, exc: Exception) -> LLMError:
        return classify_sdk_exception(exc, openai)

    async def close(self) -> None:
        await self._client.close()
