"""Cliente asíncrono para Anthropic (Messages API) usando `AsyncAnthropic`."""

from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLMClient, classify_sdk_exception
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Role, Usage


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        assert config.api_key is not None
        self._client = AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout,
            max_retries=0,
        )

    def _request_kwargs(self, messages: list[ChatMessage]) -> dict[str, Any]:
        # Anthropic recibe el system prompt como parámetro aparte, no como mensaje.
        # La Messages API actual ya no acepta parámetros de muestreo (temperature / top_p),
        # así que esos campos de ModelConfig solo se aplican en OpenAI.
        system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "messages": [m.model_dump(mode="json") for m in messages if m.role != Role.SYSTEM],
        }
        if system:
            kwargs["system"] = system
        return kwargs

    async def _generate_once(self, messages: list[ChatMessage]) -> ModelResponse:
        response = await self._client.messages.create(**self._request_kwargs(messages))
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(
            provider="anthropic",
            model=response.model,
            content=text,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason,
        )

    async def _stream_once(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._request_kwargs(messages)) as stream:
            async for text in stream.text_stream:
                yield text

    def _classify_error(self, exc: Exception) -> LLMError:
        return classify_sdk_exception(exc, anthropic)

    async def close(self) -> None:
        await self._client.close()
