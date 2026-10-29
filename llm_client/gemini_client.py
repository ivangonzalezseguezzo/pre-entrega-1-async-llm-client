"""Cliente asíncrono para Google Gemini a través de su endpoint compatible con OpenAI.

Gemini acepta el mismo formato que Chat Completions de OpenAI, así que se reutiliza
`OpenAIClient` (y el SDK `openai`) cambiando solo la URL base y la API key.
"""

from typing import Any

from .openai_client import OpenAIClient
from .schemas import ChatMessage, ErrorType, LLMError


class GeminiClient(OpenAIClient):
    provider = "gemini"
    api_key_env = "GEMINI_API_KEY"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def _request_kwargs(self, messages: list[ChatMessage]) -> dict[str, Any]:
        kwargs = super()._request_kwargs(messages)
        # El endpoint de compatibilidad usa el nombre clásico `max_tokens`.
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        return kwargs

    def _classify_error(self, exc: Exception) -> LLMError:
        error = super()._classify_error(exc)
        # Gemini manda el error como lista (`[{"error": {...}}]`), formato que el SDK de OpenAI
        # no sabe leer: se extrae el mensaje a mano para que el error sea entendible.
        body = getattr(exc, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], dict):
            detail = body[0].get("error", {}).get("message")
            if detail:
                error = error.model_copy(update={"message": f"{error.message} - {detail}"})
        # Gemini responde una key inválida con 400 en lugar de 401.
        if error.type == ErrorType.BAD_REQUEST and "API key" in error.message:
            error = error.model_copy(update={"type": ErrorType.AUTHENTICATION})
        return error
