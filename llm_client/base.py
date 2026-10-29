"""Clase base abstracta con la lógica común: reintentos, backoff y manejo de errores."""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from types import ModuleType
from typing import Any, ClassVar

from .schemas import ChatMessage, ErrorType, LLMError, ModelConfig, ModelResponse, Role

logger = logging.getLogger(__name__)

MessagesInput = str | Sequence[ChatMessage | dict[str, Any]]

MAX_RETRY_AFTER_SECONDS = 30.0


class LLMClientError(Exception):
    """Excepción controlada del cliente: siempre trae un `LLMError` estructurado."""

    def __init__(self, error: LLMError) -> None:
        super().__init__(f"[{error.type}] {error.message}")
        self.error = error


def normalize_messages(messages: MessagesInput) -> list[ChatMessage]:
    """Acepta un string, dicts o ChatMessage y devuelve una lista validada de ChatMessage."""
    if isinstance(messages, str):
        return [ChatMessage(role=Role.USER, content=messages)]
    return [m if isinstance(m, ChatMessage) else ChatMessage.model_validate(m) for m in messages]


def classify_sdk_exception(exc: Exception, sdk: ModuleType) -> LLMError:
    """Traduce una excepción de los SDKs `openai` / `anthropic` a un LLMError.

    Ambos SDKs exponen la misma jerarquía de excepciones (generados con Stainless),
    así que la misma tabla sirve para los dos pasando el módulo correspondiente.
    """
    status = getattr(exc, "status_code", None)
    table: list[tuple[type[Exception], ErrorType, bool, str]] = [
        (sdk.AuthenticationError, ErrorType.AUTHENTICATION, False, "API key inválida o ausente"),
        (sdk.PermissionDeniedError, ErrorType.PERMISSION, False, "La API key no tiene permisos para este recurso"),
        (sdk.RateLimitError, ErrorType.RATE_LIMIT, True, "Límite de tasa o cuota excedido"),
        # APITimeoutError hereda de APIConnectionError: tiene que ir antes.
        (sdk.APITimeoutError, ErrorType.TIMEOUT, True, "La request superó el timeout"),
        (sdk.APIConnectionError, ErrorType.CONNECTION, True, "Error de red al conectar con la API"),
        (sdk.NotFoundError, ErrorType.NOT_FOUND, False, "Modelo o recurso inexistente"),
        (sdk.BadRequestError, ErrorType.BAD_REQUEST, False, "Request inválida"),
        (sdk.UnprocessableEntityError, ErrorType.BAD_REQUEST, False, "Request inválida"),
        (sdk.InternalServerError, ErrorType.SERVER, True, "Error interno del proveedor"),
    ]
    for exc_cls, kind, retryable, summary in table:
        if isinstance(exc, exc_cls):
            return LLMError(type=kind, message=f"{summary}: {exc}", retryable=retryable, status_code=status)

    if isinstance(exc, sdk.APIStatusError):
        # p. ej. 529 "overloaded" de Anthropic
        retryable = status is not None and status >= 500
        kind = ErrorType.SERVER if retryable else ErrorType.UNKNOWN
        return LLMError(type=kind, message=str(exc), retryable=retryable, status_code=status)

    return LLMError(type=ErrorType.UNKNOWN, message=f"{type(exc).__name__}: {exc}", retryable=False)


class BaseLLMClient(ABC):
    """Interfaz común para todos los proveedores.

    Las subclases solo implementan una llamada "cruda" (`_generate_once`, `_stream_once`)
    y la traducción de errores; los reintentos y el manejo de errores viven acá.
    """

    provider: ClassVar[str]
    api_key_env: ClassVar[str]

    def __init__(self, config: ModelConfig) -> None:
        if config.provider != self.provider:
            raise ValueError(f"{type(self).__name__} requiere provider='{self.provider}', no '{config.provider}'")
        if config.api_key is None or not config.api_key.get_secret_value().strip():
            raise LLMClientError(
                LLMError(type=ErrorType.CONFIGURATION, message=f"Falta la variable de entorno {self.api_key_env}")
            )
        self.config = config

    @property
    def model(self) -> str:
        assert self.config.model is not None  # lo completa el validador de ModelConfig
        return self.config.model

    # --- Métodos que implementa cada proveedor -------------------------------

    @abstractmethod
    async def _generate_once(self, messages: list[ChatMessage]) -> ModelResponse:
        """Una sola llamada no-streaming al SDK."""

    @abstractmethod
    def _stream_once(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Una sola llamada streaming al SDK; debe ser un generador asíncrono de texto."""

    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError:
        """Traduce una excepción del SDK a un LLMError."""

    @abstractmethod
    async def close(self) -> None:
        """Libera las conexiones HTTP del SDK."""

    # --- API pública ---------------------------------------------------------

    async def generate(self, messages: MessagesInput) -> ModelResponse:
        """Genera una respuesta completa. Nunca lanza errores de la API: los devuelve en `response.error`."""
        chat = normalize_messages(messages)
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._generate_once(chat)
                return response.model_copy(update={"latency_ms": _elapsed_ms(start), "attempts": attempt})
            except Exception as exc:
                error = self._classify_error(exc).model_copy(update={"attempts": attempt})
                if not self._should_retry(error, attempt):
                    logger.warning("%s: fallo definitivo tras %d intento(s): %s", self.provider, attempt, error.message)
                    return ModelResponse(
                        provider=self.config.provider,
                        model=self.model,
                        latency_ms=_elapsed_ms(start),
                        attempts=attempt,
                        error=error,
                    )
                await self._backoff(attempt, exc, error)

    async def stream(self, messages: MessagesInput) -> AsyncIterator[str]:
        """Generador asíncrono que va devolviendo fragmentos de texto a medida que llegan.

        Reintenta solo si el error ocurre antes del primer token (reintentar a mitad de
        respuesta duplicaría texto). Si falla, lanza `LLMClientError` con el error estructurado.
        """
        chat = normalize_messages(messages)
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            try:
                async for token in self._stream_once(chat):
                    emitted = True
                    yield token
                return
            except Exception as exc:
                error = self._classify_error(exc).model_copy(update={"attempts": attempt})
                if emitted or not self._should_retry(error, attempt):
                    raise LLMClientError(error) from exc
                await self._backoff(attempt, exc, error)

    # --- Helpers -------------------------------------------------------------

    def _should_retry(self, error: LLMError, attempt: int) -> bool:
        return error.retryable and attempt <= self.config.max_retries

    async def _backoff(self, attempt: int, exc: Exception, error: LLMError) -> None:
        """Backoff exponencial con jitter; respeta el header `retry-after` si viene."""
        delay = self.config.retry_base_delay * 2 ** (attempt - 1) * random.uniform(1.0, 1.5)
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay = min(retry_after, MAX_RETRY_AFTER_SECONDS)
        logger.info(
            "%s: %s (intento %d/%d), reintentando en %.1fs",
            self.provider, error.type, attempt, self.config.max_retries + 1, delay,
        )
        await asyncio.sleep(delay)

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
