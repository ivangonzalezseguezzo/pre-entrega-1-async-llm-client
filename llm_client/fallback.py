"""`FallbackClient`: si un proveedor falla (aun después de sus reintentos), pasa al siguiente."""

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence

from .base import BaseLLMClient, LLMClientError, MessagesInput, normalize_messages
from .schemas import ModelResponse

logger = logging.getLogger(__name__)


class FallbackClient:
    """Expone la misma interfaz que BaseLLMClient (`generate`, `stream`, `close`) sobre una
    lista ordenada de clientes: el primero es el principal y el resto son los de respaldo.

    Cada cliente ya hace sus propios reintentos con backoff; el fallback entra en juego
    cuando un proveedor se da por vencido (rate limit persistente, caída, key inválida...).
    """

    def __init__(self, clients: Sequence[BaseLLMClient]) -> None:
        if not clients:
            raise ValueError("FallbackClient necesita al menos un cliente")
        self.clients = list(clients)
        self.last_provider: str | None = None

    @property
    def provider(self) -> str:
        return self.clients[0].provider

    @property
    def model(self) -> str:
        return self.clients[0].model

    async def generate(self, messages: MessagesInput) -> ModelResponse:
        chat = normalize_messages(messages)
        failed: list[str] = []
        response: ModelResponse | None = None
        for client in self.clients:
            response = await client.generate(chat)
            if response.ok:
                self.last_provider = client.provider
                if failed:
                    logger.info("Fallback: respondió %s tras fallar %s", client.provider, ", ".join(failed))
                return response.model_copy(update={"fallback_from": failed})
            assert response.error is not None
            logger.warning("Fallback: %s falló (%s), probando el siguiente proveedor", client.provider, response.error.type)
            failed.append(client.provider)
        # Fallaron todos: se devuelve el último error, sin lanzar excepción.
        assert response is not None
        return response.model_copy(update={"fallback_from": failed[:-1]})

    async def stream(self, messages: MessagesInput) -> AsyncIterator[str]:
        """Si un proveedor falla antes de emitir texto, se pasa al siguiente.

        Si falla a mitad de la respuesta no se cambia de proveedor (el usuario ya vio parte del
        texto y otro modelo respondería algo distinto): se propaga el `LLMClientError`.
        """
        chat = normalize_messages(messages)
        last_error: LLMClientError | None = None
        for client in self.clients:
            emitted = False
            try:
                async for token in client.stream(chat):
                    if not emitted:
                        emitted = True
                        self.last_provider = client.provider
                    yield token
                return
            except LLMClientError as exc:
                if emitted:
                    raise
                logger.warning("Fallback: %s falló (%s), probando el siguiente proveedor", client.provider, exc.error.type)
                last_error = exc
        assert last_error is not None
        raise last_error

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients))
