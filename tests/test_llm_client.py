"""Tests sin red: simulan errores de rate limit y verifican reintentos, streaming y fallback.

Correr con:  python -m unittest -v
"""

import unittest
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace

import anthropic
import openai
from pydantic import ValidationError

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    ChatMessage,
    ErrorType,
    FallbackClient,
    GeminiClient,
    LLMClientError,
    ModelConfig,
    ModelResponse,
    OpenAIClient,
)


def rate_limit_error(sdk: ModuleType) -> Exception:
    """Crea el RateLimitError (HTTP 429) real del SDK, sin hacer ninguna request."""
    response = SimpleNamespace(request=None, status_code=429, headers={"retry-after": "0"})
    return sdk.RateLimitError("429 Too Many Requests (simulado)", response=response, body=None)


def auth_error(sdk: ModuleType) -> Exception:
    response = SimpleNamespace(request=None, status_code=401, headers={})
    return sdk.AuthenticationError("401 Invalid API key (simulado)", response=response, body=None)


class ScriptedMixin:
    """Reemplaza la llamada a la API por un guion: cada paso es una excepción a lanzar o un texto a responder.

    Todo lo demás (reintentos, backoff, clasificación de errores) es el código real del cliente.
    """

    def __init__(self, config: ModelConfig, script: list[Exception | str]) -> None:
        super().__init__(config)  # type: ignore[call-arg]
        self.script = list(script)
        self.calls = 0

    def _next_step(self) -> str:
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def _generate_once(self, messages: list[ChatMessage]) -> ModelResponse:
        text = self._next_step()
        return ModelResponse(provider=self.config.provider, model=self.model, content=text)  # type: ignore[attr-defined]

    async def _stream_once(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        text = self._next_step()
        for word in text.split():
            yield word + " "


class FakeOpenAI(ScriptedMixin, OpenAIClient):
    pass


class FakeAnthropic(ScriptedMixin, AnthropicClient):
    pass


def make_config(provider: str, max_retries: int = 2) -> ModelConfig:
    # retry_base_delay mínimo para que los tests no esperen de verdad.
    return ModelConfig(provider=provider, api_key="test-key", max_retries=max_retries, retry_base_delay=0.001)


async def collect(stream: AsyncIterator[str]) -> str:
    return "".join([token async for token in stream])


class TestSchemas(unittest.TestCase):
    def test_temperature_fuera_de_rango(self) -> None:
        with self.assertRaises(ValidationError):
            ModelConfig(provider="openai", temperature=2.5)

    def test_max_tokens_debe_ser_positivo(self) -> None:
        with self.assertRaises(ValidationError):
            ModelConfig(provider="openai", max_tokens=0)

    def test_proveedor_inexistente(self) -> None:
        with self.assertRaises(ValidationError):
            ModelConfig(provider="proveedor-inventado")

    def test_mensaje_vacio(self) -> None:
        with self.assertRaises(ValidationError):
            ChatMessage(role="user", content="   ")

    def test_modelo_por_defecto(self) -> None:
        self.assertTrue(ModelConfig(provider="gemini").model)

    def test_api_key_oculta_al_imprimir(self) -> None:
        self.assertNotIn("test-key", repr(make_config("openai")))


class TestRetries(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limit_se_reintenta_y_luego_responde(self) -> None:
        client = FakeOpenAI(make_config("openai"), [rate_limit_error(openai), rate_limit_error(openai), "ok"])
        response = await client.generate("¿Qué es la entropía?")
        self.assertTrue(response.ok)
        self.assertEqual(response.content, "ok")
        self.assertEqual(response.attempts, 3)

    async def test_rate_limit_persistente_devuelve_error_controlado(self) -> None:
        client = FakeOpenAI(make_config("openai", max_retries=2), [rate_limit_error(openai)] * 3)
        response = await client.generate("hola")  # no debe lanzar excepción
        self.assertFalse(response.ok)
        assert response.error is not None
        self.assertEqual(response.error.type, ErrorType.RATE_LIMIT)
        self.assertEqual(response.error.status_code, 429)
        self.assertEqual(client.calls, 3)

    async def test_api_key_invalida_no_se_reintenta(self) -> None:
        client = FakeAnthropic(make_config("anthropic"), [auth_error(anthropic)])
        response = await client.generate("hola")
        assert response.error is not None
        self.assertEqual(response.error.type, ErrorType.AUTHENTICATION)
        self.assertEqual(client.calls, 1)

    async def test_falta_api_key_falla_claro(self) -> None:
        with self.assertRaises(LLMClientError) as ctx:
            OpenAIClient(ModelConfig(provider="openai"))
        self.assertEqual(ctx.exception.error.type, ErrorType.CONFIGURATION)


class TestStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_stream_devuelve_tokens_de_a_uno(self) -> None:
        client = FakeOpenAI(make_config("openai"), ["La entropía mide el desorden"])
        tokens = [token async for token in client.stream("¿Qué es la entropía?")]
        self.assertEqual(len(tokens), 5)
        self.assertEqual("".join(tokens).strip(), "La entropía mide el desorden")

    async def test_stream_reintenta_rate_limit_antes_del_primer_token(self) -> None:
        client = FakeOpenAI(make_config("openai"), [rate_limit_error(openai), "hola mundo"])
        self.assertEqual(await collect(client.stream("hola")), "hola mundo ")
        self.assertEqual(client.calls, 2)

    async def test_stream_con_rate_limit_persistente_lanza_error_controlado(self) -> None:
        client = FakeOpenAI(make_config("openai", max_retries=1), [rate_limit_error(openai)] * 2)
        with self.assertRaises(LLMClientError) as ctx:
            await collect(client.stream("hola"))
        self.assertEqual(ctx.exception.error.type, ErrorType.RATE_LIMIT)


class TestFallback(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limit_en_openai_responde_anthropic(self) -> None:
        primary = FakeOpenAI(make_config("openai", max_retries=1), [rate_limit_error(openai)] * 2)
        backup = FakeAnthropic(make_config("anthropic"), ["respuesta de respaldo"])
        client = FallbackClient([primary, backup])

        response = await client.generate("¿Qué es la entropía?")

        self.assertTrue(response.ok)
        self.assertEqual(response.provider, "anthropic")
        self.assertEqual(response.content, "respuesta de respaldo")
        self.assertEqual(response.fallback_from, ["openai"])
        self.assertEqual(primary.calls, 2)  # primero agotó sus reintentos

    async def test_fallback_en_streaming(self) -> None:
        primary = FakeOpenAI(make_config("openai", max_retries=0), [rate_limit_error(openai)])
        backup = FakeAnthropic(make_config("anthropic"), ["hola desde anthropic"])
        client = FallbackClient([primary, backup])

        self.assertEqual(await collect(client.stream("hola")), "hola desde anthropic ")
        self.assertEqual(client.last_provider, "anthropic")

    async def test_si_fallan_todos_devuelve_error_sin_crash(self) -> None:
        primary = FakeOpenAI(make_config("openai", max_retries=0), [rate_limit_error(openai)])
        backup = FakeAnthropic(make_config("anthropic", max_retries=0), [rate_limit_error(anthropic)])
        response = await FallbackClient([primary, backup]).generate("hola")

        self.assertFalse(response.ok)
        assert response.error is not None
        self.assertEqual(response.error.type, ErrorType.RATE_LIMIT)
        self.assertEqual(response.fallback_from, ["openai"])

    async def test_sin_errores_no_usa_el_respaldo(self) -> None:
        primary = FakeOpenAI(make_config("openai"), ["respuesta principal"])
        backup = FakeAnthropic(make_config("anthropic"), [])
        response = await FallbackClient([primary, backup]).generate("hola")
        self.assertEqual(response.provider, "openai")
        self.assertEqual(backup.calls, 0)


class TestManager(unittest.IsolatedAsyncioTestCase):
    async def test_factory_crea_el_cliente_segun_el_proveedor(self) -> None:
        for provider, expected in [("openai", OpenAIClient), ("anthropic", AnthropicClient), ("gemini", GeminiClient)]:
            manager = AsyncLLMManager(make_config(provider))
            self.assertIs(type(manager.client), expected)
            await manager.close()

    async def test_gemini_usa_el_endpoint_compatible_con_openai(self) -> None:
        client = GeminiClient(make_config("gemini"))
        self.assertIn("generativelanguage.googleapis.com", str(client._client.base_url))
        await client.close()

    async def test_gemini_key_invalida_se_clasifica_como_autenticacion(self) -> None:
        # Así responde Gemini de verdad a una key inválida: 400 con el error dentro de una lista.
        response = SimpleNamespace(request=None, status_code=400, headers={})
        body = [{"error": {"code": 400, "message": "Please pass a valid API key", "status": "INVALID_ARGUMENT"}}]
        exc = openai.BadRequestError("Error code: 400", response=response, body=body)
        client = GeminiClient(make_config("gemini"))
        error = client._classify_error(exc)
        self.assertEqual(error.type, ErrorType.AUTHENTICATION)
        self.assertIn("valid API key", error.message)
        await client.close()

    async def test_manager_con_fallbacks_usa_fallback_client(self) -> None:
        manager = AsyncLLMManager(make_config("openai"), fallbacks=[make_config("gemini")])
        self.assertIsInstance(manager.client, FallbackClient)
        await manager.close()


if __name__ == "__main__":
    unittest.main()
