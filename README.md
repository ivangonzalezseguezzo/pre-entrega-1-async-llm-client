# Unified Async LLM Client

Pre-entrega 1 del curso de AI Engineering: un cliente **asíncrono** que expone una interfaz común
para **OpenAI**, **Anthropic** y **Gemini**. Incluye **streaming**, validación con **Pydantic**,
**reintentos con backoff** y **fallback automático** entre proveedores, así que un error (API key
inválida, rate limit, red) no hace que el programa se caiga.

## Qué incluye

| Requisito | Dónde está |
|---|---|
| Esquemas Pydantic (`ChatMessage`, `ModelConfig`, `ModelResponse`, `LLMError`) | [`llm_client/schemas.py`](llm_client/schemas.py) |
| Clase base abstracta `BaseLLMClient` con `async def generate()` y `stream()` | [`llm_client/base.py`](llm_client/base.py) |
| `OpenAIClient` sobre `AsyncOpenAI` | [`llm_client/openai_client.py`](llm_client/openai_client.py) |
| `AnthropicClient` sobre `AsyncAnthropic` | [`llm_client/anthropic_client.py`](llm_client/anthropic_client.py) |
| `GeminiClient` (endpoint de Gemini compatible con OpenAI) | [`llm_client/gemini_client.py`](llm_client/gemini_client.py) |
| `FallbackClient`: si un proveedor falla, responde el siguiente | [`llm_client/fallback.py`](llm_client/fallback.py) |
| `AsyncLLMManager` + factory `create_client()`: elige el proveedor según la configuración | [`llm_client/manager.py`](llm_client/manager.py) |
| Script de prueba (modo normal, streaming, fallback y errores) | [`main.py`](main.py) |
| Tests que simulan errores de rate limit | [`tests/test_llm_client.py`](tests/test_llm_client.py) |
| Variables de entorno de ejemplo | [`.env.example`](.env.example) |

```
.
├── llm_client/
│   ├── __init__.py
│   ├── schemas.py           # Modelos Pydantic
│   ├── base.py              # BaseLLMClient: reintentos, backoff, errores
│   ├── openai_client.py     # OpenAIClient (AsyncOpenAI)
│   ├── anthropic_client.py  # AnthropicClient (AsyncAnthropic)
│   ├── gemini_client.py     # GeminiClient (reutiliza OpenAIClient)
│   ├── fallback.py          # FallbackClient
│   └── manager.py           # AsyncLLMManager, create_client(), config_from_env()
├── tests/
│   └── test_llm_client.py   # Tests sin red (rate limit simulado, fallback, streaming)
├── main.py                  # Script de validación
├── requirements.txt
├── .env.example
└── .python-version          # 3.12
```

## Cómo ejecutarlo

### 1. Crear el entorno virtual con Python 3.12

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencias: `openai`, `anthropic`, `pydantic` y `python-dotenv`. Gemini no necesita una
librería aparte: se usa con el SDK `openai` apuntando a su endpoint compatible.

### 2. Configurar las variables de entorno

```bash
cp .env.example .env      # en Windows: copy .env.example .env
```

Completá en `.env` **al menos una** API key; los proveedores sin key se omiten con un aviso.
El archivo `.env` está en `.gitignore` y **no se sube al repo**.

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `OPENAI_API_KEY` | Para usar OpenAI | — | API key de OpenAI |
| `ANTHROPIC_API_KEY` | Para usar Anthropic | — | API key de Anthropic |
| `GEMINI_API_KEY` | Para usar Gemini | — | API key de Google AI Studio |
| `LLM_PROVIDER` | No | `openai` | Proveedor principal de `AsyncLLMManager.from_env()` (`openai`, `anthropic` o `gemini`) |
| `LLM_FALLBACKS` | No | vacío | Proveedores de respaldo, en orden, separados por coma (ej. `openai,anthropic`) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-5-5` | Modelo de Anthropic |
| `GEMINI_MODEL` | No | `gemini-3.6-flash` | Modelo de Gemini |
| `LLM_TEMPERATURE` | No | `0.7` | Entre 0 y 2. Se aplica en OpenAI y Gemini (ver nota abajo) |
| `LLM_MAX_TOKENS` | No | `2048` | Máximo de tokens de salida (1 a 32000). Los modelos que razonan antes de responder (Gemini 3) gastan parte de este límite, y si es muy bajo la respuesta sale cortada |
| `LLM_TIMEOUT` | No | `30` | Timeout por request, en segundos |
| `LLM_MAX_RETRIES` | No | `3` | Reintentos ante rate limit, timeouts, errores de red o 5xx |

Todos estos valores pasan por `ModelConfig` (Pydantic): si ponés `LLM_TEMPERATURE=3` o un proveedor
inexistente, falla la validación con un mensaje claro antes de llamar a la API.

### 3. Correr el script de prueba

```bash
python main.py                       # prueba todos los proveedores que tengan API key
python main.py --provider gemini     # solo uno (openai | anthropic | gemini)
```

El script pregunta **"¿Qué es la entropía?"** y muestra:

1. **Modo normal** (`generate`): la respuesta completa, el proveedor, el modelo, los tokens usados, la latencia y la cantidad de intentos.
2. **Modo streaming** (`stream`): los fragmentos de texto a medida que llegan.
3. **Concurrencia**: si hay más de un proveedor, les pregunta a todos a la vez con `asyncio.gather`.
4. **Fallback**: el proveedor principal usa una API key inválida a propósito y responde el de respaldo.
   Si hay un solo proveedor configurado, el respaldo es ese mismo proveedor con la key correcta.
5. **Manejo de errores**: una API key inválida sin respaldo devuelve un error estructurado y el programa sigue.

Ejemplo del error controlado del paso 5:

```json
{
  "type": "authentication",
  "message": "API key inválida o ausente: Error code: 401 - ...",
  "retryable": false,
  "status_code": 401,
  "attempts": 1
}
```

### 4. Correr los tests

Los tests no usan la red ni gastan créditos: reemplazan la llamada a la API por errores
`RateLimitError` (HTTP 429) reales del SDK y verifican que el cliente reintente, haga fallback y
nunca se caiga. Usan `unittest` (viene con Python, no hay que instalar nada):

```bash
python -m unittest -v
```

Qué se prueba:
- Validaciones de Pydantic (temperature fuera de rango, `max_tokens` inválido, mensaje vacío, API key oculta al imprimir).
- Un rate limit se reintenta con backoff y luego responde; si persiste, devuelve un error controlado.
- Una API key inválida **no** se reintenta (no tiene sentido) y falla de forma clara.
- Streaming token por token, con reintento si el error ocurre antes del primer token.
- **Fallback**: rate limit persistente en OpenAI → responde Anthropic, tanto en `generate` como en `stream`;
  si fallan todos, se devuelve el último error sin excepción.
- La factory crea la clase correcta para cada proveedor.

## Uso desde código

```python
import asyncio
from dotenv import load_dotenv
from llm_client import AsyncLLMManager, LLMClientError

async def main():
    load_dotenv()
    # Gemini como principal; si falla, OpenAI y después Anthropic.
    async with AsyncLLMManager.from_env("gemini", fallbacks=["openai", "anthropic"]) as llm:
        # Modo normal: nunca lanza errores de la API, los devuelve en response.error
        response = await llm.generate("¿Qué es la entropía?", system="Respondé en español.")
        print(response.content if response.ok else response.error)
        print("Respondió:", response.provider, "| fallaron antes:", response.fallback_from)

        # Streaming: generador asíncrono de fragmentos de texto
        try:
            async for token in llm.stream("¿Qué es la entropía?"):
                print(token, end="", flush=True)
        except LLMClientError as exc:
            print(exc.error)

asyncio.run(main())
```

Para cambiar de proveedor alcanza con cambiar `"gemini"` por `"openai"` o `"anthropic"` (o
`LLM_PROVIDER` en el `.env`): el resto del código no cambia.

## Decisiones de diseño

- **Todo es no bloqueante.** Se usan `AsyncOpenAI` y `AsyncAnthropic` con `await`; nunca el cliente
  síncrono dentro de una función `async`, así el event loop sigue libre mientras el modelo responde
  (por eso funciona la demo con `asyncio.gather`).
- **ABC + factory.** `BaseLLMClient` es el contrato; cada proveedor solo implementa la llamada "cruda"
  al SDK (`_generate_once` / `_stream_once`) y la traducción de sus excepciones. Los reintentos, el
  backoff y la respuesta de error están una sola vez en la clase base. `create_client()` es el único
  lugar que decide qué clase concreta instanciar.
- **Gemini reutiliza el cliente de OpenAI.** Gemini ofrece un endpoint compatible con Chat Completions,
  así que `GeminiClient` hereda de `OpenAIClient` y solo cambia la URL base, la API key y la lectura de
  sus errores.
- **Errores controlados.** Las excepciones de los SDKs se traducen a un `LLMError` con un `type`
  (`authentication`, `rate_limit`, `timeout`, `connection`, `server`, ...) y un flag `retryable`.
  - `generate()` nunca lanza errores de la API: devuelve un `ModelResponse` con `ok == False` y el `error` completo.
  - `stream()` lanza una única excepción propia, `LLMClientError`, que trae el mismo `LLMError` adentro.
- **Reintentos con backoff exponencial + jitter** solo para errores transitorios (429, timeouts, red,
  5xx). Si la API manda el header `retry-after`, se respeta. Los reintentos internos de los SDKs se
  desactivan (`max_retries=0`) para que la política sea una sola y visible en los logs.
- **Fallback en dos niveles.** Primero cada proveedor reintenta con backoff; si se da por vencido,
  `FallbackClient` pasa al siguiente de la lista. `response.fallback_from` y los logs dicen qué
  proveedores fallaron y cuál respondió.
- **En streaming no se cambia de proveedor a mitad de respuesta**: si ya se mostraron tokens,
  reintentar o pasar a otro modelo duplicaría o mezclaría texto. Solo se reintenta o hace fallback
  antes del primer token.
- **System prompt.** OpenAI y Gemini lo reciben como un mensaje con `role="system"`; Anthropic como
  parámetro `system` aparte. El cliente de Anthropic hace esa conversión.
- **Nota sobre `temperature` en Anthropic.** La versión actual de la Messages API de Anthropic (SDK
  `anthropic` 2.x) ya no acepta `temperature` ni `top_p`. Esos campos siguen validados en `ModelConfig`
  y se envían a OpenAI y Gemini, pero el cliente de Anthropic no los manda.
