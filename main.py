"""Script de validación: hace la misma pregunta a cada proveedor en modo normal y en streaming.

Uso:
    python main.py                       # todos los proveedores con API key configurada
    python main.py --provider gemini     # solo uno
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import AsyncLLMManager, LLMClientError, ModelResponse

QUESTION = "¿Qué es la entropía?"
SYSTEM_PROMPT = "Respondé en español, de forma clara y en no más de 4 oraciones."
PROVIDERS = ("openai", "anthropic", "gemini")
INVALID_KEY = "clave-invalida-a-proposito"


def title(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def print_response(response: ModelResponse) -> None:
    if response.ok:
        print(response.content)
        usage = f"{response.usage.total_tokens} tokens" if response.usage else "uso desconocido"
        print(f"\n[{response.provider} | {response.model} | {usage} | {response.latency_ms:.0f} ms | intentos: {response.attempts}]")
    else:
        assert response.error is not None
        print(f"ERROR CONTROLADO -> {response.error.model_dump_json(indent=2)}")
    if response.fallback_from:
        print(f"[fallback: antes fallaron {', '.join(response.fallback_from)}]")
    if response.finish_reason in ("length", "max_tokens"):
        print("[AVISO: la respuesta se cortó por el límite de tokens; subí LLM_MAX_TOKENS en el .env]")


async def demo_generate(manager: AsyncLLMManager) -> None:
    title(f"[{manager.provider}] Modo normal (generate)")
    response = await manager.generate(QUESTION, system=SYSTEM_PROMPT)
    print_response(response)


async def demo_stream(manager: AsyncLLMManager) -> None:
    title(f"[{manager.provider}] Modo streaming (stream)")
    try:
        async for token in manager.stream(QUESTION, system=SYSTEM_PROMPT):
            print(token, end="", flush=True)
        print()
    except LLMClientError as exc:
        print(f"\nERROR CONTROLADO -> {exc.error.model_dump_json(indent=2)}")


async def demo_concurrency(managers: list[AsyncLLMManager]) -> None:
    """Como las llamadas no bloquean el event loop, varias pueden correr en paralelo."""
    if len(managers) < 2:
        return
    title("Concurrencia: todos los proveedores a la vez con asyncio.gather")
    responses = await asyncio.gather(*(m.generate("Definí 'entropía' en una sola frase.") for m in managers))
    for response in responses:
        status = response.content if response.ok else f"ERROR: {response.error and response.error.type}"
        print(f"- {response.provider} ({response.latency_ms:.0f} ms): {status}")


async def demo_fallback(configured: list[str]) -> None:
    """El proveedor principal falla (API key inválida a propósito) y responde el de respaldo.

    Si hay un solo proveedor configurado, el respaldo es ese mismo proveedor con la key correcta.
    """
    primary = configured[0]
    backups = configured[1:] or [primary]
    title(f"Fallback: {primary} (key inválida) -> {', '.join(backups)}")
    async with AsyncLLMManager.from_env(primary, fallbacks=backups, api_key=INVALID_KEY, max_retries=0) as manager:
        print_response(await manager.generate(QUESTION, system=SYSTEM_PROMPT))


async def demo_error_handling(provider: str) -> None:
    """Una API key inválida sin respaldo no rompe el programa: vuelve como error estructurado."""
    title(f"[{provider}] Manejo de errores: API key inválida sin fallback")
    async with AsyncLLMManager.from_env(provider, fallbacks=[], api_key=INVALID_KEY, max_retries=1) as manager:
        print_response(await manager.generate(QUESTION))


def build_managers(providers: list[str]) -> list[AsyncLLMManager]:
    managers = []
    for provider in providers:
        try:
            # fallbacks=[]: acá queremos probar cada proveedor por separado.
            managers.append(AsyncLLMManager.from_env(provider, fallbacks=[]))
        except LLMClientError as exc:
            print(f"Se omite '{provider}': {exc.error.message}")
        except ValidationError as exc:
            print(f"Configuración inválida para '{provider}':\n{exc}")
    return managers


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=PROVIDERS, help="probar un solo proveedor")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx loguea cada request HTTP; lo silenciamos para que la salida se lea mejor.
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)

    providers = [args.provider] if args.provider else list(PROVIDERS)
    managers = build_managers(providers)
    if not managers:
        print("\nNo hay proveedores configurados. Copiá .env.example a .env y completá al menos una API key.")
        return
    configured = [m.provider for m in managers]

    try:
        for manager in managers:
            await demo_generate(manager)
            await demo_stream(manager)
        await demo_concurrency(managers)
    finally:
        await asyncio.gather(*(m.close() for m in managers))

    await demo_fallback(configured)
    await demo_error_handling(configured[0])


if __name__ == "__main__":
    # Evita problemas con tildes / "¿" en la consola de Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
