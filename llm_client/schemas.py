"""Modelos Pydantic: mensajes de entrada, configuración del modelo y respuestas.

Definir estas estructuras primero evita pasar diccionarios anidados sin validar
entre capas (el clásico `msg["choices"][0]["message"]["content"]`).
"""

from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)

Provider = Literal["openai", "anthropic", "gemini"]

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-3.6-flash",
}


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación (role + content)."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("el contenido del mensaje no puede estar vacío")
        return value


class ModelConfig(BaseModel):
    """Parámetros de configuración del proveedor y del modelo."""

    provider: Provider
    model: str | None = Field(default=None, description="Si es None se usa DEFAULT_MODELS[provider]")
    api_key: SecretStr | None = Field(default=None, repr=False)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0, le=32_000)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    timeout: float = Field(default=30.0, gt=0, description="Timeout por request, en segundos")
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, gt=0, description="Base del backoff exponencial, en segundos")

    @model_validator(mode="after")
    def set_default_model(self) -> "ModelConfig":
        if not self.model:
            self.model = DEFAULT_MODELS[self.provider]
        return self


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ErrorType(StrEnum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    SERVER = "server"
    UNKNOWN = "unknown"


class LLMError(BaseModel):
    """Error estructurado que se devuelve en lugar de dejar escapar la excepción del SDK."""

    type: ErrorType
    message: str
    retryable: bool = False
    status_code: int | None = None
    attempts: int = 1


class ModelResponse(BaseModel):
    """Respuesta unificada, independiente del proveedor."""

    provider: Provider
    model: str
    content: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    error: LLMError | None = None
    fallback_from: list[str] = Field(
        default_factory=list, description="Proveedores que fallaron antes de que respondiera este"
    )

    @computed_field
    @property
    def ok(self) -> bool:
        return self.error is None
