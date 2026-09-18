"""Optional VLM provider adapters."""

from .openai_compatible import (
    CURRENT_MAX_TOKENS,
    CURRENT_MEDIA_TYPE,
    CURRENT_SEED,
    OPENAI_COMPATIBLE_INFERENCE_IDENTITY_SCHEMA,
    TOKEN_LIMIT_PARAMETERS,
    OpenAICompatibleVLMError,
    OpenAICompatibleVLMInference,
)

__all__ = [name for name in globals() if not name.startswith("_")]
