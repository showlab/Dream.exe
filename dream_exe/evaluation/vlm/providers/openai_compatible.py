"""Concrete OpenAI-compatible inference for saved VLM image artifacts.

The adapter is directly callable with the ``(prompt, media_path, options)``
contract consumed by :mod:`dream_exe.evaluation.vlm`.  Authentication, model,
and endpoint configuration are explicit constructor inputs.  The optional
OpenAI SDK is imported only when an injected client was not supplied.

The paper evaluators all use the same request shape: one user message
containing a text part followed by a JPEG data-URL image part, a completion
token limit of 8000, seed 2026, and the stripped first response text.  That is
the default here.  ``token_limit_parameter`` makes the common
``max_tokens``-compatible provider variant explicit without duplicating the
request implementation.
"""

from __future__ import annotations

import base64
import posixpath
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..base import BaseVLMBackend, VLM_BACKEND_CONTRACT_VERSION

CURRENT_MAX_TOKENS = 8000
CURRENT_SEED = 2026
CURRENT_MEDIA_TYPE = "image/jpeg"
TOKEN_LIMIT_PARAMETERS = (
    "max_completion_tokens",
    "max_tokens",
)
OPENAI_COMPATIBLE_INFERENCE_IDENTITY_SCHEMA = (
    "dream-exe.openai-compatible-vlm-inference"
)

_REDACTED = "[REDACTED]"
_SENSITIVE_OPTION_NAMES = {
    "api_key",
    "api_token",
    "access_token",
    "auth_token",
    "bearer_token",
    "token",
    "password",
    "secret",
    "client_secret",
    "authorization",
    "credential",
    "credentials",
    "headers",
    "extra_headers",
    "default_headers",
    "client",
    "http_client",
    "base_url",
}
_RESERVED_REQUEST_OPTIONS = {
    "model",
    "messages",
}
_DATA_URL_PATTERN = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[a-zA-Z0-9+/=_-]+")
_NAMED_CREDENTIAL_PATTERN = re.compile(
    (
        r"(?i)\b(api[ _-]?key|access[ _-]?token|"
        r"refresh[ _-]?token|authorization|password|"
        r"client[ _-]?secret|credential)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    )
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")


class OpenAICompatibleVLMError(RuntimeError):
    """A credential- and media-redacted adapter failure."""


def _normalize_option_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value or "").strip().lower(),
    ).strip("_")


def _is_sensitive_option_name(value: Any) -> bool:
    normalized = _normalize_option_name(value)
    collapsed = normalized.replace("_", "")
    return (
        normalized in _SENSITIVE_OPTION_NAMES
        or collapsed
        in {
            "apikey",
            "apitoken",
            "accesstoken",
            "authtoken",
            "bearertoken",
            "clientsecret",
            "authorization",
            "credential",
            "credentials",
            "extraheaders",
            "defaultheaders",
            "httpclient",
            "baseurl",
        }
        or normalized.endswith(
            (
                "_api_key",
                "_access_token",
                "_auth_token",
                "_secret",
                "_password",
                "_authorization",
                "_headers",
            )
        )
    )


def _sensitive_option_paths(
    value: Any,
    *,
    prefix: str = "",
    seen: set[int] | None = None,
) -> list[str]:
    visited = seen if seen is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in visited:
            return []
        visited.add(identity)
        paths = []
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if _is_sensitive_option_name(key):
                paths.append(path)
            paths.extend(
                _sensitive_option_paths(
                    item,
                    prefix=path,
                    seen=visited,
                )
            )
        return paths
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in visited:
            return []
        visited.add(identity)
        paths = []
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            paths.extend(
                _sensitive_option_paths(
                    item,
                    prefix=path,
                    seen=visited,
                )
            )
        return paths
    return []


def _contains_text(
    value: Any,
    text: str,
    *,
    seen: set[int] | None = None,
) -> bool:
    if not text:
        return False
    if isinstance(value, str):
        return text in value
    visited = seen if seen is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in visited:
            return False
        visited.add(identity)
        return any(
            _contains_text(key, text, seen=visited)
            or _contains_text(item, text, seen=visited)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in visited:
            return False
        visited.add(identity)
        return any(_contains_text(item, text, seen=visited) for item in value)
    return False


def _redact_error_text(
    value: Any,
    *,
    secrets: tuple[str, ...],
) -> str:
    try:
        text = str(value or "")
    except Exception:
        text = f"<{type(value).__name__}>"
    for secret in secrets:
        if secret:
            text = text.replace(secret, _REDACTED)
    text = _DATA_URL_PATTERN.sub(
        "data:image/[REDACTED]",
        text,
    )
    text = _NAMED_CREDENTIAL_PATTERN.sub(
        rf"\1\2{_REDACTED}",
        text,
    )
    text = _BEARER_PATTERN.sub(
        f"Bearer {_REDACTED}",
        text,
    )
    clean = text.strip() or "backend request failed"
    return clean[:2048]


def _safe_failure(
    *,
    operation: str,
    error: Exception,
    api_key: str,
) -> OpenAICompatibleVLMError:
    message = _redact_error_text(
        error,
        secrets=(api_key,),
    )
    return OpenAICompatibleVLMError(f"OpenAI-compatible {operation} failed: {message}")


def _validate_base_url(base_url: str) -> str:
    clean = str(base_url or "").strip()
    try:
        parsed = urlsplit(clean)
    except ValueError:
        parsed = None
    if (
        parsed is None
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise ValueError(
            "base_url must be an absolute http(s) URL without "
            "credentials, query, or fragment"
        )
    return clean.rstrip("/")


def _endpoint_identity(base_url: str) -> dict[str, Any]:
    """Return a normalized endpoint identity without credentials or query data."""

    parsed = urlsplit(_validate_base_url(base_url))
    hostname = str(parsed.hostname or "").lower()
    port = parsed.port
    if (parsed.scheme == "https" and port == 443) or (
        parsed.scheme == "http" and port == 80
    ):
        port = None
    raw_path = parsed.path or "/"
    normalized_path = posixpath.normpath(raw_path)
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    return {
        "scheme": parsed.scheme.lower(),
        "host": hostname,
        "port": port,
        "path": normalized_path,
    }


def _validate_positive_token_limit(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("completion token limit must be a positive integer")
    invalid = False
    try:
        limit = int(value)
    except (TypeError, ValueError):
        invalid = True
        limit = 0
    if invalid:
        raise ValueError("completion token limit must be a positive integer")
    if limit < 1:
        raise ValueError("completion token limit must be a positive integer")
    return limit


def _validate_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("seed must be an integer or None")
    invalid = False
    try:
        seed = int(value)
    except (TypeError, ValueError):
        invalid = True
        seed = 0
    if invalid:
        raise ValueError("seed must be an integer or None")
    return seed


class OpenAICompatibleVLMInference(BaseVLMBackend):
    """Callable OpenAI-compatible VLM adapter with explicit credentials."""

    provider_kind = "builtin"
    backend_id = "openai_compatible"
    contract_version = VLM_BACKEND_CONTRACT_VERSION

    __slots__ = (
        "_api_key",
        "_base_url",
        "_client",
        "_client_origin",
        "_max_tokens",
        "_media_type",
        "_model",
        "_seed",
        "_token_limit_parameter",
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        client: Any | None = None,
        max_tokens: int = CURRENT_MAX_TOKENS,
        seed: int | None = CURRENT_SEED,
        media_type: str = CURRENT_MEDIA_TYPE,
        token_limit_parameter: str = "max_completion_tokens",
    ) -> None:
        clean_api_key = str(api_key or "").strip()
        if not clean_api_key:
            raise ValueError("api_key must be non-empty")
        clean_model = str(model or "").strip()
        if not clean_model:
            raise ValueError("model must be non-empty")
        clean_media_type = str(media_type or "").strip().lower()
        if (
            not clean_media_type.startswith("image/")
            or ";" in clean_media_type
            or "," in clean_media_type
        ):
            raise ValueError("media_type must be a plain image/* MIME type")
        clean_token_parameter = str(token_limit_parameter or "").strip()
        if clean_token_parameter not in TOKEN_LIMIT_PARAMETERS:
            raise ValueError(
                "token_limit_parameter must be max_completion_tokens or max_tokens"
            )

        clean_base_url = _validate_base_url(base_url)
        if clean_api_key in clean_model or clean_api_key in clean_base_url:
            raise ValueError("model and base_url must not contain the api credential")

        self._api_key = clean_api_key
        self._model = clean_model
        self._base_url = clean_base_url
        self._client = client
        self._client_origin = "injected" if client is not None else "lazy"
        self._max_tokens = _validate_positive_token_limit(max_tokens)
        self._seed = _validate_seed(seed)
        self._media_type = clean_media_type
        self._token_limit_parameter = clean_token_parameter

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def token_limit_parameter(self) -> str:
        return self._token_limit_parameter

    def inference_identity(self) -> dict[str, Any]:
        """Describe behavior-affecting adapter settings without credentials."""

        return {
            "format": OPENAI_COMPATIBLE_INFERENCE_IDENTITY_SCHEMA,
            "protocol": "openai-compatible-chat-completions",
            "model": self._model,
            "endpoint": _endpoint_identity(self._base_url),
            "request": {
                "media_type": self._media_type,
                "token_limit_parameter": self._token_limit_parameter,
                "max_tokens": self._max_tokens,
                "seed": self._seed,
            },
        }

    def __repr__(self) -> str:
        safe_model = _redact_error_text(
            self._model,
            secrets=(self._api_key,),
        )
        return (
            f"{type(self).__name__}("
            f"model={safe_model!r}, "
            f"token_limit_parameter="
            f"{self._token_limit_parameter!r}, "
            f"client={self._client_origin!r})"
        )

    def _runtime_client(self) -> Any:
        if self._client is not None:
            return self._client

        failure: OpenAICompatibleVLMError | None = None
        client: Any | None = None
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        except ModuleNotFoundError:
            failure = OpenAICompatibleVLMError(
                "OpenAI-compatible client setup failed: "
                "the optional openai SDK is not installed"
            )
        except Exception as error:
            failure = _safe_failure(
                operation="client setup",
                error=error,
                api_key=self._api_key,
            )
        if failure is not None:
            raise failure
        self._client = client
        self._client_origin = "lazy-initialized"
        return client

    def _request_options(
        self,
        generation_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        option_failure: OpenAICompatibleVLMError | None = None
        options: dict[str, Any] = {}
        try:
            options = dict(generation_options or {})
        except Exception as error:
            option_failure = _safe_failure(
                operation="generation options",
                error=error,
                api_key=self._api_key,
            )
        if option_failure is not None:
            raise option_failure
        if _contains_text(options, self._api_key):
            raise OpenAICompatibleVLMError(
                "generation options contain the configured credential"
            )
        sensitive_paths = _sensitive_option_paths(options)
        if sensitive_paths:
            raise OpenAICompatibleVLMError(
                "generation options contain sensitive client "
                "configuration at: " + ", ".join(sensitive_paths)
            )

        reserved = {
            str(key)
            for key in options
            if _normalize_option_name(key) in _RESERVED_REQUEST_OPTIONS
        }
        if reserved:
            raise ValueError(
                "generation options may not override: " + ", ".join(sorted(reserved))
            )
        supplied_token_parameters = [
            name for name in TOKEN_LIMIT_PARAMETERS if name in options
        ]
        if len(supplied_token_parameters) > 1:
            raise ValueError(
                "generation options may contain only one of "
                "max_completion_tokens or max_tokens"
            )
        if supplied_token_parameters:
            token_name = supplied_token_parameters[0]
            options[token_name] = _validate_positive_token_limit(options[token_name])
        else:
            options[self._token_limit_parameter] = self._max_tokens
        if self._seed is not None:
            options.setdefault("seed", self._seed)
        if bool(options.get("stream", False)):
            raise ValueError(
                "streaming responses are incompatible with the "
                "current first-choice text contract"
            )
        return options

    def infer(
        self,
        prompt: str,
        media_path: str | Path,
        generation_options: Mapping[str, Any] | None = None,
    ) -> str:
        """Run one current-compatible multimodal chat completion."""

        path = Path(media_path)
        read_failure: OpenAICompatibleVLMError | None = None
        media_bytes = b""
        try:
            media_bytes = path.read_bytes()
        except Exception as error:
            read_failure = _safe_failure(
                operation="media read",
                error=error,
                api_key=self._api_key,
            )
        if read_failure is not None:
            raise read_failure
        if self._api_key.encode("utf-8") in media_bytes:
            raise OpenAICompatibleVLMError(
                "media contains the configured credential and was discarded"
            )
        prompt_text = str(prompt)
        if self._api_key in prompt_text:
            raise OpenAICompatibleVLMError(
                "prompt contains the configured credential and was discarded"
            )

        encoded = base64.b64encode(media_bytes).decode("utf-8")
        data_url = f"data:{self._media_type};base64,{encoded}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ]
        options = self._request_options(generation_options)
        client = self._runtime_client()

        request_failure: OpenAICompatibleVLMError | None = None
        response_text: str | None = None
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                **options,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise TypeError("first completion choice did not contain text")
            response_text = content.strip()
        except Exception as error:
            request_failure = _safe_failure(
                operation="completion request",
                error=error,
                api_key=self._api_key,
            )
        if request_failure is not None:
            raise request_failure
        assert response_text is not None
        if self._api_key in response_text:
            raise OpenAICompatibleVLMError(
                "backend response contained the configured credential and was discarded"
            )
        return response_text

    def __call__(
        self,
        prompt: str,
        media_path: str | Path,
        generation_options: Mapping[str, Any] | None = None,
    ) -> str:
        return self.infer(
            prompt,
            media_path,
            generation_options,
        )


__all__ = [
    "CURRENT_MAX_TOKENS",
    "CURRENT_MEDIA_TYPE",
    "CURRENT_SEED",
    "OPENAI_COMPATIBLE_INFERENCE_IDENTITY_SCHEMA",
    "OpenAICompatibleVLMError",
    "OpenAICompatibleVLMInference",
    "TOKEN_LIMIT_PARAMETERS",
]
