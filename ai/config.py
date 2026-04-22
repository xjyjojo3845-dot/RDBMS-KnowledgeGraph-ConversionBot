from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AISettings:
    provider: str
    api_key: str
    base_url: str
    model: str


def _load_settings_dict(secrets_path: str | Path = "config/secrets.json") -> dict[str, str]:
    path = Path(secrets_path)
    data: dict[str, str] = {}

    if path.exists():
        try:
            parsed = json.loads(path.read_text())
            if isinstance(parsed, dict):
                data = {str(k): str(v) for k, v in parsed.items() if v is not None}
        except json.JSONDecodeError:
            data = {}
    return data


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def load_ai_settings(secrets_path: str | Path = "config/secrets.json") -> AISettings | None:
    """Load provider credentials from local secrets first, then environment fallback."""
    data = _load_settings_dict(secrets_path)

    provider = (data.get("AI_PROVIDER") or os.getenv("AI_PROVIDER") or "qwen").strip().lower()

    if provider == "qwen":
        api_key = (data.get("QWEN_API_KEY") or os.getenv("QWEN_API_KEY") or "").strip()
        base_url = _normalize_base_url(
            data.get("QWEN_BASE_URL")
            or os.getenv("QWEN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        model = (data.get("QWEN_MODEL") or os.getenv("QWEN_MODEL") or "qwen-plus").strip()
    else:
        api_key = (data.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        base_url = _normalize_base_url(data.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        model = (data.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()

    if not api_key:
        return None

    return AISettings(provider=provider, api_key=api_key, base_url=base_url, model=model)


def semantic_review_enabled(secrets_path: str | Path = "config/secrets.json") -> bool:
    data = _load_settings_dict(secrets_path)
    raw = data.get("SEMANTIC_REVIEW_ENABLED") or os.getenv("SEMANTIC_REVIEW_ENABLED")
    return _as_bool(raw, default=True)


def load_required_qwen_settings(secrets_path: str | Path = "config/secrets.json") -> AISettings:
    settings = load_ai_settings(secrets_path=secrets_path)
    if not settings:
        raise RuntimeError("Qwen AI is required. Configure QWEN_API_KEY in config/secrets.json or the environment.")
    if settings.provider != "qwen":
        raise RuntimeError("This app is configured to require Qwen AI review. Set AI_PROVIDER to `qwen`.")
    return settings
