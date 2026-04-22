from __future__ import annotations

import json
from typing import Any
from urllib import error, request


class AIClient:
    """Generic chat-completions JSON client (Qwen/OpenAI compatible endpoints)."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete_json(
        self,
        prompt_payload: dict[str, Any],
        *,
        system_prompt: str = "Return strict JSON with concise explanations. Do not include chain-of-thought.",
    ) -> tuple[str, dict[str, Any]]:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"AI request failed: {exc}") from exc

        content = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            raise RuntimeError("AI response did not contain message content.")
        return content, json.loads(content)

    def review_mapping(self, prompt_payload: dict[str, Any]) -> dict[str, Any]:
        _, parsed = self.complete_json(prompt_payload)
        return parsed
