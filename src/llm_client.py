from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, TypeVar
from urllib import request

from .config import LlmConfig


T = TypeVar("T")


@dataclass
class OpenAICompatibleClient:
    config: LlmConfig
    timeout: int = 60

    def chat(self, prompt: str, max_tokens: int = 2048) -> str:
        endpoint = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


def retry_call(func: Callable[[], T], max_attempts: int = 3, delay_seconds: float = 1) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            if delay_seconds:
                time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def chat_with_optional_max_tokens(llm_client, prompt: str, max_tokens: int) -> str:
    try:
        return llm_client.chat(prompt, max_tokens=max_tokens)
    except TypeError as exc:
        if "max_tokens" not in str(exc):
            raise
        return llm_client.chat(prompt)
