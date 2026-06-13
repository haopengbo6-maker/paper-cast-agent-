from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LlmConfig:
    api_key: str
    base_url: str
    model: str


def load_dotenv_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def load_llm_config(path: Path = Path(".env")) -> LlmConfig:
    load_dotenv_file(path)
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("LLM_MODEL", "").strip()

    missing = [
        name
        for name, value in (
            ("LLM_API_KEY", api_key),
            ("LLM_BASE_URL", base_url),
            ("LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing LLM config: {', '.join(missing)}. Please check .env")

    return LlmConfig(api_key=api_key, base_url=base_url, model=model)
