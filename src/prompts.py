from __future__ import annotations

from pathlib import Path


def load_prompt(path: Path, required_placeholder: str) -> str:
    if not path.exists():
        raise RuntimeError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if required_placeholder not in text:
        raise RuntimeError(f"Prompt file {path} must contain {required_placeholder}")
    return text
