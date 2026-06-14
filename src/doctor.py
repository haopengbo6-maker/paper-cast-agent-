from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from .config import load_llm_config
from .media_config import load_media_config
from .prompts import load_prompt
from .utils import ensure_project_dirs


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    messages: list[str]
    errors: list[str]

    def format(self) -> str:
        lines = ["PaperCast Agent doctor"]
        lines.extend(f"[OK] {message}" for message in self.messages)
        lines.extend(f"[ERROR] {error}" for error in self.errors)
        lines.append("Status: OK" if self.ok else "Status: FAILED")
        return "\n".join(lines)


def run_doctor(root: Path, check_optional_imports: bool = True) -> DoctorReport:
    messages: list[str] = []
    errors: list[str] = []

    try:
        ensure_project_dirs(root)
        messages.append("Created or verified data directories")
    except Exception as exc:
        errors.append(f"Could not create data directories: {exc}")

    prompt_dir = root / "prompts"
    _check_prompt(prompt_dir / "map_prompt.txt", "{chunk}", messages, errors)
    _check_prompt(prompt_dir / "reduce_prompt.txt", "{summaries}", messages, errors)

    try:
        load_llm_config(root / ".env")
        messages.append("LLM config found")
    except Exception as exc:
        errors.append(str(exc))

    try:
        media_config = load_media_config(root / ".env")
        if media_config.image.enabled:
            if media_config.image.base_url:
                messages.append(f"Image provider configured: {media_config.image.provider}")
            else:
                errors.append("COMFYUI_BASE_URL is required when MEDIA_IMAGE_PROVIDER=comfyui")
        else:
            messages.append("Image provider disabled")

        if media_config.voice.enabled:
            if media_config.voice.provider == "edge_tts" or media_config.voice.base_url:
                messages.append(f"Voice provider configured: {media_config.voice.provider}")
            else:
                errors.append("COSYVOICE_BASE_URL is required when MEDIA_VOICE_PROVIDER=cosyvoice")
        else:
            messages.append("Voice provider disabled")
    except Exception as exc:
        errors.append(str(exc))

    if check_optional_imports:
        if importlib.util.find_spec("markitdown") is None:
            errors.append("MarkItDown is not installed. Run: pip install -r requirements.txt")
        else:
            messages.append("MarkItDown import found")

    return DoctorReport(ok=not errors, messages=messages, errors=errors)


def _check_prompt(path: Path, placeholder: str, messages: list[str], errors: list[str]) -> None:
    try:
        load_prompt(path, required_placeholder=placeholder)
        messages.append(f"Prompt OK: {path.name}")
    except Exception as exc:
        errors.append(str(exc))
