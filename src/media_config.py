from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import load_dotenv_file


@dataclass(frozen=True)
class ImageProviderConfig:
    provider: str
    base_url: str
    timeout_seconds: int

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(frozen=True)
class VoiceProviderConfig:
    provider: str
    base_url: str
    timeout_seconds: int
    voice: str

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(frozen=True)
class MediaConfig:
    image: ImageProviderConfig
    voice: VoiceProviderConfig


def load_media_config(path: Path = Path(".env")) -> MediaConfig:
    load_dotenv_file(path)
    image_provider = _provider("MEDIA_IMAGE_PROVIDER")
    voice_provider = _provider("MEDIA_VOICE_PROVIDER")

    return MediaConfig(
        image=ImageProviderConfig(
            provider=image_provider,
            base_url=_url("COMFYUI_BASE_URL") if image_provider != "none" else "",
            timeout_seconds=_int("COMFYUI_TIMEOUT_SECONDS", 180),
        ),
        voice=VoiceProviderConfig(
            provider=voice_provider,
            base_url=_voice_base_url(voice_provider),
            timeout_seconds=_int("COSYVOICE_TIMEOUT_SECONDS", 180),
            voice=_voice_name(voice_provider),
        ),
    )


def _provider(name: str) -> str:
    value = os.getenv(name, "").strip().lower()
    return value or "none"


def _url(name: str) -> str:
    return os.getenv(name, "").strip().rstrip("/")


def _voice_base_url(provider: str) -> str:
    if provider == "none" or provider == "edge_tts":
        return ""
    return _url("COSYVOICE_BASE_URL")


def _voice_name(provider: str) -> str:
    if provider == "edge_tts":
        return os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural").strip() or "zh-CN-XiaoxiaoNeural"
    return os.getenv("COSYVOICE_VOICE", "default").strip() or "default"


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0")
    return value
