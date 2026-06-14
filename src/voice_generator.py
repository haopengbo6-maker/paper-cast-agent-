from __future__ import annotations

import json
import re
import asyncio
from pathlib import Path
from urllib import request

from .media_config import VoiceProviderConfig
from .tts import _extract_script_body
from .utils import read_text


def build_speech_text(script: str) -> str:
    text = _extract_script_body(script)
    text = text.replace("[uv_break]", "\n")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def generate_voice_audio(
    paper_id: str,
    script: str | Path,
    output_dir: Path,
    config: VoiceProviderConfig,
    force: bool = False,
    request_audio=None,
) -> Path | None:
    if not config.enabled:
        return None
    if config.provider not in ("cosyvoice", "edge_tts"):
        raise RuntimeError(f"Unsupported voice provider: {config.provider}")
    if config.provider == "cosyvoice" and not config.base_url:
        raise RuntimeError("COSYVOICE_BASE_URL is required when MEDIA_VOICE_PROVIDER=cosyvoice")

    suffix = ".mp3" if config.provider == "edge_tts" else ".wav"
    output = output_dir / f"{paper_id}_podcast{suffix}"
    if output.exists() and not force:
        return output

    output_dir.mkdir(parents=True, exist_ok=True)
    script_text = read_text(script) if isinstance(script, Path) else script
    speech_text = build_speech_text(script_text)
    if config.provider == "edge_tts":
        synthesize = request_audio or _synthesize_edge_tts
        synthesize(speech_text, output, config.voice, config.timeout_seconds)
    else:
        payload = {"text": speech_text, "voice": config.voice}
        requester = request_audio or _request_cosyvoice_audio
        audio_bytes = requester(f"{config.base_url}/papercast/tts", payload, config.timeout_seconds)
        if not audio_bytes:
            raise RuntimeError("CosyVoice returned empty audio data")
        output.write_bytes(audio_bytes)
    return output


def _request_cosyvoice_audio(url: str, payload: dict, timeout: int) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise RuntimeError(f"CosyVoice request failed at {url}: {exc}") from exc


def _synthesize_edge_tts(text: str, output: Path, voice: str, timeout: int) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed") from exc

    async def _save() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await asyncio.wait_for(communicate.save(str(output)), timeout=timeout)

    try:
        asyncio.run(_save())
    except Exception as exc:
        raise RuntimeError(f"Edge TTS synthesis failed: {exc}") from exc
