"""Local TTS audio generation using Windows built-in voices (no network needed)."""

from __future__ import annotations

import re
import threading
from pathlib import Path


VOICES = {
    "huihui": 0,    # Microsoft Huihui — Chinese female, gentle
}


def generate_audio(
    script_path: Path,
    output_dir: Path | None = None,
    voice: str = "huihui",
    rate: int = 160,
) -> Path:
    """Convert a podcast script .md to WAV audio using Windows TTS.

    Extracts only the 播报脚本 section and replaces [uv_break] markers
    with commas for natural pauses (SAPI doesn't support SSML).
    """
    if output_dir is None:
        output_dir = script_path.parent.parent / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    text = script_path.read_text(encoding="utf-8")
    script_body = _extract_script_body(text)

    # Replace pause markers: [uv_break] -> newline + comma for SAPI pause
    script_body = script_body.replace("[uv_break]", "\n")

    # Strip markdown formatting that doesn't read well
    script_body = re.sub(r"\*\*(.*?)\*\*", r"\1", script_body)  # bold
    script_body = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", script_body)  # links

    stem = script_path.stem
    audio_id = stem.replace("_script", "")
    output = output_dir / f"{audio_id}_podcast.wav"

    _synthesize_sapi(script_body, output, rate)
    return output


def _synthesize_sapi(text: str, output: Path, rate: int) -> None:
    """Run pyttsx3 in a dedicated thread to avoid event loop conflicts."""
    result = {"error": None}

    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Use Chinese voice (index 0 = Huihui)
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
            engine.setProperty("rate", rate)
            engine.save_to_file(text, str(output))
            engine.runAndWait()
        except Exception as exc:
            result["error"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=120)

    if result["error"]:
        raise RuntimeError(f"TTS failed: {result['error']}") from result["error"]

    if not output.exists():
        raise RuntimeError(f"TTS failed: output file not created: {output}")


def _extract_script_body(text: str) -> str:
    """Extract only the # 播报脚本 section."""
    match = re.search(r"# 播报脚本\s*\n(.*?)(?=\n# (?:关键词|适合延伸|播报标题))", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = text.split("\n")
    body_lines = []
    in_script = False
    for line in lines:
        if line.startswith("# 播报脚本"):
            in_script = True
            continue
        if in_script:
            if line.startswith("# ") and not line.startswith("## "):
                break
            body_lines.append(line)
    return "\n".join(body_lines).strip() or text
