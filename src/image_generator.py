from __future__ import annotations

import json
from pathlib import Path
from urllib import request

from .media_config import ImageProviderConfig
from .utils import read_text


def build_cover_prompt(script: str) -> str:
    title = _section_first_line(script, "# 播报标题") or "PaperCast AI podcast"
    keywords = _section_lines(script, "# 关键词", limit=6)
    keyword_text = ", ".join(line.lstrip("- ").strip() for line in keywords)
    return (
        f"podcast cover art for an AI research episode, title: {title}, "
        f"keywords: {keyword_text}, clean editorial illustration, readable, high contrast"
    )


def generate_cover_image(
    paper_id: str,
    script: str | Path,
    output_dir: Path,
    config: ImageProviderConfig,
    force: bool = False,
    request_image=None,
) -> Path | None:
    if not config.enabled:
        return None
    if config.provider != "comfyui":
        raise RuntimeError(f"Unsupported image provider: {config.provider}")
    if not config.base_url:
        raise RuntimeError("COMFYUI_BASE_URL is required when MEDIA_IMAGE_PROVIDER=comfyui")

    output = output_dir / f"{paper_id}_cover.png"
    if output.exists() and not force:
        return output

    output_dir.mkdir(parents=True, exist_ok=True)
    script_text = read_text(script) if isinstance(script, Path) else script
    payload = {"prompt": build_cover_prompt(script_text), "paper_id": paper_id}
    requester = request_image or _request_comfyui_image
    image_bytes = requester(f"{config.base_url}/papercast/txt2img", payload, config.timeout_seconds)
    if not image_bytes:
        raise RuntimeError("ComfyUI returned empty image data")
    output.write_bytes(image_bytes)
    return output


def _request_comfyui_image(url: str, payload: dict, timeout: int) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise RuntimeError(f"ComfyUI request failed at {url}: {exc}") from exc


def _section_first_line(text: str, heading: str) -> str:
    lines = _section_lines(text, heading, limit=1)
    return lines[0] if lines else ""


def _section_lines(text: str, heading: str, limit: int) -> list[str]:
    lines = text.splitlines()
    result: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == heading:
            in_section = True
            continue
        if in_section and line.startswith("# "):
            break
        if in_section and line.strip():
            result.append(line.strip())
            if len(result) >= limit:
                break
    return result
