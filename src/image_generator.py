from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib import request

from .media_config import ImageProviderConfig
from .utils import read_text


COVER_PROMPT_VERSION = "v6"


def build_cover_prompt(script: str) -> str:
    title = _section_first_line(script, "# 播报标题") or "PaperCast AI podcast"
    keywords = _section_lines(script, "# 关键词", limit=6)
    keyword_text = ", ".join(line.lstrip("- ").strip() for line in keywords)
    topic_terms = _infer_topic_visual_terms(title, keyword_text)
    return (
        "2D hand-drawn AI research podcast cover illustration, expressive messy line art, "
        "colored pencil sketch mixed with acrylic paint and subtle oil-paint texture, "
        f"paper topic: {title}, core concepts: {keyword_text}, "
        f"must visualize this exact research topic as an artistic conceptual scene: {topic_terms}, "
        "loose graphite construction lines, visible hand strokes, layered pencil texture, "
        "painterly acrylic color blocks, warm paper grain, elegant controlled chaos, "
        "clear composition with one readable idea, art-book editorial cover, "
        "no visible text, no title lettering, no typography, no Chinese characters, "
        "avoid generic scenery, avoid ancient art, subject must match the paper topic"
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

    output = cover_image_path(output_dir, paper_id)
    if output.exists() and not force:
        return output

    output_dir.mkdir(parents=True, exist_ok=True)
    script_text = read_text(script) if isinstance(script, Path) else script
    payload = {"prompt": build_cover_prompt(script_text), "paper_id": paper_id}
    requester = request_image or _request_comfyui_image
    image_bytes = requester(config.base_url, payload, config.timeout_seconds)
    if not image_bytes:
        raise RuntimeError("ComfyUI returned empty image data")
    output.write_bytes(image_bytes)
    return output


def cover_image_path(output_dir: Path, paper_id: str) -> Path:
    return output_dir / f"{paper_id}_cover_{COVER_PROMPT_VERSION}.png"


def select_comfyui_checkpoint(object_info: dict) -> str:
    try:
        names = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Cannot read ComfyUI checkpoint list from /object_info") from exc
    if not names:
        raise RuntimeError(
            "No ComfyUI checkpoint found. Put a .safetensors or .ckpt model in "
            "ComfyUI/models/checkpoints and restart ComfyUI."
        )
    return str(names[0])


def build_sdxl_workflow(checkpoint: str, positive_prompt: str, output_prefix: str) -> dict:
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": positive_prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": (
                    "low quality, blurry, unreadable text, watermark, visible letters, typography, "
                    "jpeg artifacts, noisy, muddy details, flat lighting, malformed geometry, "
                    "3d render, glossy plastic, device interface, control panel, cassette, screen UI, "
                    "cluttered random objects, many spheres, orbit rings, planets, fractal pattern, maze, "
                    "circuit board texture, repetitive stripes, meaningless abstraction, decorative wallpaper, "
                    "Chinese characters, Chinese calligraphy, ink painting, Chinese landscape, "
                    "landscape painting, mountains, river, lake, pagoda, temple, boat, cherry blossom, "
                    "ancient style, traditional painting, travel poster, unrelated scenery, random poster text"
                ),
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": int(time.time() * 1000) % 1_000_000_000,
                "steps": 32,
                "cfg": 6.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["4", 2]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": output_prefix},
        },
    }


def _request_comfyui_image(base_url: str, payload: dict, timeout: int) -> bytes:
    object_info = _get_json(f"{base_url}/object_info", timeout)
    checkpoint = select_comfyui_checkpoint(object_info)
    workflow = build_sdxl_workflow(
        checkpoint=checkpoint,
        positive_prompt=payload["prompt"],
        output_prefix=f"papercast_{payload['paper_id']}",
    )
    prompt_response = _post_json(f"{base_url}/prompt", {"prompt": workflow}, timeout)
    prompt_id = prompt_response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {prompt_response}")

    image_info = _wait_for_comfyui_image(base_url, prompt_id, timeout)
    return _get_bytes(f"{base_url}/view?{urlencode(image_info)}", timeout)


def _wait_for_comfyui_image(base_url: str, prompt_id: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = _get_json(f"{base_url}/history/{prompt_id}", timeout)
        item = history.get(prompt_id)
        if item:
            outputs = item.get("outputs", {})
            for output in outputs.values():
                images = output.get("images") or []
                if images:
                    first = images[0]
                    return {
                        "filename": first["filename"],
                        "subfolder": first.get("subfolder", ""),
                        "type": first.get("type", "output"),
                    }
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for ComfyUI prompt {prompt_id}")


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"ComfyUI request failed at {url}: {exc}") from exc


def _get_json(url: str, timeout: int) -> dict:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"ComfyUI request failed at {url}: {exc}") from exc


def _get_bytes(url: str, timeout: int) -> bytes:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise RuntimeError(f"ComfyUI request failed at {url}: {exc}") from exc


def _section_first_line(text: str, heading: str) -> str:
    lines = _section_lines(text, heading, limit=1)
    return lines[0] if lines else ""


def _infer_topic_visual_terms(title: str, keywords: str) -> str:
    text = f"{title} {keywords}".lower()
    if _contains_any(
        text,
        (
            "flow matching",
            "flow machine",
            "flow model",
            "flow-based",
            "normalizing flow",
            "continuous normalizing flow",
            "optimal transport",
            "vector field",
        ),
    ):
        return (
            "left side shows a compact blue Gaussian noise cloud, right side shows an orange data distribution manifold, "
            "between them 8 to 12 smooth vector arrows follow straight optimal transport paths, "
            "a faint coordinate grid and contour lines show the learned vector field, "
            "minimal ODE trajectory lines connect noise to samples, generative model sampling geometry"
        )
    if "diffusion" in text or "stable diffusion" in text:
        return "noise cloud turning into image samples, denoising trajectory, latent space grid"
    if "controlnet" in text:
        return "conditioned image generation, edge map guidance, neural network control branches"
    if "agent" in text or "tool" in text:
        return "AI agent workflow graph, connected tools, planning nodes, terminal and document icons"
    return "paper-specific concept diagram, neural network blocks, data flow arrows, abstract research visualization"


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


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
