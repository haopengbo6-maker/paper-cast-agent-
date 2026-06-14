from __future__ import annotations

import json
import re
import time
from pathlib import Path
from io import BytesIO
from urllib.parse import urlencode
from urllib import request

from PIL import Image, ImageEnhance, ImageFilter

from .media_config import ImageProviderConfig
from .utils import read_text


COVER_PROMPT_VERSION = "v10"

_TITLE_HEADINGS = ("# 播报标题", "# 标题", "# title")
_SCRIPT_HEADINGS = ("# 播报脚本", "# 脚本", "# script")
_KEYWORD_HEADINGS = ("# 关键词", "# 关键字", "# keywords")

def build_cover_prompt(script: str, summary_hint: str = "") -> str:
    title = _section_first_line(script, *_TITLE_HEADINGS) or "PaperCast AI podcast"
    keywords = _section_lines(script, _KEYWORD_HEADINGS, limit=6)
    keyword_text = ", ".join(line.lstrip("- ").strip() for line in keywords)
    summary_line = _first_meaningful_line(summary_hint)
    script_line = _extract_script_hint(script)
    summary_text = f"summary cue: {summary_line}, " if summary_line else ""
    script_text = f"script cue: {script_line}, " if script_line else ""
    topic_terms = _infer_topic_visual_terms(title, keyword_text, summary_line, script_line)
    return (
        "2D art-book research plate cover, matching a warm editorial catalogue UI, "
        "expressive messy line art with controlled composition, colored pencil sketch, "
        "graphite construction lines, acrylic gouache blocks, restrained oil-paint texture, "
        f"paper topic: {title}, core concepts: {keyword_text}, "
        f"{summary_text}"
        f"{script_text}"
        f"must visualize this exact research topic as one meaningful conceptual scene: {topic_terms}, "
        "cream archival paper, subtle paper grain, letterpress impression, faint binding gutter, "
        "thin print registration lines, plate-margin crop marks, hand-drawn diagram marks, "
        "muted terracotta, ink brown, mineral blue and olive accents, no glossy realism, "
        "one central readable idea, elegant controlled chaos, curated art catalogue plate, "
        "flat 2D illustration, tactile printed-paper feeling, generous negative space, "
        "no visible text, no title lettering, no typography, no Chinese characters, "
        "avoid generic scenery, avoid ancient art, avoid random decorative abstraction, "
        "avoid photorealism, avoid 3D render, subject must match the paper topic"
    )


def generate_cover_image(
    paper_id: str,
    script: str | Path,
    output_dir: Path,
    config: ImageProviderConfig,
    force: bool = False,
    request_image=None,
    summary_hint: str = "",
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
    summary_hint = summary_hint or _extract_summary_hint(script_text)
    payload = {"prompt": build_cover_prompt(script_text, summary_hint), "paper_id": paper_id}
    requester = request_image or _request_comfyui_image
    image_bytes = requester(config.base_url, payload, config.timeout_seconds)
    if not image_bytes:
        raise RuntimeError("ComfyUI returned empty image data")
    output.write_bytes(_apply_print_finish(image_bytes))
    return output


def cover_image_path(output_dir: Path, paper_id: str) -> Path:
    return output_dir / f"{paper_id}_cover_{COVER_PROMPT_VERSION}.png"


def _apply_print_finish(image_bytes: bytes) -> bytes:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            softened = image.filter(ImageFilter.GaussianBlur(0.25))
            grain = Image.effect_noise(image.size, 5).convert("L")
            grain_rgb = Image.merge("RGB", (grain, grain, grain))
            blended = Image.blend(softened, grain_rgb, 0.035)
            blended = ImageEnhance.Contrast(blended).enhance(1.02)
            blended = ImageEnhance.Sharpness(blended).enhance(0.96)
            output = BytesIO()
            blended.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except Exception:
        return image_bytes


def _extract_summary_hint(script: str) -> str:
    lines = script.splitlines()
    for index, line in enumerate(lines):
        if _matches_heading(line, _SCRIPT_HEADINGS):
            for candidate in lines[index + 1:]:
                candidate = candidate.strip()
                if candidate and not candidate.startswith("#"):
                    return candidate[:180]
            break
    return ""


def _extract_script_hint(script: str) -> str:
    lines = script.splitlines()
    for index, line in enumerate(lines):
        if _matches_heading(line, _SCRIPT_HEADINGS):
            for candidate in lines[index + 1:]:
                candidate = candidate.strip()
                if candidate and not candidate.startswith("#"):
                    return candidate[:180]
            return ""

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("-"):
            return stripped[:180]
    return ""


def _first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


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
                    "photorealistic, 3d render, glossy plastic, metallic chrome, device interface, "
                    "control panel, cassette, screen UI, sci-fi dashboard, neon cyberpunk poster, "
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


def _section_first_line(text: str, *headings: str) -> str:
    lines = _section_lines(text, headings, limit=1)
    return lines[0] if lines else ""


def _infer_topic_visual_terms(title: str, keywords: str, summary_hint: str = "", script_hint: str = "") -> str:
    text = f"{title} {keywords} {summary_hint} {script_hint}".lower()
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
            "向量场",
            "最优传输",
            "生成模型",
        ),
    ):
        return (
            "left side shows a compact blue Gaussian noise cloud, right side shows an orange data distribution manifold, "
            "between them 8 to 12 smooth vector arrows follow straight optimal transport paths, "
            "a faint coordinate grid and contour lines show the learned vector field, "
            "minimal ODE trajectory lines connect noise to samples, generative model sampling geometry"
        )

    topic_blocks = [
        (
            (
                "humanoid", "\u4eba\u5f62\u673a\u5668\u4eba", "\u53cc\u8db3\u884c\u8d70", "\u5168\u8eab\u63a7\u5236", "\u5177\u8eab\u667a\u80fd",
                "human robot", "human-like robot", "bipedal robot", "bipedal humanoid", "android",
                "embodied intelligence", "embodied ai", "robot locomotion", "robot manipulation", "robot grasping",
                "whole-body control", "kinematics", "humanoid motion",
            ),
            "a front-facing humanoid robot with clear head, torso, arms, hands, hips, knees and feet, show articulated joints and human-like proportions rather than a generic machine, place the figure in a calm research-lab or motion-analysis setting with floor markers, reference lines, or a test rig, emphasize locomotion, balance, reach, grasp, and whole-body control, use one central readable pose, subtle mechanical silhouette, and a sense of embodied intelligence",
        ),
        (
            (
                "vision transformer", "\u8ba1\u7b97\u673a\u89c6\u89c9", "\u76ee\u6807\u68c0\u6d4b", "\u56fe\u50cf\u5206\u5272", "\u59ff\u6001\u4f30\u8ba1",
                "image recognition", "object detection", "segmentation", "pose estimation", "visual grounding",
                "multimodal", "diffusion", "image synthesis", "generative vision",
            ),
            "layered image patches, bounding boxes, heatmaps, segmentation contours, cross-attention traces, sample thumbnails, and a clear vision-research diagram with one focal object",
        ),
        (
            (
                "llm", "\u5927\u8bed\u8a00\u6a21\u578b", "\u8bed\u8a00\u6a21\u578b", "\u63d0\u793a\u8bcd", "\u5bf9\u9f50",
                "large language model", "language model", "instruction tuning", "alignment", "transformer", "token", "reasoning", "prompt", "rlhf",
            ),
            "a text-and-token research diagram with stacked attention bands, prompt blocks, arrowed token flow, and a calm editorial layout around a central language-model schematic",
        ),
        (
            (
                "medical", "\u533b\u5b66", "\u4e34\u5e8a", "\u75c5\u7406", "\u653e\u5c04", "\u624b\u672f", "\u8bca\u65ad",
                "clinical", "radiology", "pathology", "medical image", "healthcare", "disease", "diagnosis", "surgery", "anatomy", "biomedical", "bioimaging", "bioinformatics", "genomics", "proteomics",
            ),
            "a clinical paper plate with scan-like silhouettes, diagnostic overlays, anatomical contours, restrained chart markings, and a sober research-lab atmosphere",
        ),
        (
            (
                "physics", "\u91cf\u5b50", "\u5149\u5b66", "\u70ed\u529b\u5b66", "\u7edf\u8ba1\u7269\u7406", "\u6ce2\u51fd\u6570",
                "quantum", "mechanics", "optics", "wave", "field theory", "particles", "thermodynamics", "statistical physics", "astronomy", "astrophysics",
            ),
            "a physics plate with fields, trajectories, wave fronts, equation fragments, and a clean diagrammatic composition with measured scientific tension",
        ),
        (
            (
                "materials", "\u6750\u6599", "\u6676\u4f53", "\u805a\u5408\u7269", "\u534a\u5bfc\u4f53", "\u7535\u6c60", "\u50ac\u5316",
                "material", "crystal", "polymer", "nanomaterial", "semiconductor", "battery", "electrode", "catalyst", "surface", "microstructure",
            ),
            "layered material cross-sections, lattice hints, grain boundaries, surface textures, and a restrained lab diagram with specimen-like clarity",
        ),
        (
            (
                "biology", "\u751f\u7269", "\u57fa\u56e0", "\u86cb\u767d", "\u7ec6\u80de", "\u5206\u5b50", "\u795e\u7ecf\u79d1\u5b66",
                "genomics", "protein", "cell", "molecule", "molecular", "neuroscience", "gene", "pathway", "biochemical", "bioinformatics",
            ),
            "a life-science plate with cellular forms, molecular traces, pathway arcs, and a clean microscope-like composition with biological layering",
        ),
        (
            (
                "economics", "\u7ecf\u6d4e", "\u91d1\u878d", "\u793e\u4f1a", "\u653f\u7b56", "\u884c\u4e3a", "\u8c03\u67e5",
                "finance", "market", "social", "sociology", "policy", "behavior", "survey", "education", "humanities", "politics", "psychology",
            ),
            "an analytical editorial plate with charts, abstract human-scale markers, timeline bands, and a restrained social-research composition",
        ),
        (
            (
                "climate", "\u6c14\u5019", "\u5929\u6c14", "\u5730\u7403", "\u6d77\u6d0b", "\u751f\u6001", "\u9065\u611f",
                "weather", "earth", "geology", "ocean", "environment", "ecology", "remote sensing", "urban", "hydrology", "sustainability", "carbon",
            ),
            "an earth-system plate with topographic bands, atmospheric layers, contour maps, and a measured environmental-research composition",
        ),
        (
            (
                "energy", "\u80fd\u6e90", "\u7535\u529b", "\u7535\u7f51", "renewable", "solar", "wind", "photovoltaic", "storage system", "grid",
            ),
            "layered energy-system diagrams, grid lines, power flow arrows, solar and wind motifs, and a technical sustainability plate",
        ),
        (
            (
                "systems", "\u7cfb\u7edf", "\u7f51\u7edc", "\u5206\u5e03\u5f0f", "\u901a\u4fe1", "\u8c03\u5ea6", "\u6570\u636e\u5e93",
                "network", "distributed", "communication", "scheduling", "database", "storage", "compiler", "operating system", "architecture", "algorithm", "infrastructure",
            ),
            "modular blocks, routing lines, stack layers, system-call or pipeline diagrams, and a clean archival-technical plate composition",
        ),
        (
            (
                "control", "\u63a7\u5236", "\u5f3a\u5316\u5b66\u4e60", "\u89c4\u5212", "\u8f68\u8ff9", "\u4f18\u5316",
                "reinforcement learning", "rl", "robot learning", "planning", "policy", "control theory", "trajectory", "optimization",
            ),
            "trajectory lines, control curves, policy arrows, state-to-action diagrams, and a disciplined technical composition with one visible system loop",
        ),
        (
            (
                "chemistry", "\u5316\u5b66", "\u5206\u5b50\u53cd\u5e94", "\u5408\u6210", "\u50ac\u5316\u53cd\u5e94", "reaction", "synthesis", "molecular interaction", "spectroscopy",
            ),
            "chemical bonds, reaction pathways, molecule contours, catalyst surfaces, and a restrained lab plate with analytical diagram logic",
        ),
        (
            (
                "math", "\u6570\u5b66", "\u5b9a\u7406", "\u8bc1\u660e", "\u4ee3\u6570", "\u51e0\u4f55", "\u62d3\u6251", "\u77e9\u9635", "\u6982\u7387", "\u7edf\u8ba1",
                "graph theory", "linear algebra", "calculus", "theorem", "proof", "matrix", "spectral",
            ),
            "clean mathematical diagrams, theorem blocks, geometric constructions, matrix grids, contour curves, and a disciplined abstract research plate",
        ),
        (
            (
                "law", "\u6cd5\u5f8b", "\u6cd5\u5b66", "\u53f8\u6cd5", "\u6cd5\u5ead", "\u5408\u89c4",
                "regulation", "legal", "court", "justice", "policy analysis",
            ),
            "editorial legal-document composition, scales of justice, citation bands, document blocks, and a formal institutional research plate",
        ),
        (
            (
                "agriculture", "\u519c\u4e1a", "\u4f5c\u7269", "\u571f\u58e4", "planting", "crop", "yield", "farm", "irrigation", "remote farming",
            ),
            "field contour lines, crop rows, soil layers, irrigation traces, and a grounded agricultural research plate",
        ),
    ]
    for terms, description in topic_blocks:
        if _contains_any(text, terms):
            return description
    if "agent" in text or "tool" in text or "智能体" in text:
        return "AI agent workflow graph, connected tools, planning nodes, terminal and document icons"
    return "paper-specific concept diagram, meaningful domain-aware objects, data flow arrows, abstract research visualization"
def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if _is_ascii_term(term):
            pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
            if re.search(pattern, text):
                return True
        elif term in text:
            return True
    return False


def _is_ascii_term(term: str) -> bool:
    return all(ord(char) < 128 for char in term)


def _section_lines(text: str, headings: str | tuple[str, ...], limit: int) -> list[str]:
    lines = text.splitlines()
    result: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if _matches_heading(stripped, headings):
            in_section = True
            continue
        if in_section and stripped.startswith("# "):
            break
        if in_section and stripped:
            result.append(stripped)
            if len(result) >= limit:
                break
    return result


def _matches_heading(line: str, headings: str | tuple[str, ...]) -> bool:
    candidates = (headings,) if isinstance(headings, str) else headings
    normalized = line.strip().lower()
    return any(normalized == candidate.strip().lower() for candidate in candidates)
