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


COVER_PROMPT_VERSION = "v13"

_TITLE_HEADINGS = ("# 播报标题", "# 标题", "# title")
_SCRIPT_HEADINGS = ("# 播报脚本", "# 脚本", "# script")
_KEYWORD_HEADINGS = ("# 关键词", "# 关键字", "# keywords")

# ── Artistic style (front-loaded — SDXL weights first tokens most heavily) ──
_STYLE = (
    "archival scientific plate illustration, "
    "colored pencil and gouache on cream-toned archival paper, "
    "printmaking aesthetic with subtle plate-mark registration lines, "
    "letterpress impression texture, editorial art-book composition, "
    "muted scholarly palette: burnt sienna, ochre, indigo ink, olive green accents, "
    "soft paper grain, generous negative space around the subject, "
    "elegant diagrammatic quality, hand-rendered precision"
)

# ── Global constraints ──
_CONSTRAINTS = (
    "clean centered single-read composition, "
    "no visible text, no letters, no typography, no Chinese characters, "
    "no photorealism, no 3D rendering, no glossy plastic, no metallic chrome, "
    "no generic scenery, no ancient art, no random decorative abstraction"
)


# ═══════════════════════════════════════════════════════════════════
#  Keyword → visual element mapping
#  Converts abstract academic terms into concrete, paintable visuals.
#  Each entry maps a lowercase keyword/phrase to a short visual cue.
# ═══════════════════════════════════════════════════════════════════

_KEYWORD_VISUALS: dict[str, str] = {
    # ── AI / ML general ──
    "foundation model": "a unifying architectural backbone connecting diverse capability modules",
    "generalist": "multiple task domains arranged as linked but distinct visual panels",
    "multi-task": "parallel task lanes with shared backbone feeding into specialized heads",
    "transfer learning": "knowledge flowing from a source domain to a target domain as a bridge",
    "few-shot": "a model making correct predictions from only a handful of labeled examples",
    "zero-shot": "a model recognizing an object category it has never seen during training",
    "fine-tuning": "a pre-existing neural structure being refined by a stream of new data points",
    "pre-training": "a vast corpus of data being absorbed into a growing neural representation",
    "scaling law": "a log-log plot showing performance rising with compute and model size",
    "emergent ability": "a sudden capability appearing beyond a critical scale threshold",
    "in-context learning": "a few demonstration examples changing model behavior without weight updates",
    "chain-of-thought": "step-by-step reasoning nodes connected by logical inference arrows",
    "instruction tuning": "natural language commands being mapped to structured model responses",
    "alignment": "model outputs being steered toward human-preferred regions of a value space",
    "rlhf": "reinforcement learning from human feedback shown as a reward signal loop",
    "benchmark": "comparative bar charts across multiple methods with one bar highlighted",
    "ablation": "component removal study shown as importance-ranked bars with drops marked",
    "latent space": "a low-dimensional manifold with clustered point clouds in distinct colors",
    "embedding": "high-dimensional vectors projected into visible 2D space as scattered stars",
    "attention": "highlighted connection bands between related token pairs",
    "self-attention": "a sequence attending to itself with weighted connection density",
    "cross-attention": "two sequences aligned by bridging attention weights",
    "transformer": "stacked layers with multi-head attention and feed-forward blocks",
    "tokenizer": "raw text being split into discrete colored token blocks",
    "mixture of experts": "a router directing inputs to specialized expert sub-networks",
    "moe": "a router directing inputs to specialized expert sub-networks",

    # ── Generative models ──
    "diffusion model": "noise gradually resolving into a clean structured image through denoising steps",
    "denoising": "a noisy image becoming progressively cleaner across horizontal stages",
    "score-based": "gradient fields pointing toward higher data density regions",
    "generative adversarial": "a generator and discriminator locked in a minimax game as two opposing forces",
    "gan": "a generator and discriminator as two opposing forces in equilibrium",
    "vae": "an encoder compressing data into a latent distribution and a decoder reconstructing it",
    "variational autoencoder": "an encoder mapping to a probability distribution, decoder sampling from it",
    "autoregressive": "tokens generated one-by-one in left-to-right sequential order",
    "discrete token": "continuous data being quantized into a codebook of discrete visual tokens",

    # ── Computer Vision ──
    "object detection": "bounding boxes with confidence scores localizing objects in a scene",
    "segmentation": "pixel-precise boundary contours partitioning an image into semantic regions",
    "instance segmentation": "each individual object instance outlined in a distinct color mask",
    "semantic segmentation": "every pixel assigned a category label shown as a color-coded map",
    "panoptic segmentation": "both semantic regions and individual instances shown in unified overlay",
    "pose estimation": "skeleton keypoints connected by bones overlaid on a moving figure",
    "depth estimation": "a scene rendered with distance-based shading from near to far",
    "optical flow": "motion vectors as small arrows showing pixel movement between frames",
    "3d reconstruction": "a 2D image projecting outward into a volumetric 3D form",
    "nerf": "rays passing through a volumetric radiance field with density sampling points",
    "gaussian splatting": "ellipsoid gaussians scattered in 3D space rendering a scene",
    "visual grounding": "language phrases linked by arrows to specific image regions",
    "image captioning": "an image on the left connected to descriptive output tokens on the right",
    "super-resolution": "a low-resolution patch enlarged into a high-resolution detailed version",
    "image inpainting": "a masked region being filled in with context-consistent content",
    "image generation": "a new image materializing from a text description or noise seed",

    # ── Robotics / Embodied AI ──
    "manipulation": "a robotic hand precisely grasping an object with force-feedback markers",
    "locomotion": "bipedal walking trajectory with footstep pressure distribution maps",
    "navigation": "path-planning topology with obstacle avoidance force fields",
    "grasping": "fingertip contact points on an object surface with grasp quality metrics",
    "dexterous": "multi-finger hand with independent joint articulations manipulating a small object",
    "teleoperation": "a human operator motion on the left mirrored by a robot on the right",
    "sim-to-real": "a simulation domain on the left transferring policies to a real robot on the right",
    "domain randomization": "diverse training environments with varying textures lighting and physics",
    "motion planning": "a collision-free path winding through obstacle-filled configuration space",
    "kinematics": "joint angle arcs and end-effector reachable workspace boundaries",
    "dynamics": "force and torque vectors acting on a rigid body with acceleration traces",
    "whole-body control": "coordinated joint torques distributed across the entire robot body",

    # ── Medical ──
    "tumor": "a highlighted irregular mass with spiculated boundary in tissue cross-section",
    "lesion": "an abnormal tissue region marked by a diagnostic contour overlay",
    "mri": "a grayscale brain or body cross-section slice with anatomical annotation marks",
    "ct scan": "sequential axial slice views stacked into a volumetric perspective",
    "x-ray": "a radiographic projection with bones and soft tissue in inverse contrast",
    "ultrasound": "a sonographic image with speckle texture and measurement calipers",
    "pathology": "a stained tissue slide at cellular magnification with region annotations",
    "histology": "tissue microstructure with cell nuclei visible as small stained dots",
    "endoscopy": "an interior tubular organ view with a highlighted polyp or anomaly",
    "diagnosis": "a diagnostic decision tree branching from symptoms to possible conditions",
    "prognosis": "a time-series projection of disease trajectory with confidence bands",
    "survival analysis": "a Kaplan-Meier curve showing survival probability over time",

    # ── Biology / Life Sciences ──
    "protein structure": "a folded polypeptide chain as a 3D ribbon diagram with alpha helices and beta sheets",
    "protein folding": "an unfolded chain progressively collapsing into a compact folded state",
    "dna": "a double helix with complementary base pairs connected by hydrogen bond rungs",
    "rna": "a single-stranded polynucleotide chain folding into secondary structure loops",
    "gene expression": "a heatmap grid of activation levels across multiple conditions and genes",
    "mutation": "a highlighted base-pair substitution in a DNA sequence alignment",
    "cell signaling": "ligand-receptor binding at the membrane triggering a cascade of kinase arrows",
    "metabolic pathway": "a network of enzymatic reaction nodes with substrate-to-product arrows",
    "microbiome": "diverse bacterial colony shapes clustered in a host-environment interface",
    "crispr": "a Cas9 protein with guide RNA targeting a specific genomic locus for editing",
    "single-cell": "individual cells plotted as points in a 2D gene-expression embedding space",

    # ── Physics ──
    "quantum": "wavefunction probability density clouds around an atomic nucleus",
    "entanglement": "two particles connected by a correlated-state line spanning space",
    "superposition": "a single system existing in multiple basis states shown as overlapping ghost images",
    "superconductivity": "electrical resistance dropping abruptly to zero at a critical temperature",
    "phase transition": "a system transforming from one phase to another at a critical threshold",
    "condensed matter": "atoms arranged in a periodic lattice with electron band structure overlay",
    "particle physics": "subatomic particle tracks in a detector with curvature from magnetic field",

    # ── Materials Science ──
    "crystal structure": "atoms arranged in periodic 3D lattice with unit cell boundaries marked",
    "defect": "an irregularity or vacancy highlighted in an otherwise regular crystal lattice",
    "interface": "a boundary layer where two materials meet with atomic interaction zone",
    "microstructure": "polycrystalline grain boundaries with different orientation shading",
    "nanoparticle": "a tiny faceted particle with surface ligand molecules attached",

    # ── Chemistry ──
    "catalyst": "a surface with active sites where reactant molecules dock and transform",
    "reaction mechanism": "step-by-step electron-pushing arrows showing bond-making and bond-breaking",
    "molecular dynamics": "a molecule with motion trails showing vibrational and rotational modes",
    "spectroscopy": "an absorption or emission spectrum with characteristic peak pattern",

    # ── Energy ──
    "solar cell": "a photovoltaic panel cross-section with photon-to-electron conversion diagram",
    "battery": "layered anode-electrolyte-cathode with lithium-ion flow arrows between electrodes",
    "fuel cell": "hydrogen and oxygen inputs with proton-exchange membrane and water output",
    "wind turbine": "three aerodynamic blades with wake flow patterns and generator cutaway",

    # ── Systems / Networks ──
    "distributed system": "multiple computing nodes with message-passing arrows and consensus markers",
    "load balancing": "incoming requests being distributed across server nodes by a dispatcher",
    "caching": "a fast small memory layer intercepting requests before reaching slow storage",
    "database": "a cylinder abstraction with indexed retrieval paths and query execution plan",

    # ── Control / RL ──
    "reinforcement learning": "reward signal flowing through a decision network, value function as a heatmap",
    "policy gradient": "a policy surface with gradient arrows pointing toward higher-reward regions",
    "exploration": "branching trajectories exploring unknown regions before converging to optimal path",
    "regret": "a gap closing between optimal and achieved performance over time steps",

    # ── Math ──
    "optimization": "a loss surface with gradient descent path spiraling into a minimum basin",
    "convergence": "iterates approaching a fixed point with decreasing step sizes visible",
    "manifold": "a curved lower-dimensional surface embedded in higher-dimensional space",
    "topology": "shapes being continuously deformed while preserving connectedness properties",
    "graph": "nodes and edges with degree distribution and community structure visible",

    # ── Generic research concepts ──
    "dataset": "data samples arranged in a structured grid showing diversity and distribution",
    "efficiency": "a gauge or ratio indicator showing performance relative to resource usage",
    "trade-off": "a Pareto frontier curve showing the optimal balance between two competing objectives",
    "robustness": "a system maintaining performance under various perturbation types and magnitudes",
    "generalization": "training distribution on the left and a different test distribution on the right",
    "uncertainty": "predictions shown with confidence intervals or error bars of varying widths",
    "interpretability": "a black-box model being opened to reveal its internal decision logic",
    "distillation": "a large teacher model transferring knowledge to a smaller student model",
    "pruning": "a dense network with inactive connections being removed leaving a sparse substructure",
    "quantization": "continuous weight values being discretized into a reduced set of precision levels",
    "federated learning": "multiple local data silos contributing model updates to a central aggregator",
    "contrastive learning": "positive pairs pulled together and negative pairs pushed apart in embedding space",
    "self-supervised": "a model learning representations from unlabeled data by predicting masked parts",
    "data augmentation": "a single sample transformed into multiple variants through rotations crops and noise",
    "adversarial": "a clean input and its imperceptibly perturbed version producing different outputs",
    "out-of-distribution": "in-distribution samples clustered tightly with OOD samples falling outside",
    "active learning": "a model querying an oracle for labels on the most uncertain or diverse samples",
    "meta-learning": "a meta-learner extracting common patterns across multiple learning episodes",
    "continual learning": "a model sequentially acquiring new skills while retaining old ones without forgetting",
    "causality": "a directed acyclic graph with causal arrows showing cause-effect relationships",
    "attention mechanism": "query-key-value dot-product with softmax weights highlighting relevant positions",
    "normalization": "data being centered and scaled to zero mean and unit variance distributions",
    "regularization": "model complexity being constrained by a penalty term shrinking weight magnitudes",
    "ensemble": "multiple diverse models voting or averaging to produce a combined prediction",
    "knowledge graph": "entities as nodes connected by typed relation edges forming a semantic web",
    "retrieval-augmented": "a query fetching relevant documents from an external knowledge base before generation",
    "rag": "a retriever fetching documents and a generator incorporating them into the output",
    "vector database": "high-dimensional vectors indexed in a searchable space with nearest-neighbor lookup",
    "token": "discrete text units shown as small colored squares flowing through processing layers",
    "open-source": "code and model weights released publicly as an open resource for the community",
}


def build_cover_prompt(script: str, summary_hint: str = "") -> str:
    """Build a paper-specific SDXL prompt: style → title as subject → discipline stage → keyword props → constraints.

    The key insight: the paper's specific title and keywords drive the visual content,
    while the discipline-matched scene provides the stage and visual vocabulary.
    """
    title = _section_first_line(script, *_TITLE_HEADINGS) or "academic research paper"
    raw_keywords = _section_lines(script, _KEYWORD_HEADINGS, limit=8)
    keywords_clean = [line.lstrip("- ").strip() for line in raw_keywords]
    summary_line = _first_meaningful_line(summary_hint)
    script_line = _extract_script_hint(script)

    keyword_text = ", ".join(keywords_clean)

    # Match discipline to get the base scene (stage + visual vocabulary)
    base_scene = _infer_topic_visual_scene(title, keyword_text, summary_line, script_line)

    # Map the paper's specific keywords to concrete visual elements
    keyword_visuals = _map_keywords_to_visuals(keywords_clean)

    # Compose: the title IS the subject, keywords drive the visual details
    subject_clause = f'illustrating this specific research: "{title}"'
    scene_clause = f"scene — {base_scene}"

    detail_clauses: list[str] = []
    if keyword_visuals:
        # Weave keyword visuals into the scene description
        visual_str = "; ".join(keyword_visuals[:5])
        detail_clauses.append(f"this paper's key concepts are visualized: {visual_str}")
        # Remaining keywords as conceptual anchors
        remaining = [kw for kw in keywords_clean if kw.lower() not in _KEYWORD_VISUALS
                     and not any(v.lower().startswith(kw.lower()) for k in _KEYWORD_VISUALS for v in [k])]
        # Actually let's just use all keywords that weren't mapped as direct visual cues
        mapped_lower = {k.lower() for k in _KEYWORD_VISUALS}
        unmapped = [kw for kw in keywords_clean if kw.lower() not in mapped_lower]
        if unmapped:
            detail_clauses.append(f"additional concepts: {', '.join(unmapped[:4])}")
    elif keywords_clean:
        detail_clauses.append(f"concepts: {', '.join(keywords_clean[:5])}")

    if summary_line:
        detail_clauses.append(f"research context: {summary_line[:150]}")

    details = ". ".join(detail_clauses)

    return (
        f"{_STYLE}. "
        f"{subject_clause}. "
        f"{scene_clause}. "
        f"{details}. "
        f"{_CONSTRAINTS}"
    )


def _map_keywords_to_visuals(keywords: list[str]) -> list[str]:
    """Map paper keywords to concrete visual elements using the keyword-visual dictionary.

    Returns a list of strings like "keyword shown as visual_description".
    """
    results: list[str] = []
    seen_visuals: set[str] = set()  # deduplicate similar visuals
    for kw in keywords:
        kw_lower = kw.lower().rstrip("s")  # normalize plurals
        # Try exact match first, then without trailing 's', then substring match
        visual = (
            _KEYWORD_VISUALS.get(kw_lower)
            or _KEYWORD_VISUALS.get(kw.lower())
            or _KEYWORD_VISUALS.get(kw_lower.rstrip("s"))
        )
        if not visual:
            # Fuzzy: check if any dict key is a substring of the keyword or vice versa
            for dict_key, dict_visual in _KEYWORD_VISUALS.items():
                if len(dict_key) >= 4 and (dict_key in kw_lower or kw_lower in dict_key):
                    visual = dict_visual
                    break
        if visual and visual not in seen_visuals:
            results.append(f"{kw} shown as {visual}")
            seen_visuals.add(visual)
        elif not visual:
            # Include the keyword directly — SDXL can interpret many terms natively
            if len(kw) >= 3 and kw.lower() not in {"paper", "method", "result", "study", "research",
                                                     "analysis", "approach", "model", "system", "data",
                                                     "learning", "training", "test", "performance"}:
                results.append(kw)
    return results


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
                "text": _NEGATIVE_PROMPT,
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
                "steps": 35,
                "cfg": 5.5,
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


_NEGATIVE_PROMPT = (
    "low quality, blurry, jpeg artifacts, watermark, signature, "
    "photorealistic, 3d render, CGI, glossy plastic, metallic chrome, "
    "visible text, letters, typography, Chinese characters, calligraphy, "
    "ancient Chinese painting, ink wash landscape, pagoda, temple, "
    "traditional scroll, rice paper, chop seal, 水墨画, 国画, "
    "human skeleton, skull, bones, rib cage, anatomical diagram, "
    "house, building, residential architecture, city street, village, "
    "random abstract pattern, decorative wallpaper, meaningless shapes, "
    "circuit board texture, neon cyberpunk, sci-fi dashboard, device UI, "
    "many spheres, orbit rings, planets as decoration, fractal pattern, maze, "
    "photograph of a person, portrait, face, selfie, stock photo, "
    "cluttered composition, busy background, too many objects"
)


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


# ═══════════════════════════════════════════════════════════════════
#  Topic → visual scene mapping (discipline stage)
#  Each entry provides the STAGE and visual VOCABULARY for a discipline.
#  Paper-specific details (title, keywords) are layered on top by
#  build_cover_prompt().
# ═══════════════════════════════════════════════════════════════════

def _infer_topic_visual_scene(
    title: str, keywords: str, summary_hint: str = "", script_hint: str = ""
) -> str:
    """Return the base visual scene for the paper's discipline."""
    text = f"{title} {keywords} {summary_hint} {script_hint}".lower()

    # ── Flow Matching / Generative Models ──
    if _contains_any(
        text,
        (
            "flow matching", "flow machine", "flow model", "flow-based",
            "normalizing flow", "continuous normalizing flow",
            "optimal transport", "vector field", "向量场", "最优传输", "生成模型",
        ),
    ):
        return (
            "left: Gaussian noise particles dissolving into smooth vector-field streamlines, "
            "right: trajectories converging into a data-distribution manifold, "
            "ODE trajectory lines connecting the two realms, coordinate grid and contour lines tracing the learned mapping"
        )

    topic_blocks: list[tuple[tuple[str, ...], str]] = [
        # ── Humanoid Robots / Embodied AI ──
        (
            (
                "humanoid", "人形机器人", "双足行走", "全身控制", "具身智能",
                "human robot", "human-like robot", "bipedal robot", "bipedal humanoid",
                "android", "embodied intelligence", "embodied ai",
                "robot locomotion", "robot manipulation", "robot grasping",
                "whole-body control", "kinematics", "humanoid motion",
            ),
            (
                "a clearly mechanical humanoid robot as the central figure on a laboratory "
                "motion-capture floor, metal composite shell panels with visible panel gaps, "
                "servo motors at joints, sensor head unit, motion trajectory arcs traced behind "
                "limbs, balance force-vector arrows rising from center of mass, engineering grid "
                "floor; mechanical not biological — no skin, no skeleton, no human anatomy"
            ),
        ),
        # ── Computer Vision ──
        (
            (
                "vision transformer", "计算机视觉", "目标检测", "图像分割", "姿态估计",
                "image recognition", "object detection", "segmentation",
                "pose estimation", "visual grounding", "multimodal",
                "image synthesis", "generative vision", "diffusion",
            ),
            (
                "a photograph in the center being analytically deconstructed: patch-grid overlay, "
                "bounding box around a focal object, segmentation mask contours, attention heatmap "
                "glowing over salient regions, feature activation trails radiating outward to "
                "filter-response thumbnails in the margin"
            ),
        ),
        # ── LLM / Language Models ──
        (
            (
                "llm", "大语言模型", "语言模型", "提示词", "对齐",
                "large language model", "language model", "instruction tuning",
                "alignment", "transformer", "token", "reasoning", "prompt", "rlhf",
            ),
            (
                "stacked transformer layers as horizontal attention bands, token tiles flowing "
                "through them, embedding vectors scattered as a starfield in low-dimensional "
                "projection, prompt block on the left connected by a widening river of meaning "
                "to a response block on the right; no robot face, no chat app, no magic book, "
                "no glowing brain"
            ),
        ),
        # ── Medical / Clinical ──
        (
            (
                "medical", "医学", "临床", "病理", "放射", "手术", "诊断",
                "clinical", "radiology", "pathology", "medical image",
                "healthcare", "disease", "diagnosis", "surgery",
                "biomedical", "bioimaging", "bioinformatics", "genomics", "proteomics",
            ),
            (
                "a radiology scan or pathology slide as the central panel, diagnostic contour "
                "overlays in colored pencil, measurement calipers marking distances, tissue "
                "texture detail at one edge, clinical chart traces and lab markers as marginal "
                "annotations; no hospital room, no doctor portrait, no stethoscope"
            ),
        ),
        # ── Physics ──
        (
            (
                "physics", "量子", "光学", "热力学", "统计物理", "波函数",
                "quantum", "mechanics", "optics", "wave", "field theory",
                "particles", "thermodynamics", "statistical physics",
                "astronomy", "astrophysics",
            ),
            (
                "wave interference as the main motif — concentric ripples from two sources "
                "crossing into a standing pattern, particle trajectories as dashed arrow lines, "
                "field lines as continuous flowing curves, abstract equation fragments as "
                "unreadable symbolic marks, coordinate axes; no fantasy space art, no planets "
                "as decoration, no landscape"
            ),
        ),
        # ── Materials Science ──
        (
            (
                "materials", "材料", "晶体", "聚合物", "半导体", "电池", "催化",
                "material", "crystal", "polymer", "nanomaterial", "semiconductor",
                "battery", "electrode", "catalyst", "surface", "microstructure",
            ),
            (
                "crystal lattice as an isometric 3D construction of spheres connected by rods "
                "in repeating unit cells, a cross-section slice at the bottom revealing grain "
                "boundaries as irregular polygons, electron-microscope texture at one corner, "
                "electrode or catalyst interface as a horizontal boundary with reaction arrows; "
                "no buildings, no furniture, no circuit board wallpaper"
            ),
        ),
        # ── Biology / Life Sciences ──
        (
            (
                "biology", "生物", "基因", "蛋白", "细胞", "分子", "神经科学",
                "genomics", "protein", "cell", "molecule", "molecular",
                "neuroscience", "gene", "pathway", "biochemical", "bioinformatics",
            ),
            (
                "a single cell as the central subject — rounded form with visible nucleus and "
                "organelle silhouettes in muted colors, molecular pathway arrows connecting "
                "internal components, DNA double-helix trace curling through one region, assay "
                "well-plate grid as faint background, gene-expression heatmap blocks in the "
                "margin; no human skeleton, no full-body anatomy, no garden landscape"
            ),
        ),
        # ── Social Science / Economics / Policy ──
        (
            (
                "economics", "经济", "金融", "社会", "政策", "行为", "调查",
                "finance", "market", "social", "sociology", "policy",
                "behavior", "survey", "education", "humanities", "politics", "psychology",
            ),
            (
                "population nodes as circles of varying sizes across the composition, "
                "social-network edges connecting them, survey response data as clustered bar "
                "charts rising from each node, policy intervention as a color shift propagating "
                "through the network, timeline bands marking before-and-after periods; no office "
                "buildings, no money piles, no portraits, no city skyline"
            ),
        ),
        # ── Geomagnetic Storm / Space Weather ──
        (
            (
                "geomagnetic storm", "geomagnetic", "magnetic storm", "space weather",
                "solar storm", "solar wind", "magnetosphere", "ionosphere",
                "aurora", "coronal mass ejection", "cme",
                "地磁暴", "地磁", "空间天气", "太阳风", "磁层", "电离层", "极光", "日冕物质抛射",
            ),
            (
                "the Sun on the left emitting a coronal mass ejection arc, solar wind streamlines "
                "flowing rightward toward Earth, Earth's magnetosphere as protective blue magnetic "
                "field lines bending around the planet, aurora oval glowing near the polar region, "
                "satellite orbit markers and magnetometer graph traces in the margin; no houses, "
                "no buildings, no landscape painting"
            ),
        ),
        # ── Climate / Earth System / Remote Sensing ──
        (
            (
                "climate", "气候", "天气", "地球", "海洋", "生态", "遥感",
                "weather", "earth", "geology", "ocean", "environment",
                "ecology", "remote sensing", "urban", "hydrology",
                "sustainability", "carbon",
            ),
            (
                "Earth as a blue marble sphere with atmospheric layer bands wrapping around it, "
                "temperature contour isotherms tracing across the globe, satellite sensor swath "
                "as a trapezoidal footprint on the surface, carbon flux arrows between land and "
                "ocean, topographic relief in muted earth tones; no houses, no villages, no roads"
            ),
        ),
        # ── Energy Systems ──
        (
            (
                "energy", "能源", "电力", "电网", "renewable", "solar", "wind",
                "photovoltaic", "storage system", "grid",
            ),
            (
                "power grid as an organic network of nodes connected by transmission lines, "
                "solar panel cells as a geometric blue pattern on one side, wind turbine as "
                "elegant blades on the other, energy storage as stacked block modules, "
                "power-flow arrows following grid paths, load-curve chart as a marginal graph; "
                "no houses as main subject, no residential street, no city skyline"
            ),
        ),
        # ── Systems / Networks / Distributed ──
        (
            (
                "systems", "系统", "网络", "分布式", "通信", "调度", "数据库",
                "network", "distributed", "communication", "scheduling",
                "database", "storage", "compiler", "operating system",
                "architecture", "algorithm", "infrastructure",
            ),
            (
                "modular computing blocks in a clear layered architecture, routing lines "
                "connecting them, queue-depth indicators as stack heights, database cylinder "
                "at the foundation, scheduler lanes as parallel tracks with moving tasks, "
                "pipeline arrows showing data flow; each block has labeled-looking but "
                "unreadable surface detail; no physical buildings, no city networks, no "
                "circuit-board wallpaper, no glowing dashboards"
            ),
        ),
        # ── Control / Reinforcement Learning ──
        (
            (
                "control", "控制", "强化学习", "规划", "轨迹", "优化",
                "reinforcement learning", "rl", "robot learning", "planning",
                "policy", "control theory", "trajectory", "optimization",
            ),
            (
                "a feedback loop as the central motif: state→action→reward→next state drawn "
                "as a continuous cycle, trajectory curves branching from start to goal regions, "
                "a policy surface with gradient arrows pointing uphill toward the optimum, "
                "controller block with input-output signal traces; no human skeleton, no houses, "
                "no abstract maze"
            ),
        ),
        # ── Chemistry ──
        (
            (
                "chemistry", "化学", "分子反应", "合成", "催化反应",
                "reaction", "synthesis", "molecular interaction", "spectroscopy",
            ),
            (
                "a molecular structure as central sculptural form — ball-and-stick model with "
                "bond angles, two molecules approaching with a dashed transition-state line, "
                "reaction pathway as arched arrows from reactants to products, energy profile "
                "curve beneath showing activation barrier and exothermic drop, spectroscopy "
                "trace as marginal graph; no kitchen, no medicine bottles, no floral patterns"
            ),
        ),
        # ── Mathematics ──
        (
            (
                "math", "数学", "定理", "证明", "代数", "几何", "拓扑", "矩阵", "概率", "统计",
                "graph theory", "linear algebra", "calculus", "theorem", "proof",
                "matrix", "spectral",
            ),
            (
                "a geometric proof construction as the central subject — triangle with auxiliary "
                "construction lines in dashed strokes, circle with inscribed polygons and secant "
                "lines, algebraic symbol fragments as abstract calligraphic marks, matrix grid "
                "with highlighted entries, probability distribution as a smooth bell curve with "
                "shaded tails; no buildings, no books as main subject, no chalkboard classroom"
            ),
        ),
        # ── Law ──
        (
            (
                "law", "法律", "法学", "司法", "法庭", "合规",
                "regulation", "legal", "court", "justice", "policy analysis",
            ),
            (
                "legal documents as layered paper planes with colored citation threads connecting "
                "passages, a balance scale as the central weighing metaphor tilting slightly, "
                "regulation flowchart as tributary streams merging into a main channel, case "
                "timeline as a horizontal band with decision points, institutional seal-like "
                "circular mark; no courthouse building, no judge portrait, no prison"
            ),
        ),
        # ── Agriculture ──
        (
            (
                "agriculture", "农业", "作物", "土壤", "planting", "crop",
                "yield", "farm", "irrigation", "remote farming",
            ),
            (
                "crop rows receding in perspective across the upper composition, a soil "
                "cross-section cutaway at the bottom revealing distinct horizon layers with "
                "root systems penetrating downward, irrigation water flow as blue dashed lines, "
                "sensor-marker icons at sampling points, yield bar chart in the margin; "
                "no farmhouse, no barn, no rural sunset landscape"
            ),
        ),
    ]

    for terms, description in topic_blocks:
        if _contains_any(text, terms):
            return description

    # ── Fallbacks ──
    if "agent" in text or "tool" in text or "智能体" in text:
        return (
            "an AI agent workflow as interconnected tool nodes with planning arrows, central "
            "reasoning block connected to document icons, code snippets, and database symbols, "
            "execution traces as step-by-step horizontal lanes"
        )
    return (
        "the core concept rendered as a meaningful abstract composition — domain-relevant "
        "symbolic objects, data-flow arrows connecting key ideas, clean editorial diagram "
        "capturing the research contribution"
    )


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
