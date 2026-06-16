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


COVER_PROMPT_VERSION = "v15"

_TITLE_HEADINGS = ("# 播报标题", "# 标题", "# title")
_SCRIPT_HEADINGS = ("# 播报脚本", "# 脚本", "# script")
_KEYWORD_HEADINGS = ("# 关键词", "# 关键字", "# keywords")

# ── Artistic style — compact, front-loaded for SDXL ──
_STYLE = (
    "archival scientific plate, colored pencil and gouache on cream paper, "
    "printmaking aesthetic with plate-mark registration lines, "
    "letterpress impression, editorial art-book layout, "
    "palette: burnt sienna, ochre, indigo ink, olive green, "
    "soft paper grain, generous negative space"
)

# ── Global constraints ──
_CONSTRAINTS = (
    "clean centered single-read composition, "
    "no visible text, no letters, no typography, no Chinese characters, "
    "no photorealism, no 3D rendering, no glossy plastic, no metallic chrome, "
    "no generic scenery, no ancient art, no random decorative abstraction"
)


# ═══════════════════════════════════════════════════════════════════
#  Keyword → visual element mapping (~150 entries)
#  Converts abstract academic terms into concrete, paintable visuals.
# ═══════════════════════════════════════════════════════════════════

_KEYWORD_VISUALS: dict[str, str] = {
    # ── Robot embodiments ──
    "humanoid robot": "a clearly mechanical humanoid figure with metal composite shell panels, servo joints, sensor head, and cable hints — not biological, no skin, no skeleton",
    "humanoid": "a clearly mechanical humanoid figure with metal composite shell panels, servo joints, and sensor head — not biological, no skin, no skeleton",
    "bipedal robot": "a two-legged robot mid-stride with leg joint trajectories and footstep pressure maps",
    "quadruped robot": "a four-legged robot with articulated leg joints and terrain contact sensors",
    "robot arm": "a multi-axis industrial robot arm with joint angle arcs and end-effector pose markers",
    "mobile robot": "a wheeled or tracked robot platform with sensor array and navigation path overlay",
    "drone": "a multi-rotor aerial vehicle with flight path trajectory and onboard camera payload",
    "soft robot": "a compliant flexible-bodied robot with pneumatic or tendon-driven actuation channels",
    "bimanual": "two robotic arms coordinating together to manipulate a shared object",
    "aloha": "two dexterous robotic arms performing fine-grained bimanual tasks over a tabletop workspace",
    "cross-embodied": "multiple different robot forms sharing a common learned policy across distinct embodiments",
    "aviation": "an aerial vehicle or drone in flight with trajectory path and onboard perception sensors",
    "quadrotor": "a four-rotor aerial vehicle with flight dynamics and onboard camera payload",

    # ── AI / ML core ──
    "foundation model": "a unifying architectural backbone feeding into diverse downstream capability modules arranged radially",
    "generalist": "multiple distinct task domains arranged as linked visual panels sharing a common backbone",
    "multi-task": "parallel task-specific heads branching from a shared neural trunk",
    "transfer learning": "knowledge flowing as a colored bridge from source domain on left to target domain on right",
    "few-shot": "a model making correct predictions from only a handful of labeled support examples",
    "zero-shot": "a model recognizing an unseen category represented only by a semantic description",
    "fine-tuning": "a pre-existing neural structure being refined by an incoming stream of new data points",
    "pre-training": "a vast corpus of varied data being absorbed into a growing neural representation",
    "scaling law": "a log-log plot showing performance rising predictably with compute and model size axes",
    "emergent ability": "a sudden capability spike appearing beyond a critical scale threshold line",
    "in-context learning": "a few demonstration exemplars in a prompt reshaping model behavior without weight updates",
    "chain-of-thought": "step-by-step reasoning nodes connected by logical deduction arrows toward a conclusion",
    "instruction tuning": "natural language commands being mapped to structured task-completion outputs",
    "alignment": "model outputs being steered toward human-preferred regions in a value space diagram",
    "rlhf": "a reward signal loop from human preference judgments back into policy optimization",
    "benchmark": "comparative bar chart across methods with one bar prominently highlighted as best",
    "ablation": "component removal study shown as a descending importance-ranked bar chart",
    "latent space": "a low-dimensional manifold projection with distinct colored point-cloud clusters",
    "embedding": "high-dimensional vectors projected into visible 2D space as scattered constellation points",
    "attention": "weighted connection bands between query and key positions, brighter for stronger links",
    "self-attention": "a sequence attending to its own elements with weighted connection density patterns",
    "cross-attention": "two sequences of different lengths aligned by bridging attention weight bands",
    "transformer": "stacked layers with multi-head attention and feed-forward blocks in alternation",
    "token": "discrete text units rendered as small colored squares flowing through processing layers",
    "tokenizer": "raw text fragmenting into discrete colored token blocks along a segmentation boundary",
    "mixture of experts": "a router dispersing incoming tokens across specialized expert sub-networks in parallel",
    "moe": "a router dispersing tokens across specialized expert sub-networks in parallel",

    # ── Generative models ──
    "diffusion model": "noise gradually resolving into a clean structured image across horizontal denoising stages",
    "diffusion": "noise gradually resolving into a clean structured image across horizontal denoising stages",
    "denoising": "a noisy image becoming progressively cleaner and sharper across a sequence of steps",
    "score-based": "gradient vector fields pointing from noisy regions toward higher data-density regions",
    "gan": "a generator and discriminator shown as two opposing forces reaching equilibrium",
    "generative adversarial": "a generator and discriminator locked in a minimax game as opposing diagrams",
    "vae": "an encoder compressing to a latent bottleneck distribution and a decoder reconstructing the output",
    "variational autoencoder": "an encoder mapping to a gaussian latent distribution, decoder sampling and reconstructing",
    "autoregressive": "tokens predicted one-by-one in left-to-right sequential order with probability bars",
    "discrete token": "continuous data being vector-quantized into a codebook of discrete visual tokens",

    # ── Computer Vision ──
    "object detection": "bounding boxes with confidence scores localizing objects across a scene image",
    "segmentation": "pixel-precise boundary contours partitioning an image into semantically labeled regions",
    "instance segmentation": "each individual object instance outlined in a distinct color mask overlay",
    "semantic segmentation": "every pixel assigned a category label rendered as a color-coded segmentation map",
    "panoptic segmentation": "semantic regions and individual instances unified in a single overlay visualization",
    "pose estimation": "skeleton keypoints connected by bone links overlaid on a moving human or animal figure",
    "depth estimation": "a scene rendered with warm-to-cool distance-based shading from near foreground to far background",
    "optical flow": "dense motion vector arrows showing pixel displacement direction and magnitude between frames",
    "3d reconstruction": "multiple 2D views projecting into a unified volumetric 3D form with surface mesh",
    "nerf": "rays passing through a volumetric radiance field with density sampling points along each ray",
    "gaussian splatting": "ellipsoid 3D gaussians scattered in space collectively rendering a coherent scene",
    "visual grounding": "language phrase arrows pointing to the corresponding highlighted image regions",
    "image captioning": "an input image on the left connected by generation arrows to descriptive output tokens on the right",
    "super-resolution": "a low-resolution pixelated patch enlarged side-by-side into a high-resolution detailed version",
    "image inpainting": "a masked erased region being seamlessly filled with context-consistent content",
    "image generation": "a new image materializing from a text prompt or noise seed through iterative refinement",

    # ── Robotics skills ──
    "manipulation": "a robotic hand or gripper precisely grasping an object with fingertip force-feedback markers",
    "locomotion": "bipedal walking trajectory with footstep pressure distribution maps on the ground plane",
    "navigation": "path-planning topology with optimal route highlighted and obstacle avoidance force fields",
    "grasping": "fingertip contact points distributed on an object surface with grasp quality score indicators",
    "dexterous": "a multi-finger robotic hand with independent joint articulations manipulating a small object",
    "teleoperation": "a human operator motion capture on the left mirrored by a robot executing the same motion on the right",
    "sim-to-real": "a simulation domain on the left transferring learned policies across a domain gap to a real robot on the right",
    "domain randomization": "diverse training environments with varying textures, lighting conditions, and physics parameters",
    "motion planning": "a collision-free trajectory winding through a configuration space filled with obstacle regions",
    "kinematics": "joint angle arcs and end-effector reachable workspace envelope boundaries",
    "dynamics": "force and torque vectors acting on rigid body segments with acceleration motion trails",
    "whole-body control": "coordinated joint torque commands distributed across the entire robot body from a central controller",

    # ── Medical ──
    "tumor": "an irregular mass with spiculated boundary highlighted in a tissue cross-section",
    "lesion": "an abnormal tissue region marked by a colored diagnostic contour overlay",
    "mri": "a grayscale brain or body cross-section slice with anatomical region annotation markers",
    "ct scan": "sequential axial slice views stacked into a volumetric perspective projection",
    "x-ray": "a radiographic projection with bones and soft tissue shown in inverse grayscale contrast",
    "ultrasound": "a sonographic image with characteristic speckle texture and measurement caliper marks",
    "pathology": "a stained tissue slide at cellular magnification with region-of-interest annotations",
    "histology": "tissue microstructure with cell nuclei visible as small stained dots in a regular arrangement",
    "endoscopy": "an interior tubular organ view with a highlighted polyp or lesion anomaly",
    "diagnosis": "a diagnostic decision tree branching from observed symptoms to ranked possible conditions",
    "prognosis": "a time-series disease trajectory projection with confidence interval bands",
    "survival analysis": "a Kaplan-Meier curve showing survival probability declining over follow-up time",

    # ── Biology / Life Sciences ──
    "protein structure": "a folded polypeptide chain as a 3D ribbon diagram with alpha helices and beta sheet arrows",
    "protein folding": "an unfolded chain progressively collapsing through intermediates into a compact native state",
    "dna": "a double helix with complementary base-pair rungs connecting the two sugar-phosphate backbones",
    "rna": "a single-stranded polynucleotide chain folding back on itself into secondary structure stem-loops",
    "gene expression": "a heatmap grid of activation levels across multiple conditions and gene rows",
    "mutation": "a highlighted base-pair substitution site in a DNA sequence alignment view",
    "cell signaling": "ligand-receptor binding at the cell membrane triggering a cascade of intracellular kinase arrows",
    "metabolic pathway": "a network of enzymatic reaction nodes connected by substrate-to-product conversion arrows",
    "microbiome": "diverse bacterial colony shapes clustered in a host-environment interface zone",
    "crispr": "a Cas9 protein with guide RNA directing a cut at a specific genomic target locus",
    "single-cell": "individual cells plotted as points in a 2D reduced-dimension gene-expression embedding space",

    # ── Physics ──
    "quantum": "wavefunction probability density clouds surrounding an atomic nucleus center",
    "entanglement": "two particles connected by a glowing correlated-state correlation line across space",
    "superposition": "a single system existing in multiple basis states shown as overlapping semi-transparent ghost images",
    "superconductivity": "electrical resistance dropping abruptly to zero at a marked critical temperature point",
    "phase transition": "a system transforming from one ordered phase to another at a sharp critical threshold",
    "condensed matter": "atoms in periodic lattice arrangement with electron band structure energy overlay",
    "particle physics": "subatomic particle tracks curving through a detector under magnetic field influence",

    # ── Materials Science ──
    "crystal structure": "atoms arranged in a periodic 3D lattice with unit cell boundary box marked",
    "defect": "an irregularity or atomic vacancy highlighted within an otherwise regular crystal lattice",
    "interface": "an atomic boundary layer where two different materials meet with an interaction transition zone",
    "microstructure": "polycrystalline grain boundaries with different crystallographic orientation shading per grain",
    "nanoparticle": "a tiny faceted particle with surface ligand molecules attached like a molecular corona",

    # ── Chemistry ──
    "catalyst": "a solid surface with highlighted active sites where reactant molecules dock and transform",
    "reaction mechanism": "step-by-step electron-pushing curved arrows showing bond-making and bond-breaking",
    "molecular dynamics": "a molecule with motion trail ghosts showing vibrational and rotational movement modes",
    "spectroscopy": "an absorption or emission spectrum trace with characteristic peak pattern and baseline",

    # ── Energy ──
    "solar cell": "a photovoltaic panel in cross-section showing photon absorption and electron-hole pair generation",
    "battery": "layered anode-electrolyte-cathode structure with lithium-ion migration arrows between electrodes",
    "fuel cell": "hydrogen input and oxygen input combining through a membrane to produce water and electrical output",
    "wind turbine": "three aerodynamic blades with wake flow streamlines and generator nacelle cutaway view",

    # ── Systems / Networks ──
    "distributed system": "multiple computing nodes connected by message-passing arrows with consensus protocol markers",
    "load balancing": "incoming request stream being distributed evenly across a pool of server nodes by a dispatcher",
    "caching": "a fast small memory layer intercepting requests before they reach the slower large storage tier below",
    "database": "a storage cylinder abstraction with index lookup paths and a query execution plan tree",

    # ── Control / RL ──
    "reinforcement learning": "a reward signal flowing through a decision network, value function rendered as a heatmap overlay",
    "policy gradient": "a policy parameter surface with gradient arrows converging uphill toward higher-reward regions",
    "exploration": "branching search trajectories spreading into unknown regions before converging to an optimal path",
    "regret": "a gap gradually closing between optimal and achieved cumulative performance over time steps",

    # ── Math ──
    "optimization": "a loss surface landscape with a gradient descent path spiraling into a deep minimum basin",
    "convergence": "iterates approaching a fixed point with visibly decreasing step sizes between successive positions",
    "manifold": "a curved lower-dimensional surface smoothly embedded within a higher-dimensional ambient space",
    "topology": "shapes being continuously deformed while preserving their essential connectedness properties",
    "graph": "nodes connected by edges with visible community structure and degree distribution patterns",

    # ── Generic research concepts ──
    "dataset": "data samples arranged in a structured grid showing diversity across classes and feature distributions",
    "efficiency": "a gauge showing performance relative to resource consumption with a highlighted sweet spot",
    "trade-off": "a Pareto frontier curve revealing the optimal balance between two competing objective axes",
    "robustness": "a system maintaining stable performance under various perturbation types and increasing magnitudes",
    "generalization": "training data distribution on the left and a distinctly shifted test distribution on the right",
    "uncertainty": "predictions displayed with confidence intervals or error bars showing varying certainty widths",
    "interpretability": "a black-box model being opened with a cutaway view to reveal its internal decision structure",
    "distillation": "a large complex teacher model transferring condensed knowledge to a compact student model",
    "pruning": "a dense network with inactive low-weight connections being removed, leaving a sparse efficient substructure",
    "quantization": "continuous weight values being discretized into a reduced set of discrete precision levels",
    "federated learning": "multiple isolated local data silos each contributing model updates to a central aggregation server",
    "contrastive learning": "positive paired samples pulled together and negative paired samples pushed apart in embedding space",
    "self-supervised": "a model learning rich representations from unlabeled data by predicting deliberately masked portions",
    "data augmentation": "a single sample transformed into multiple variants through rotations, crops, flips, and color jitter",
    "adversarial": "a clean input and its imperceptibly perturbed adversarial version producing dramatically different outputs",
    "out-of-distribution": "in-distribution samples clustered tightly with out-of-distribution samples falling clearly outside",
    "active learning": "a model selectively querying an oracle for labels on the most uncertain or informative unlabeled samples",
    "meta-learning": "a meta-learner extracting common learning patterns aggregated across multiple distinct training episodes",
    "continual learning": "a model sequentially acquiring new task skills while retaining previously learned ones without catastrophic forgetting",
    "causality": "a directed acyclic graph with causal arrow edges showing genuine cause-to-effect relationships",
    "attention mechanism": "query-key-value dot-product computation with softmax-normalized attention weight distribution",
    "normalization": "data being centered and scaled to zero mean and unit variance, shown as a distribution shift operation",
    "regularization": "model complexity constrained by a penalty term visualized as shrinking weight magnitude toward zero",
    "ensemble": "multiple diverse base models voting or averaging their outputs to produce a combined final prediction",
    "knowledge graph": "entities as labeled nodes connected by typed semantic relation edges forming an interlinked web",
    "retrieval-augmented": "a query first fetching relevant documents from an external knowledge store before response generation",
    "rag": "a two-stage system: a retriever fetching documents from a knowledge base, then a generator incorporating them",
    "vector database": "high-dimensional embedding vectors indexed and searchable in a nearest-neighbor lookup space",
    "open-source": "code and model weights symbolically released as an open public resource with community contributions",
    "training-free": "a model achieving results without any additional training or fine-tuning steps",
    "lighting control": "directional light sources casting controlled illumination with shadow manipulation on a scene",
    "policy": "a decision boundary or control surface mapping observations to optimal actions",
    "sequence prediction": "a temporal sequence with past observations on the left and future predictions extending to the right",
    "time series": "a temporal signal waveform with past and future segments separated by a prediction horizon marker",
}


def build_cover_prompt(script: str, summary_hint: str = "") -> str:
    """Build a paper-specific SDXL prompt.

    Architecture: style → title-visual-core → discipline-stage → keyword-props → summary → constraints.
    Title terms are extracted and front-loaded for maximum SDXL weight.
    """
    title = _section_first_line(script, *_TITLE_HEADINGS) or "academic research paper"
    raw_keywords = _section_lines(script, _KEYWORD_HEADINGS, limit=8)
    keywords_clean = [line.lstrip("- ").strip() for line in raw_keywords]
    summary_line = _first_meaningful_line(summary_hint)
    script_line = _extract_script_hint(script)
    keyword_text = ", ".join(keywords_clean)

    # ── Extract visual terms from the TITLE (highest priority) ──
    title_terms = _extract_title_visual_terms(title)
    # Map title terms to visuals
    title_visuals, title_covered = _map_keywords_to_visuals(title_terms)
    # Also extract any raw title terms that didn't map (for direct visual cue)
    raw_title_terms = [t for t in title_terms if t.lower() not in title_covered and t.lower() not in _GENERIC_WORDS]

    # ── Match discipline → base stage scene ──
    base_scene = _infer_topic_visual_scene(title, keyword_text, summary_line, script_line)

    # ── Map keyword terms to visuals ──
    keyword_visuals, kw_covered = _map_keywords_to_visuals(keywords_clean)

    # ── Merge visuals: title visuals first (higher priority), then keyword visuals ──
    all_visuals: list[str] = []
    seen: set[str] = set()
    for v in title_visuals:
        if v not in seen:
            all_visuals.append(v)
            seen.add(v)
    for v in keyword_visuals:
        if v not in seen:
            all_visuals.append(v)
            seen.add(v)

    # ── Compose prompt ──
    parts: list[str] = []

    # Layer 1: Style
    parts.append(_STYLE)

    # Layer 2: Title-derived visual core — HIGHEST weight, front-loaded
    title_parts: list[str] = []
    title_parts.extend(title_visuals[:4])
    # Raw unmapped terms from title also carry unique visual signal
    title_parts.extend(raw_title_terms[:2])
    if title_parts:
        core = "; ".join(title_parts)
        parts.append(f"visual subject: {core}")

    # Layer 3: Discipline stage + all visuals woven together
    if all_visuals:
        props = "; ".join(all_visuals[:6])
        parts.append(f"scene: {base_scene} — specifically showing {props}")
    else:
        parts.append(f"scene: {base_scene}")

    # Layer 4: Unmapped keywords as conceptual anchors
    all_covered = title_covered | kw_covered
    unmapped = [kw for kw in keywords_clean
                if kw.lower() not in all_covered
                and len(kw) >= 3
                and kw.lower() not in _GENERIC_WORDS]
    if unmapped:
        parts.append(f"concepts: {', '.join(unmapped[:4])}")

    # Layer 5: Summary context
    if summary_line:
        parts.append(f"context: {summary_line[:140]}")

    # Layer 6: Constraints
    parts.append(_CONSTRAINTS)

    return ". ".join(parts)


# Words too generic to provide visual signal — filtered from prompts
_GENERIC_WORDS = {
    "paper", "method", "result", "study", "research", "analysis", "approach",
    "model", "system", "data", "learning", "training", "test", "performance",
    "experiment", "evaluation", "implementation", "proposed", "novel", "improved",
    "基于", "方法", "研究", "实验", "结果", "分析", "模型", "系统", "数据",
}

# Words to strip from titles — they carry no visual information
_TITLE_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "out", "off", "up", "down",
    "about", "than", "then", "also", "very", "just", "only", "such", "no", "not",
    "or", "and", "but", "if", "because", "so", "yet", "while", "although",
    "its", "it", "this", "that", "these", "those", "we", "our", "you", "your",
    "towards", "toward", "via", "using", "based", "new", "one",
    "learning", "scaling", "low", "cost", "high", "quality",
    "improved", "efficient", "scalable", "robust", "towards",
    "review", "survey", "comprehensive", "study", "approach", "method",
    "analysis", "evaluation", "comparison", "case", "overview",
}

# Prepositions & connectors that join two separate concepts — split on these
_TITLE_SPLIT_WORDS = {"for", "with", "via", "using", "from", "through", "based", "towards", "toward"}


def _extract_title_visual_terms(title: str) -> list[str]:
    """Extract meaningful visual noun phrases from a paper title.

    Splits on colon, comma, 'and', 'or', and prepositions ('for', 'with', 'via'...),
    then extracts multi-word phrases that carry visual meaning.

    Examples:
        "GROOT-N1: An Open Foundation Model for Generalist Humanoid Robots"
        → ["GROOT-N1", "Open Foundation Model", "Generalist Humanoid Robots"]

        "Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation,
         Locomotion, and Aviation"
        → ["Cross-Embodied Learning", "Manipulation", "Navigation",
            "Locomotion", "Aviation"]

        "Mobile ALOHA: Learning Bimanual Mobile Manipulation with
         Low-Cost Whole-Body Teleoperation"
        → ["Mobile ALOHA", "Bimanual Mobile Manipulation",
            "Whole-Body Teleoperation"]
    """
    # Step 1: Split on major delimiters (colon, semicolon, em-dash)
    segments = re.split(r"[:;]|\s+[–—]\s+", title)

    # Step 2: Within each segment, split on commas, 'and', 'or', and prepositions
    split_pattern = r",\s*|\s+(?:and|or|" + "|".join(re.escape(w) for w in _TITLE_SPLIT_WORDS) + r")\s+"
    all_phrases: list[str] = []
    for segment in segments:
        sub_segments = re.split(split_pattern, segment.strip(), flags=re.IGNORECASE)
        for sub in sub_segments:
            words = sub.strip().split()
            # Remove stop words
            content_words = [w for w in words
                           if w.lower() not in _TITLE_STOP_WORDS
                           and len(w) >= 2]
            if not content_words:
                continue
            phrase = " ".join(content_words).rstrip(".,;:!?()[]{}")
            # Keep phrases with ≥2 content words or a single significant word (acronym, proper noun)
            if len(content_words) >= 2:
                if phrase and len(phrase) >= 3:
                    all_phrases.append(phrase)
            elif len(content_words) == 1:
                word = content_words[0]
                # Single word: keep if it looks significant (acronym, compound, or ≥4 chars)
                if word.isupper() or "-" in word or len(word) >= 4:
                    if word.lower() not in _GENERIC_WORDS:
                        all_phrases.append(word)

    # Deduplicate, preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in all_phrases:
        lower = phrase.lower()
        if lower not in seen:
            unique.append(phrase)
            seen.add(lower)

    return unique


def _map_keywords_to_visuals(keywords: list[str]) -> tuple[list[str], set[str]]:
    """Map paper keywords to concrete visual elements.

    Returns (visual_strings, covered_keyword_lowercase_set).
    """
    results: list[str] = []
    covered: set[str] = set()
    seen_visuals: set[str] = set()

    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in _GENERIC_WORDS or len(kw) < 3:
            covered.add(kw_lower)
            continue

        visual = _lookup_visual(kw_lower)
        if visual:
            if visual not in seen_visuals:
                results.append(f"{kw} as {visual}")
                seen_visuals.add(visual)
            covered.add(kw_lower)
        # If no mapping found: don't pass through as bare keyword —
        # unmapped keywords are handled separately as "concepts:"
        # This prevents "humanoid robot" appearing twice (once bare, once in concepts)

    return results, covered


def _lookup_visual(kw_lower: str) -> str | None:
    """Look up a keyword in the visual mapping dict with fuzzy matching."""
    # Exact match
    if kw_lower in _KEYWORD_VISUALS:
        return _KEYWORD_VISUALS[kw_lower]
    # Singular/plural variants
    if kw_lower.rstrip("s") in _KEYWORD_VISUALS:
        return _KEYWORD_VISUALS[kw_lower.rstrip("s")]
    if kw_lower + "s" in _KEYWORD_VISUALS:
        return _KEYWORD_VISUALS[kw_lower + "s"]
    # Substring: key is substring of kw or vice versa (min 4 chars to avoid false matches)
    for dict_key, dict_visual in _KEYWORD_VISUALS.items():
        if len(dict_key) >= 4:
            if dict_key in kw_lower or kw_lower in dict_key:
                return dict_visual
    return None


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
#  Discipline stage descriptions
#  Each provides the STAGE and visual VOCABULARY. Kept compact —
#  the paper's own keywords supply the specific props on this stage.
# ═══════════════════════════════════════════════════════════════════

def _infer_topic_visual_scene(
    title: str, keywords: str, summary_hint: str = "", script_hint: str = ""
) -> str:
    """Return a compact discipline-stage scene description."""
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
            "Gaussian noise particles on the left dissolving into smooth vector-field streamlines "
            "that curve across the composition and converge into a data-distribution manifold on "
            "the right, with ODE trajectory lines and a coordinate grid tracing the learned mapping"
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
                "motion-capture floor with engineering grid, metal composite shell panels, "
                "servo joints with motion trajectory arcs, force-vector arrows from center of "
                "mass, sensor head unit with camera apertures; mechanical not biological — "
                "no skin, no skeleton, no human anatomy"
            ),
        ),
        # ── Computer Vision ──
        (
            (
                "vision transformer", "计算机视觉", "目标检测", "图像分割", "姿态估计",
                "image recognition", "object detection", "segmentation",
                "pose estimation", "visual grounding", "multimodal",
                "image synthesis", "image generation", "generative vision",
                "lighting control", "image editing",
            ),
            (
                "a single photograph in the center being analytically deconstructed: patch-grid "
                "overlay, bounding box around a focal object, segmentation mask contours in "
                "contrasting color, attention heatmap glowing over salient regions, feature "
                "activation trails radiating to filter-response thumbnails in the margin"
            ),
        ),
        # ── LLM / Language Models ──
        (
            (
                "llm", "大语言模型", "语言模型", "提示词", "对齐",
                "large language model", "language model", "instruction tuning",
                "alignment", "transformer", "reasoning", "prompt", "rlhf",
            ),
            (
                "stacked transformer layers as horizontal attention bands with token tiles "
                "flowing through them, embedding vectors scattered as a constellation in "
                "low-dimensional projection, a prompt block on the left connected by a widening "
                "river of meaning to a response block on the right; no robot face, no chat app "
                "screenshot, no glowing brain, no magic book"
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
                "a radiology scan or pathology slide as the central vertical panel with "
                "diagnostic contour overlays in colored pencil, measurement calipers marking "
                "anatomical distances, tissue texture detail at one edge, clinical chart traces "
                "and lab value markers as marginal annotations; no hospital room, no doctor "
                "portrait, no stethoscope, no pill bottles"
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
                "wave interference as the main visual motif — concentric ripples from two "
                "sources crossing into a standing pattern, particle trajectories as dashed "
                "arrow lines curving through a field, field lines as flowing curves, abstract "
                "equation fragments as unreadable symbolic marks, coordinate axes; no fantasy "
                "space art, no decorative planets, no landscape"
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
                "in repeating unit cells, a cross-section slice at the bottom revealing internal "
                "grain boundaries as irregular polygons, electron-microscope texture inset at "
                "one corner, electrode or catalyst interface as a horizontal reaction boundary; "
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
                "components, DNA double-helix curling through one region, assay well-plate grid "
                "as faint background pattern, gene-expression heatmap blocks in the margin; "
                "no human skeleton, no full-body anatomy, no garden landscape"
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
                "population nodes as circles of varying sizes across the composition connected "
                "by social-network edges, survey response data as clustered bar charts rising "
                "from nodes, a policy intervention shown as a color shift propagating through "
                "the network, timeline bands marking before-and-after periods at the bottom; "
                "no office buildings, no money piles, no portraits, no city skyline"
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
                "flowing rightward toward Earth in the center, Earth's magnetosphere as protective "
                "blue magnetic field lines bending around the planet, aurora oval glowing near the "
                "polar region, satellite orbit markers and magnetometer graph traces in the margin; "
                "no houses, no buildings, no landscape painting"
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
                "temperature isotherm contours tracing across the globe surface, a satellite "
                "sensor swath shown as a trapezoidal footprint, carbon flux arrows between land "
                "and ocean, topographic relief in muted earth tones; no houses, no villages, "
                "no roads, no travel-poster scenery"
            ),
        ),
        # ── Energy Systems ──
        (
            (
                "energy", "能源", "电力", "电网", "renewable", "solar", "wind",
                "photovoltaic", "storage system", "grid",
            ),
            (
                "power grid as an organic network of nodes connected by transmission lines "
                "spanning the composition, solar panel cells as a geometric blue pattern on one "
                "side, wind turbine as elegant blades on the other, energy storage as stacked "
                "block modules below, power-flow arrows following grid paths, load-curve chart "
                "as marginal graph; no houses as main subject, no residential street, no city skyline"
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
                "modular computing blocks in a clear layered architecture with routing lines "
                "connecting them, queue-depth indicators as stack heights, a database cylinder "
                "at the foundation, scheduler lanes as parallel tracks with tasks moving along "
                "them, pipeline arrows showing data flow; each block has labeled-looking but "
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
                "as a continuous cycle with arrowheads, trajectory curves branching from start "
                "to goal regions showing exploration paths, a policy surface landscape with "
                "gradient arrows pointing uphill toward the optimum, controller block with "
                "input-output signal traces; no human skeleton, no houses, no abstract maze"
            ),
        ),
        # ── Chemistry ──
        (
            (
                "chemistry", "化学", "分子反应", "合成", "催化反应",
                "reaction", "synthesis", "molecular interaction", "spectroscopy",
            ),
            (
                "a molecular structure as the central sculptural form — ball-and-stick model "
                "with visible bond angles, two molecules approaching with a dashed transition-state "
                "line between them, reaction pathway as arched arrows from reactants to products, "
                "energy profile curve beneath showing activation barrier and exothermic drop, "
                "spectroscopy trace as marginal graph; no kitchen, no medicine bottles, no "
                "random bubbles, no floral patterns"
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
                "lines, algebraic symbol fragments as abstract calligraphic marks arranged like "
                "an architectural blueprint, matrix grid with highlighted entries, probability "
                "distribution as a smooth bell curve with shaded tail regions; no buildings, "
                "no books as main subject, no chalkboard classroom"
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
                "passages across documents, a balance scale as the central weighing metaphor with "
                "one side tilting slightly, regulation flowchart as tributary streams merging into "
                "a main channel, case timeline as a horizontal band with marked decision points, "
                "an institutional seal-like abstract circular mark; no courthouse building as "
                "main scene, no judge portrait, no prison, no city street"
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
                "root systems penetrating downward, irrigation water flow as blue dashed lines "
                "between rows, small sensor-marker icons at sampling points, yield prediction "
                "bar chart in the margin comparing treatments; no farmhouse, no barn, no rural "
                "sunset landscape, no decorative plants without research context"
            ),
        ),
    ]

    for terms, description in topic_blocks:
        if _contains_any(text, terms):
            return description

    # ── Fallbacks ──
    if "agent" in text or "tool" in text or "智能体" in text:
        return (
            "an AI agent workflow as interconnected tool nodes with planning arrows radiating "
            "from a central reasoning block, connected to document icons, code snippets, and "
            "database symbols, execution traces as step-by-step horizontal lanes"
        )
    return (
        "the core concept rendered as a meaningful abstract composition with domain-relevant "
        "symbolic objects connected by data-flow arrows, clean editorial diagram capturing "
        "the research contribution"
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
