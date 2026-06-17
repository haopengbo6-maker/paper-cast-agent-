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


COVER_PROMPT_VERSION = "v16"

_TITLE_HEADINGS = ("# 播报标题", "# 标题", "# title")
_SCRIPT_HEADINGS = ("# 播报脚本", "# 脚本", "# script")
_KEYWORD_HEADINGS = ("# 关键词", "# 关键字", "# keywords")

# ── Quality trigger + short style (front-loaded, ~20 words) ──
_QUALITY = "masterpiece, best quality"
_STYLE = (
    "scientific plate illustration, colored pencil and gouache on cream paper, "
    "printmaking aesthetic, editorial composition, muted scholarly palette, "
    "soft paper grain, generous negative space"
)

# ── Short constraints ──
_CONSTRAINTS = (
    "clean centered composition, no text, no letters, no typography, no Chinese, "
    "no photorealism, no 3D render, no generic decoration, no ancient art"
)


# ═══════════════════════════════════════════════════════════════════
#  Compact keyword → visual cue mapping
#  Each entry: short visual phrase (5–10 words) that SDXL can latch onto.
# ═══════════════════════════════════════════════════════════════════

_KEYWORD_VISUALS: dict[str, str] = {
    # ── Robot embodiments ──
    "humanoid robot": "mechanical humanoid with metal shell panels, servo joints, sensor head, no skin",
    "humanoid": "mechanical humanoid with metal shell panels, servo joints, sensor head",
    "bipedal robot": "two-legged robot mid-stride with joint trajectory arcs",
    "quadruped robot": "four-legged robot with articulated legs and terrain sensors",
    "robot arm": "multi-axis industrial robot arm with joint angle markers",
    "mobile robot": "wheeled robot platform with sensor array and path overlay",
    "drone": "multi-rotor aerial vehicle with flight path and camera payload",
    "soft robot": "compliant flexible-bodied robot with pneumatic actuation",
    "bimanual": "two robotic arms coordinating to manipulate a shared object",
    "aloha": "two dexterous tabletop robot arms performing fine bimanual tasks",
    "cross-embodied": "multiple robot forms sharing a common policy across embodiments",
    "aviation": "aerial vehicle in flight with trajectory and onboard sensors",
    "quadrotor": "four-rotor aerial vehicle with flight dynamics overlay",

    # ── AI / ML core ──
    "foundation model": "a unifying neural backbone branching into diverse downstream modules",
    "generalist": "multiple task domains linked as visual panels sharing a common backbone",
    "multi-task": "parallel task heads branching from a shared neural trunk",
    "transfer learning": "knowledge flowing as a bridge from source domain to target domain",
    "few-shot": "correct predictions from only a handful of labeled support examples",
    "zero-shot": "recognizing unseen categories from semantic descriptions alone",
    "fine-tuning": "a neural structure refined by an incoming stream of new data",
    "pre-training": "vast varied data absorbed into a growing neural representation",
    "scaling law": "log-log plot of performance rising with compute and model size",
    "emergent ability": "a sudden capability spike beyond a critical scale threshold",
    "in-context learning": "demonstration exemplars reshaping behavior without weight updates",
    "chain-of-thought": "step-by-step reasoning nodes linked by deduction arrows",
    "instruction tuning": "natural language commands mapped to structured task outputs",
    "alignment": "model outputs steered toward human-preferred value regions",
    "rlhf": "reward signal loop from human preferences back into policy optimization",
    "benchmark": "comparative bar chart with one method prominently highlighted",
    "ablation": "component removal study as descending importance-ranked bars",
    "latent space": "low-dimensional manifold with distinct colored point-cloud clusters",
    "embedding": "high-dimensional vectors projected as scattered constellation points",
    "attention": "weighted connection bands between query and key positions",
    "self-attention": "a sequence attending to itself with weighted connection density",
    "cross-attention": "two sequences aligned by bridging attention weight bands",
    "transformer": "stacked layers with multi-head attention and feed-forward blocks",
    "token": "discrete text units as small colored squares flowing through layers",
    "tokenizer": "raw text fragmenting into colored token blocks at a segmentation boundary",
    "mixture of experts": "a router dispersing tokens across parallel expert sub-networks",
    "moe": "a router dispersing tokens across parallel expert sub-networks",

    # ── Generative models ──
    "diffusion model": "noise resolving into a clean image across horizontal denoising stages",
    "diffusion": "noise resolving into a clean image across progressive denoising steps",
    "denoising": "a noisy image becoming progressively cleaner across sequential stages",
    "score-based": "gradient fields pointing from noisy toward high-density regions",
    "gan": "generator and discriminator as two opposing forces in equilibrium",
    "generative adversarial": "generator and discriminator locked in a minimax game",
    "vae": "encoder compressing to latent bottleneck, decoder reconstructing output",
    "variational autoencoder": "encoder mapping to gaussian latent, decoder sampling and reconstructing",
    "autoregressive": "tokens predicted left-to-right in sequential order with probability bars",
    "discrete token": "continuous data vector-quantized into a codebook of discrete visual tokens",

    # ── Computer Vision ──
    "object detection": "bounding boxes with confidence scores localizing scene objects",
    "segmentation": "pixel-precise contours partitioning an image into labeled regions",
    "instance segmentation": "each object instance outlined in a distinct color mask",
    "semantic segmentation": "every pixel assigned a category shown as color-coded map",
    "panoptic segmentation": "semantic regions and instances unified in a single overlay",
    "pose estimation": "skeleton keypoints linked by bones overlaid on a moving figure",
    "depth estimation": "scene with warm-to-cool distance-based shading from near to far",
    "optical flow": "dense motion vector arrows showing pixel displacement between frames",
    "3d reconstruction": "multiple 2D views projecting into a unified volumetric 3D form",
    "nerf": "rays passing through a volumetric radiance field with density samples",
    "gaussian splatting": "ellipsoid 3D gaussians scattered in space rendering a scene",
    "visual grounding": "language phrases linked by arrows to highlighted image regions",
    "image captioning": "input image connected by arrows to descriptive output tokens",
    "super-resolution": "low-res patch enlarged side-by-side into high-res detailed version",
    "image inpainting": "a masked region filled with context-consistent content",
    "image generation": "a new image materializing from a text prompt or noise seed",

    # ── Robotics skills ──
    "manipulation": "robotic hand precisely grasping an object with force markers",
    "locomotion": "bipedal walking with footstep pressure maps on the ground plane",
    "navigation": "path-planning with optimal route and obstacle avoidance fields",
    "grasping": "fingertip contact points on object surface with grasp quality indicators",
    "dexterous": "multi-finger robotic hand with independent joint articulations",
    "teleoperation": "human operator motion mirrored by a robot executing the same action",
    "sim-to-real": "simulation transferring policies across domain gap to real robot",
    "domain randomization": "diverse training environments with varied textures and physics",
    "motion planning": "collision-free path winding through obstacle-filled space",
    "kinematics": "joint angle arcs and end-effector reachable workspace envelope",
    "dynamics": "force and torque vectors on rigid bodies with acceleration trails",
    "whole-body control": "coordinated joint torques distributed across entire robot body",

    # ── Medical ──
    "tumor": "irregular mass with spiculated boundary in tissue cross-section",
    "lesion": "abnormal tissue region marked by colored diagnostic contour",
    "mri": "grayscale brain or body slice with anatomical annotation markers",
    "ct scan": "sequential axial slices stacked into volumetric perspective",
    "x-ray": "radiographic projection with inverse grayscale bone-tissue contrast",
    "ultrasound": "sonographic image with speckle texture and measurement calipers",
    "pathology": "stained tissue slide at cellular magnification with ROI annotations",
    "histology": "tissue microstructure with cell nuclei as small stained dots",
    "endoscopy": "interior tubular organ view with highlighted anomaly",
    "diagnosis": "diagnostic decision tree branching from symptoms to conditions",
    "prognosis": "disease trajectory projection with confidence interval bands",
    "survival analysis": "Kaplan-Meier curve of survival probability over time",

    # ── Biology / Life Sciences ──
    "protein structure": "folded polypeptide 3D ribbon with alpha helices and beta sheets",
    "protein folding": "unfolded chain collapsing through intermediates into compact state",
    "dna": "double helix with base-pair rungs connecting sugar-phosphate backbones",
    "rna": "single-stranded polynucleotide folding into secondary structure loops",
    "gene expression": "heatmap grid of activation levels across conditions and genes",
    "mutation": "highlighted base-pair substitution in DNA sequence alignment",
    "cell signaling": "ligand-receptor binding triggering intracellular kinase cascade",
    "metabolic pathway": "network of enzymatic reaction nodes with conversion arrows",
    "microbiome": "diverse bacterial colony shapes in host-environment interface",
    "crispr": "Cas9 protein with guide RNA targeting a specific genomic locus",
    "single-cell": "individual cells as points in 2D gene-expression embedding space",

    # ── Physics ──
    "quantum": "wavefunction probability clouds around an atomic nucleus",
    "entanglement": "two particles connected by glowing correlated-state line",
    "superposition": "system in multiple basis states as overlapping ghost images",
    "superconductivity": "resistance dropping to zero at a critical temperature point",
    "phase transition": "system transforming between ordered phases at a sharp threshold",
    "condensed matter": "atoms in periodic lattice with electron band structure overlay",
    "particle physics": "subatomic particle tracks curving through a detector field",

    # ── Materials Science ──
    "crystal structure": "atoms in periodic 3D lattice with unit cell boundary",
    "defect": "irregularity or vacancy highlighted in regular crystal lattice",
    "interface": "atomic boundary where two materials meet with transition zone",
    "microstructure": "polycrystalline grains with orientation-shaded boundaries",
    "nanoparticle": "tiny faceted particle with surface ligand corona",

    # ── Chemistry ──
    "catalyst": "solid surface with active sites where reactants dock and transform",
    "reaction mechanism": "electron-pushing arrows showing bond-making and bond-breaking",
    "molecular dynamics": "molecule with motion trails showing vibration and rotation",
    "spectroscopy": "absorption spectrum trace with characteristic peak pattern",

    # ── Energy ──
    "solar cell": "photovoltaic panel cross-section with photon-to-electron conversion",
    "battery": "layered anode-electrolyte-cathode with ion migration arrows",
    "fuel cell": "hydrogen and oxygen combining through membrane producing water and power",
    "wind turbine": "three aerodynamic blades with wake streamlines and nacelle cutaway",

    # ── Systems / Networks ──
    "distributed system": "computing nodes with message-passing arrows and consensus markers",
    "load balancing": "request stream distributed across server pool by a dispatcher",
    "caching": "fast memory layer intercepting requests before reaching slow storage",
    "database": "storage cylinder with index lookup paths and query execution plan",

    # ── Control / RL ──
    "reinforcement learning": "reward signal flowing through decision network as heatmap overlay",
    "policy gradient": "policy surface with gradient arrows converging toward optimum",
    "exploration": "branching search trajectories spreading into unknown regions",
    "regret": "gap closing between optimal and achieved performance over time",

    # ── Math ──
    "optimization": "loss surface with gradient descent path spiraling into a minimum",
    "convergence": "iterates approaching a fixed point with decreasing step sizes",
    "manifold": "curved lower-dimensional surface in higher-dimensional ambient space",
    "topology": "shapes continuously deformed preserving connectedness properties",
    "graph": "nodes and edges with community structure and degree patterns",

    # ── Generic research concepts ──
    "dataset": "data samples in structured grid showing class diversity",
    "efficiency": "gauge showing performance relative to resource consumption",
    "trade-off": "Pareto frontier curve between two competing objective axes",
    "robustness": "system maintaining performance under increasing perturbation",
    "generalization": "training distribution left, distinctly shifted test distribution right",
    "uncertainty": "predictions with confidence intervals of varying width",
    "interpretability": "black-box opened with cutaway revealing internal decision logic",
    "distillation": "large teacher transferring condensed knowledge to compact student",
    "pruning": "dense network with inactive connections removed leaving sparse structure",
    "quantization": "continuous weights discretized into reduced precision levels",
    "federated learning": "isolated local silos contributing updates to central aggregator",
    "contrastive learning": "positive pairs pulled together, negative pairs pushed apart",
    "self-supervised": "model learning from unlabeled data by predicting masked portions",
    "data augmentation": "single sample transformed into multiple variants",
    "adversarial": "clean input and imperceptibly perturbed version producing different outputs",
    "out-of-distribution": "in-distribution samples clustered, OOD samples falling outside",
    "active learning": "model querying oracle on most uncertain unlabeled samples",
    "meta-learning": "meta-learner extracting patterns across multiple learning episodes",
    "continual learning": "sequentially acquiring new skills while retaining old ones",
    "causality": "directed acyclic graph with causal arrows showing cause-effect",
    "attention mechanism": "query-key-value dot-product with softmax weight distribution",
    "normalization": "data centered and scaled to zero mean unit variance distribution",
    "regularization": "model complexity constrained by weight-shrinking penalty term",
    "ensemble": "multiple diverse models voting to produce combined prediction",
    "knowledge graph": "entities as nodes connected by typed relation edges",
    "retrieval-augmented": "query fetching documents from external store before generation",
    "rag": "retriever fetching documents, generator incorporating them into output",
    "vector database": "embedding vectors indexed in nearest-neighbor lookup space",
    "open-source": "code and model weights released as open public resource",
    "training-free": "model achieving results without any additional training or fine-tuning",
    "lighting control": "directional lights casting controlled illumination with shadow manipulation",
    "policy": "decision boundary mapping observations to optimal actions",
    "sequence prediction": "past observations left, future predictions extending right",
    "time series": "temporal waveform with past and future separated by prediction marker",
    "deep learning": "multi-layered neural network with data flowing through stacked processing levels",
    "machine learning": "data flowing into a model producing predictions with error feedback loop",
    "neural network": "interconnected nodes in layers with weighted connection lines between them",
    "medical imaging": "grayscale medical scan with highlighted region and diagnostic markers",
    "natural language processing": "text stream flowing through linguistic analysis layers into structured output",
    "speech recognition": "audio waveform transforming into transcribed text tokens",
    "geometry": "geometric shapes with construction lines, angles, and measurement arcs",
    "probability": "distribution curve with shaded regions marking significant probability areas",
    "statistics": "data points scattered around a trend line with confidence interval bands",
    "computer vision": "image with bounding boxes, segmentation masks, and feature activation overlays",
    "review": "comprehensive survey diagram connecting multiple research threads into a taxonomy tree",
    "survey": "taxonomy tree organizing research methods into hierarchical categories",
    "medical image": "grayscale medical scan with highlighted region-of-interest and measurement markers",
    "medical imaging": "grayscale medical scan with highlighted region-of-interest and measurement markers",
    "generative model": "data distribution being learned by a model that generates new samples",
    "wave function": "quantum wave pattern with interference fringes and probability amplitude peaks",
    "education": "learning progression shown as knowledge building blocks with connecting pathways",
    "neuroscience": "neuron network with synaptic connections and signal propagation pathways",
    "ecology": "ecosystem diagram with species interactions and environmental factor flows",
    "dna": "double helix with base-pair rungs connecting two spiraling backbones",
    "rna": "single-stranded polynucleotide folding into stem-loop secondary structures",
    "microbiome": "diverse bacterial colony shapes in a host-environment interface zone",
    "wind turbine": "three aerodynamic blades with wake flow and generator nacelle cutaway",
    "communication": "signal waves propagating between transmitter and receiver nodes",
    "reaction mechanism": "electron-pushing curved arrows showing step-by-step bond transformations",
    "matrix": "grid of numbers with selected entries highlighted and eigenvalue markers",
    "graph theory": "nodes connected by edges with community clusters and degree annotations",
    "regulation": "flowchart of rules branching from a central regulatory framework",
    "irrigation": "water flow lines spreading across crop rows from a central distribution point",
    "yield": "bar chart comparing agricultural output across different treatment conditions",
    "geomagnetic storm": "solar wind particles streaming toward Earth, magnetosphere as protective magnetic field shield, aurora oval glowing near poles",
    "space weather": "Sun emitting solar wind and coronal mass ejection toward Earth's magnetosphere",
    "solar wind": "stream of charged particles flowing from the Sun through interplanetary space",
    "magnetosphere": "Earth's protective magnetic field lines deflecting solar wind particles",
    "ionosphere": "upper atmospheric layer with charged particles and radio wave reflection",
    "aurora": "glowing green and purple light curtains in the polar night sky",
    "climate": "Earth globe with atmospheric layers, temperature contours, and carbon flux arrows",
    "remote sensing": "satellite with downward-looking sensor capturing Earth surface data swaths",
    "earth": "blue marble sphere with cloud patterns blue oceans and green-brown land masses",
    "ocean": "ocean surface with current flow arrows temperature gradients and wave patterns",
    "energy": "power grid network with generation sources transmission lines and load distribution",
    "power grid": "interconnected network of transmission lines linking power plants to consumers",
    "control": "feedback loop diagram with controller actuator sensor and reference signal flow",
    "scheduling": "timeline with tasks allocated to parallel processing slots showing execution order",
    "behavior": "human figures with decision flow arrows showing choice patterns and influences",
}


# Words too generic for visual signal
_GENERIC_WORDS = {
    "paper", "method", "result", "study", "research", "analysis", "approach",
    "model", "system", "data", "learning", "training", "test", "performance",
    "experiment", "evaluation", "implementation", "proposed", "novel", "improved",
    "基于", "方法", "研究", "实验", "结果", "分析", "模型", "系统", "数据",
}

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

_TITLE_SPLIT_WORDS = {"for", "with", "via", "using", "from", "through", "based", "towards", "toward"}


def build_cover_prompt(script: str, summary_hint: str = "", llm_client=None) -> str:
    """Build an SDXL cover prompt for a paper.

    If llm_client is provided, uses the LLM to generate a natural, artistic prompt.
    Otherwise falls back to programmatic keyword-based composition.
    """
    title = _section_first_line(script, *_TITLE_HEADINGS) or "academic research paper"
    raw_keywords = _section_lines(script, _KEYWORD_HEADINGS, limit=6)
    keywords_clean = [line.lstrip("- ").strip() for line in raw_keywords]
    summary_line = _first_meaningful_line(summary_hint)
    script_line = _extract_script_hint(script)

    # ── Try LLM-driven prompt generation first ──
    if llm_client:
        try:
            from .prompts import load_prompt
            from .utils import PROMPT_DIR
            cover_template = load_prompt(PROMPT_DIR / "cover_prompt.txt", required_placeholder="{title}")
            llm_prompt = (
                cover_template
                .replace("{title}", title)
                .replace("{keywords}", ", ".join(keywords_clean) or "academic research")
                .replace("{summary}", summary_line or script_line or "academic paper")
            )
            # Use a lower max_tokens since we only need ~120 words
            raw = _retry_llm_call(llm_client, llm_prompt, max_tokens=300)
            # Clean up: remove any prefixes, quotes, or markdown formatting
            cleaned = raw.strip().strip('"').strip("'")
            # Remove common LLM conversational prefixes
            for prefix in ("Here is", "Sure", "Prompt:", "Cover prompt:", "SDXL prompt:"):
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip().strip(":")
            cleaned = cleaned.strip()
            if len(cleaned.split()) >= 15:  # sanity check: must be a real prompt
                return cleaned
        except Exception:
            pass  # Fall through to programmatic generation

    # ── Programmatic fallback ──
    return _build_cover_prompt_programmatic(
        title, keywords_clean, summary_line, script_line
    )


def _retry_llm_call(llm_client, prompt: str, max_tokens: int, attempts: int = 2) -> str:
    """Call LLM with retry, returning text or raising on failure."""
    from .llm_client import chat_with_optional_max_tokens, retry_call
    return retry_call(
        lambda: chat_with_optional_max_tokens(llm_client, prompt, max_tokens=max_tokens),
        max_attempts=attempts,
    )


# ═══════════════════════════════════════════════════════════════════
#  Dynamic colour palette, composition, and lighting
# ═══════════════════════════════════════════════════════════════════

# Keyword → colour mood mapping
_KEYWORD_COLOR_MAP: dict[str, str] = {
    # Warm / earth
    "robot": "warm amber and deep umber accents",
    "manipulation": "warm amber and deep umber accents",
    "locomotion": "warm amber and deep umber accents",
    "agriculture": "olive green and ochre with clay undertones",
    "crop": "olive green and ochre with clay undertones",
    "soil": "olive green and brown ochre with terracotta accents",
    "climate": "atmospheric teal and slate blue with muted coral",
    "environmental": "atmospheric teal and slate blue with muted coral",
    "forest": "deep forest green with amber highlights",
    # Cool / deep
    "space": "deep indigo and cold silver with faint nebula cyan",
    "astronomy": "deep indigo and cold silver with faint nebula cyan",
    "physics": "deep navy and crisp white with soft gold accents",
    "quantum": "deep navy and crisp white with soft gold accents",
    "optics": "deep navy and crisp white with soft gold accents",
    "ocean": "dark teal and seafoam with warm sand highlights",
    "marine": "dark teal and seafoam with warm sand highlights",
    "water": "dark teal and seafoam with warm sand highlights",
    # Biomedical
    "medical": "clinical ivory and slate grey with muted red accents",
    "biology": "soft sage green and warm ivory with sepia detail",
    "protein": "soft sage green and warm ivory with sepia detail",
    "dna": "cool blue and ivory white with subtle amber markers",
    "cell": "warm rose and ivory with olive accents",
    "tumor": "clinical ivory and slate grey with muted red accents",
    "neuroscience": "warm grey and soft gold with faint violet",
    "brain": "warm grey and soft gold with faint violet",
    # Tech / AI
    "ai": "deep slate and electric blue with copper highlights",
    "machine learning": "deep slate and electric blue with copper highlights",
    "deep learning": "deep slate and electric blue with copper highlights",
    "neural": "deep slate and electric blue with copper highlights",
    "gpu": "dark charcoal and emerald green with copper accents",
    "hardware": "dark charcoal and emerald green with copper accents",
    "data": "cool grey and teal with amber annotation marks",
    "security": "obsidian black and crimson red with silver accents",
    "network": "dark indigo and cyan with warm node highlights",
    # Default palette (matches existing style)
    "default": "burnt sienna, ochre, indigo ink, and olive green accents",
}

_COMPOSITIONS = [
    "strong central subject with generous negative space on all sides, symmetrical balance",
    "subject weighted to the left third, open space flowing to the right, dynamic imbalance",
    "diagonal arrangement from lower-left to upper-right, creates forward movement",
    "subjects arranged in a triangle, stable and grounded",
    "subject at upper-center with detailed base tapering downward, top-heavy elegance",
]

_LIGHTING_MODES = [
    "soft diffused lighting from upper left, gentle shadows, museum-quality illumination",
    "warm side light raking across the subject, long elegant shadows, dramatic depth",
    "even ambient light, no harsh shadows, like an overcast studio, refined flatness",
    "low directional light from above, chiaroscuro depth, scholarly atmosphere",
]


def _pick_color_mood(keywords: list[str], title: str) -> str:
    """Infer colour palette from keywords and title text."""
    combined = " ".join(keywords).lower() + " " + title.lower()
    for key, mood in _KEYWORD_COLOR_MAP.items():
        if key in combined:
            return mood
    return _KEYWORD_COLOR_MAP["default"]


def _pick_composition(seed: int) -> str:
    """Rotate through compositions based on a seed value."""
    return _COMPOSITIONS[seed % len(_COMPOSITIONS)]


def _pick_lighting(seed: int) -> str:
    """Rotate through lighting modes based on a seed value."""
    return _LIGHTING_MODES[seed % len(_LIGHTING_MODES)]


def _build_cover_prompt_programmatic(
    title: str, keywords_clean: list[str], summary_line: str, script_line: str
) -> str:
    """Programmatic keyword-based prompt builder (fallback when LLM unavailable)."""
    # ── Collect visual cues ──
    visual_cues: list[str] = []

    title_terms = _extract_title_visual_terms(title)
    for term in title_terms[:3]:
        cue = _lookup_visual(term.lower())
        if cue:
            visual_cues.append(cue)
        elif term.lower() not in _GENERIC_WORDS and len(term) >= 3:
            visual_cues.append(term)

    for kw in keywords_clean:
        kw_lower = kw.lower()
        if kw_lower in _GENERIC_WORDS:
            continue
        cue = _lookup_visual(kw_lower)
        if cue and cue not in visual_cues:
            visual_cues.append(cue)
        elif not cue and len(kw) >= 3 and kw_lower not in _GENERIC_WORDS:
            visual_cues.append(kw)

    seen: set[str] = set()
    unique_cues: list[str] = []
    for cue in visual_cues:
        key = cue.lower()
        if key not in seen:
            unique_cues.append(cue)
            seen.add(key)

    keyword_text = ", ".join(keywords_clean)
    stage = _infer_topic_stage(title, keyword_text, summary_line, script_line)

    if unique_cues:
        scene = ", ".join(unique_cues[:5])
    else:
        scene = f"academic research illustration about {title[:80]}"

    colour_mood = _pick_color_mood(keywords_clean, title)
    comp = _pick_composition(len(unique_cues))
    light = _pick_lighting(len(unique_cues))

    parts: list[str] = [
        _QUALITY,
        _STYLE,
        f"colour mood: {colour_mood}",
        f"composition: {comp}",
        f"lighting: {light}",
        f"subject: {scene}",
        f"context: {stage}",
    ]
    if summary_line:
        parts.append(f"details: {summary_line[:120]}")
    parts.append(_CONSTRAINTS)

    return ". ".join(parts)


def _extract_title_visual_terms(title: str) -> list[str]:
    """Extract meaningful visual noun phrases from a paper title.

    Handles both English and Chinese punctuation as delimiters.
    """
    # Chinese + ASCII delimiters for splitting segments
    # ： is U+FF1A (Chinese colon), ； is U+FF1B (Chinese semicolon)
    segments = re.split(r"[:;：；]|\s+[–—]\s+", title)

    # Split within segments: ASCII comma, Chinese full-width comma (U+FF0C),
    # Chinese enumeration comma (U+3001), and preposition split words
    _CN_DELIM = "，、！？。"  # ，、！？。
    split_pattern = (
        r"[,，" + _CN_DELIM + r"]\s*"
        r"|\s+(?:and|or|" + "|".join(re.escape(w) for w in _TITLE_SPLIT_WORDS) + r")\s+"
    )
    # Chinese/ASCII punctuation to strip from phrase endings
    _TRAILING_PUNCT = ".,;:!?()[]{}，。！？；：、"

    all_phrases: list[str] = []
    for segment in segments:
        sub_segments = re.split(split_pattern, segment.strip(), flags=re.IGNORECASE)
        for sub in sub_segments:
            words = sub.strip().split()
            content_words = [w for w in words if w.lower() not in _TITLE_STOP_WORDS and len(w) >= 2]
            if not content_words:
                continue
            phrase = " ".join(content_words).rstrip(_TRAILING_PUNCT)
            if len(content_words) >= 2 and len(phrase) >= 3:
                all_phrases.append(phrase)
            elif len(content_words) == 1:
                word = content_words[0]
                # Keep acronyms (ALL_CAPS), hyphenated terms, and words ≥4 chars
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


# Chinese → English term bridge for visual lookup
_CN_EN_MAP: dict[str, str] = {
    # Robotics
    "人形机器人": "humanoid robot",
    "双足行走": "bipedal robot",
    "具身智能": "embodied intelligence",
    "机械臂": "robot arm",
    "操作": "manipulation",
    "抓取": "grasping",
    # AI/ML
    "计算机视觉": "computer vision",
    "目标检测": "object detection",
    "图像分割": "segmentation",
    "图像生成": "image generation",
    "姿态估计": "pose estimation",
    "大语言模型": "large language model",
    "语言模型": "language model",
    "提示词": "prompt",
    "对齐": "alignment",
    "深度学习": "deep learning",
    "机器学习": "machine learning",
    "神经网络": "neural network",
    "自然语言处理": "natural language processing",
    "语音识别": "speech recognition",
    "迁移学习": "transfer learning",
    "联邦学习": "federated learning",
    "对比学习": "contrastive learning",
    "自监督": "self-supervised",
    "扩散模型": "diffusion model",
    "生成模型": "generative model",
    "注意力机制": "attention mechanism",
    "大模型": "foundation model",
    # Medical
    "医学图像": "medical imaging",
    "医学影像": "medical imaging",
    "病理": "pathology",
    "临床": "clinical",
    "诊断": "diagnosis",
    "放射": "radiology",
    "手术": "surgery",
    "肿瘤": "tumor",
    "病灶": "lesion",
    "预后": "prognosis",
    # Physics
    "量子": "quantum",
    "光学": "optics",
    "热力学": "thermodynamics",
    "统计物理": "statistical physics",
    "波函数": "wave function",
    "纠缠": "entanglement",
    "超导": "superconductivity",
    "相变": "phase transition",
    "粒子": "particle physics",
    # Materials
    "材料": "materials",
    "晶体": "crystal structure",
    "半导体": "semiconductor",
    "电池": "battery",
    "催化": "catalyst",
    "纳米": "nanoparticle",
    "界面": "interface",
    # Biology
    "生物": "biology",
    "基因": "gene",
    "蛋白": "protein",
    "蛋白质": "protein structure",
    "细胞": "cell",
    "分子": "molecule",
    "神经科学": "neuroscience",
    "DNA": "dna",
    "RNA": "rna",
    "微生物": "microbiome",
    "突变": "mutation",
    # Social Science
    "经济": "economics",
    "金融": "finance",
    "社会": "social",
    "政策": "policy",
    "行为": "behavior",
    "调查": "survey",
    "教育": "education",
    # Space / Earth
    "地磁暴": "geomagnetic storm",
    "空间天气": "space weather",
    "太阳风": "solar wind",
    "磁层": "magnetosphere",
    "电离层": "ionosphere",
    "极光": "aurora",
    "气候": "climate",
    "遥感": "remote sensing",
    "地球": "earth",
    "海洋": "ocean",
    "生态": "ecology",
    # Energy
    "能源": "energy",
    "电力": "power grid",
    "电网": "power grid",
    "光伏": "solar cell",
    "风电": "wind turbine",
    # Systems
    "系统": "systems",
    "网络": "network",
    "分布式": "distributed system",
    "数据库": "database",
    "通信": "communication",
    "调度": "scheduling",
    # Control / RL
    "控制": "control",
    "强化学习": "reinforcement learning",
    "规划": "planning",
    "轨迹": "trajectory",
    "优化": "optimization",
    # Chemistry
    "化学": "chemistry",
    "合成": "synthesis",
    "分子反应": "reaction mechanism",
    "光谱": "spectroscopy",
    # Math
    "数学": "math",
    "定理": "theorem",
    "证明": "proof",
    "几何": "geometry",
    "拓扑": "topology",
    "概率": "probability",
    "统计": "statistics",
    "矩阵": "matrix",
    "图论": "graph theory",
    # Law
    "法律": "law",
    "司法": "legal",
    "合规": "regulation",
    # Agriculture
    "农业": "agriculture",
    "作物": "crop",
    "土壤": "soil",
    "灌溉": "irrigation",
    "产量": "yield",
}


def _lookup_visual(kw_lower: str) -> str | None:
    """Look up a keyword in the visual mapping dict.

    Tries: exact match → Chinese→English translation → plural variants → substring.
    """
    # 1. Exact match in visual dict
    if kw_lower in _KEYWORD_VISUALS:
        return _KEYWORD_VISUALS[kw_lower]

    # 2. Chinese → English translation, then lookup
    en_term = _CN_EN_MAP.get(kw_lower)
    if en_term:
        if en_term in _KEYWORD_VISUALS:
            return _KEYWORD_VISUALS[en_term]
        # Also try fuzzy on the translated term
        visual = _fuzzy_lookup(en_term)
        if visual:
            return visual

    # 3. Plural variants
    stripped = kw_lower.rstrip("s")
    if stripped != kw_lower and stripped in _KEYWORD_VISUALS:
        return _KEYWORD_VISUALS[stripped]
    if kw_lower + "s" in _KEYWORD_VISUALS:
        return _KEYWORD_VISUALS[kw_lower + "s"]

    # 4. Substring matching
    return _fuzzy_lookup(kw_lower)


def _fuzzy_lookup(kw_lower: str) -> str | None:
    """Substring-based lookup in keyword visual dict."""
    for dict_key, dict_visual in _KEYWORD_VISUALS.items():
        if len(dict_key) >= 4 and (dict_key in kw_lower or kw_lower in dict_key):
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
    llm_client=None,
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
    cover_prompt = build_cover_prompt(script_text, summary_hint, llm_client=llm_client)
    payload = {"prompt": cover_prompt, "paper_id": paper_id}
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
            "inputs": {"width": 1216, "height": 832, "batch_size": 1},
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
                "steps": 30,
                "cfg": 5.0,
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
    "low quality, blurry, jpeg artifacts, watermark, signature, artist name, "
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
    "cluttered composition, busy background, too many objects, "
    "speech bubble, comic panel, multiple panels, split screen, grid layout, "
    "frame border, ornamental border, logo, badge, emblem, icon set, "
    "lens flare, bloom glow, overexposed, high contrast, HDR"
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
#  Discipline stage descriptions (v16: compact, 1 sentence each)
#  Provides visual domain context, not a detailed scene.
# ═══════════════════════════════════════════════════════════════════

def _infer_topic_stage(
    title: str, keywords: str, summary_hint: str = "", script_hint: str = ""
) -> str:
    """Return a brief (1-sentence) visual domain cue for the paper's discipline."""
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
        return "generative modeling with vector fields mapping noise to data distributions"

    topic_stages: list[tuple[tuple[str, ...], str]] = [
        # ── Humanoid Robots ──
        (
            (
                "humanoid", "人形机器人", "双足行走", "全身控制", "具身智能",
                "human robot", "human-like robot", "bipedal robot", "bipedal humanoid",
                "android", "embodied intelligence", "embodied ai",
                "robot locomotion", "robot manipulation", "robot grasping",
                "whole-body control", "kinematics", "humanoid motion",
            ),
            "robotics laboratory setting with motion-capture floor and engineering grid overlay",
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
            "computer vision analysis with image patches and annotation overlays",
        ),
        # ── LLM ──
        (
            (
                "llm", "大语言模型", "语言模型", "提示词", "对齐",
                "large language model", "language model", "instruction tuning",
                "alignment", "transformer", "reasoning", "prompt", "rlhf",
            ),
            "language model architecture with token flow and attention patterns",
        ),
        # ── Medical ──
        (
            (
                "medical", "医学", "临床", "病理", "放射", "手术", "诊断",
                "clinical", "radiology", "pathology", "medical image",
                "healthcare", "disease", "diagnosis", "surgery",
                "biomedical", "bioimaging", "bioinformatics", "genomics", "proteomics",
            ),
            "medical imaging with diagnostic annotations and clinical measurement markers",
        ),
        # ── Physics ──
        (
            (
                "physics", "量子", "光学", "热力学", "统计物理", "波函数",
                "quantum", "mechanics", "optics", "wave", "field theory",
                "particles", "thermodynamics", "statistical physics",
                "astronomy", "astrophysics",
            ),
            "physics visualization with wave patterns, field lines, and particle trajectories",
        ),
        # ── Materials ──
        (
            (
                "materials", "材料", "晶体", "聚合物", "半导体", "电池", "催化",
                "material", "crystal", "polymer", "nanomaterial", "semiconductor",
                "battery", "electrode", "catalyst", "surface", "microstructure",
            ),
            "materials science with crystal lattices, cross-sections, and microscope textures",
        ),
        # ── Biology ──
        (
            (
                "biology", "生物", "基因", "蛋白", "细胞", "分子", "神经科学",
                "genomics", "protein", "cell", "molecule", "molecular",
                "neuroscience", "gene", "pathway", "biochemical", "bioinformatics",
            ),
            "life sciences with cellular structures, molecular pathways, and assay grids",
        ),
        # ── Social Science ──
        (
            (
                "economics", "经济", "金融", "社会", "政策", "行为", "调查",
                "finance", "market", "social", "sociology", "policy",
                "behavior", "survey", "education", "humanities", "politics", "psychology",
            ),
            "social science with population networks, survey charts, and policy flow diagrams",
        ),
        # ── Geomagnetic Storm ──
        (
            (
                "geomagnetic storm", "geomagnetic", "magnetic storm", "space weather",
                "solar storm", "solar wind", "magnetosphere", "ionosphere",
                "aurora", "coronal mass ejection", "cme",
                "地磁暴", "地磁", "空间天气", "太阳风", "磁层", "电离层", "极光", "日冕物质抛射",
            ),
            "space weather with solar wind, magnetosphere, and aurora physics",
        ),
        # ── Climate / Earth ──
        (
            (
                "climate", "气候", "天气", "地球", "海洋", "生态", "遥感",
                "weather", "earth", "geology", "ocean", "environment",
                "ecology", "remote sensing", "urban", "hydrology",
                "sustainability", "carbon",
            ),
            "earth system science with atmospheric layers, satellite swaths, and climate contours",
        ),
        # ── Energy ──
        (
            (
                "energy", "能源", "电力", "电网", "renewable", "solar", "wind",
                "photovoltaic", "storage system", "grid",
            ),
            "energy systems with power grid networks and renewable generation sources",
        ),
        # ── Systems / Networks ──
        (
            (
                "systems", "系统", "网络", "分布式", "通信", "调度", "数据库",
                "network", "distributed", "communication", "scheduling",
                "database", "storage", "compiler", "operating system",
                "architecture", "algorithm", "infrastructure",
            ),
            "computing architecture with modular blocks, routing lines, and layered structure",
        ),
        # ── Control / RL ──
        (
            (
                "control", "控制", "强化学习", "规划", "轨迹", "优化",
                "reinforcement learning", "rl", "robot learning", "planning",
                "policy", "control theory", "trajectory", "optimization",
            ),
            "control systems with feedback loops, trajectory curves, and policy landscapes",
        ),
        # ── Chemistry ──
        (
            (
                "chemistry", "化学", "分子反应", "合成", "催化反应",
                "reaction", "synthesis", "molecular interaction", "spectroscopy",
            ),
            "chemistry with molecular structures, reaction pathways, and energy profiles",
        ),
        # ── Math ──
        (
            (
                "math", "数学", "定理", "证明", "代数", "几何", "拓扑", "矩阵", "概率", "统计",
                "graph theory", "linear algebra", "calculus", "theorem", "proof",
                "matrix", "spectral",
            ),
            "mathematical visualization with geometric constructions and algebraic notation",
        ),
        # ── Law ──
        (
            (
                "law", "法律", "法学", "司法", "法庭", "合规",
                "regulation", "legal", "court", "justice", "policy analysis",
            ),
            "legal studies with document layers, citation networks, and regulatory flowcharts",
        ),
        # ── Agriculture ──
        (
            (
                "agriculture", "农业", "作物", "土壤", "planting", "crop",
                "yield", "farm", "irrigation", "remote farming",
            ),
            "agricultural research with crop rows, soil cross-sections, and sensor markers",
        ),
    ]

    for terms, stage in topic_stages:
        if _contains_any(text, terms):
            return stage

    if "agent" in text or "tool" in text or "智能体" in text:
        return "AI agent workflow with interconnected tools and execution traces"
    return "academic research diagram with domain-relevant symbols and data-flow connections"


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
