from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
MARKDOWN_DIR = DATA_DIR / "markdown"
SUMMARY_DIR = DATA_DIR / "summaries"
SCRIPT_DIR = DATA_DIR / "scripts"
CHUNK_DIR = DATA_DIR / "chunks"
PROMPT_DIR = PROJECT_ROOT / "prompts"


def ensure_project_dirs(root: Path = PROJECT_ROOT) -> list[Path]:
    paths = [
        root / "data" / "pdfs",
        root / "data" / "markdown",
        root / "data" / "chunks",
        root / "data" / "summaries",
        root / "data" / "scripts",
        root / "prompts",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def preview(text: str, length: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= length:
        return compact
    return compact[:length].rstrip() + "..."
