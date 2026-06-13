from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class PaperInput:
    paper_id: str
    pdf_url: str


ARXIV_ID_PATTERN = re.compile(r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)")


def resolve_paper_input(arxiv_id: str | None, pdf_url: str | None) -> PaperInput:
    if not arxiv_id and not pdf_url:
        raise ValueError("Provide --arxiv-id or --pdf-url")

    if arxiv_id:
        clean_id = arxiv_id.strip()
        if not ARXIV_ID_PATTERN.fullmatch(clean_id):
            raise ValueError(f"Invalid arXiv ID: {arxiv_id}")
        return PaperInput(
            paper_id=clean_id,
            pdf_url=f"https://arxiv.org/pdf/{clean_id}.pdf",
        )

    assert pdf_url is not None
    clean_url = pdf_url.strip()
    paper_id = _paper_id_from_pdf_url(clean_url)
    return PaperInput(paper_id=paper_id, pdf_url=clean_url)


def resolve_local_pdf(local_pdf: str) -> PaperInput:
    path = Path(local_pdf).resolve()
    if not path.exists():
        raise ValueError(f"Local PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    paper_id = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-") or "local-paper"
    return PaperInput(paper_id=paper_id, pdf_url=str(path))


def _paper_id_from_pdf_url(pdf_url: str) -> str:
    parsed = urlparse(pdf_url)
    match = ARXIV_ID_PATTERN.search(parsed.path)
    if match:
        return match.group("id")

    name = parsed.path.rstrip("/").split("/")[-1] or "paper"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    if safe.endswith(".pdf"):
        safe = safe[:-4]
    return safe or "paper"
