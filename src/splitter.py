from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict


@dataclass(frozen=True)
class ChunkExportPaths:
    chunk_files: list[Path]
    metadata_file: Path


def split_markdown(
    text: str,
    paper_id: str,
    source_file: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    raw_chunks = _langchain_split(text, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for index, chunk_text in enumerate(raw_chunks, start=1):
        cleaned = chunk_text.strip()
        if not cleaned:
            continue
        chunks.append(
            Chunk(
                text=cleaned,
                metadata={
                    "paper_id": paper_id,
                    "chunk_id": index,
                    "source_file": source_file,
                    "char_length": len(cleaned),
                },
            )
        )
    return chunks


def format_chunk_previews(
    chunks: list[Chunk],
    limit: int = 2,
    preview_length: int = 240,
) -> list[str]:
    previews: list[str] = []
    for chunk in chunks[:limit]:
        compact = " ".join(chunk.text.split())
        if len(compact) > preview_length:
            compact = compact[:preview_length].rstrip() + "..."
        previews.append(f"Chunk {chunk.metadata['chunk_id']} preview: {compact}")
    return previews


def export_chunks(chunks: list[Chunk], export_dir: Path, force: bool = False) -> ChunkExportPaths:
    export_dir.mkdir(parents=True, exist_ok=True)
    chunk_files: list[Path] = []
    metadata_lines: list[str] = []

    for chunk in chunks:
        paper_id = chunk.metadata["paper_id"]
        chunk_id = int(chunk.metadata["chunk_id"])
        chunk_path = export_dir / f"{paper_id}_chunk_{chunk_id:03d}.md"
        if force or not chunk_path.exists():
            chunk_path.write_text(chunk.text, encoding="utf-8")
        chunk_files.append(chunk_path)
        metadata_lines.append(json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True))

    paper_id = chunks[0].metadata["paper_id"] if chunks else "empty"
    metadata_file = export_dir / f"{paper_id}_chunks.jsonl"
    if force or not metadata_file.exists():
        metadata_file.write_text("\n".join(metadata_lines) + ("\n" if metadata_lines else ""), encoding="utf-8")

    return ChunkExportPaths(chunk_files=chunk_files, metadata_file=metadata_file)


def _langchain_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _simple_split(text, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n# ", "\n## ", "\n\n", "\n", "。", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _simple_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        split_at = _find_boundary(text, start, end)
        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        if split_at >= len(text):
            break
        next_start = max(0, split_at - chunk_overlap)
        if next_start <= start:
            next_start = split_at
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start
    return chunks


def _find_boundary(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return len(text)
    window = text[start:end]
    for marker in ("\n\n", "\n", "。", ". ", " "):
        index = window.rfind(marker)
        if index > 0:
            return start + index + len(marker)
    return end
