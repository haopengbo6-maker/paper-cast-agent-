from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .llm_client import chat_with_optional_max_tokens, retry_call
from .splitter import Chunk
from .utils import write_text


def summarize_chunks(
    chunks: list[Chunk],
    paper_id: str,
    summary_dir: Path,
    map_prompt: str,
    llm_client,
    force: bool = False,
    max_workers: int = 1,
) -> list[Path]:
    if max_workers <= 1:
        return _summarize_chunks_sequential(
            chunks,
            paper_id=paper_id,
            summary_dir=summary_dir,
            map_prompt=map_prompt,
            llm_client=llm_client,
            force=force,
        )

    return _summarize_chunks_parallel(
        chunks,
        paper_id=paper_id,
        summary_dir=summary_dir,
        map_prompt=map_prompt,
        llm_client=llm_client,
        force=force,
        max_workers=max_workers,
    )


def _summarize_chunks_sequential(
    chunks: list[Chunk],
    paper_id: str,
    summary_dir: Path,
    map_prompt: str,
    llm_client,
    force: bool,
) -> list[Path]:
    summary_paths: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        path = summary_dir / f"{paper_id}_chunk_{index:03d}.md"
        if path.exists() and not force:
            print(f"Summary exists, skipping: {path.name}")
            summary_paths.append(path)
            continue

        print(f"Processing chunk {index}/{len(chunks)}")
        prompt = map_prompt.replace("{chunk}", chunk.text)
        try:
            summary = retry_call(
                lambda: chat_with_optional_max_tokens(llm_client, prompt, max_tokens=600),
                max_attempts=3,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to summarize chunk {index}/{len(chunks)}: {exc}") from exc
        write_text(path, summary)
        summary_paths.append(path)
    return summary_paths


def _summarize_chunks_parallel(
    chunks: list[Chunk],
    paper_id: str,
    summary_dir: Path,
    map_prompt: str,
    llm_client,
    force: bool,
    max_workers: int,
) -> list[Path]:
    indexed_paths: list[Path | None] = [None] * len(chunks)
    pending: list[tuple[int, Chunk, Path]] = []

    for index, chunk in enumerate(chunks, start=1):
        path = summary_dir / f"{paper_id}_chunk_{index:03d}.md"
        indexed_paths[index - 1] = path
        if path.exists() and not force:
            print(f"Summary exists, skipping: {path.name}")
            continue
        pending.append((index, chunk, path))

    if not pending:
        return [path for path in indexed_paths if path is not None]

    def _process(item: tuple[int, Chunk, Path]) -> Path:
        index, chunk, path = item
        print(f"Processing chunk {index}/{len(chunks)}")
        prompt = map_prompt.replace("{chunk}", chunk.text)
        try:
            summary = retry_call(
                lambda: chat_with_optional_max_tokens(llm_client, prompt, max_tokens=600),
                max_attempts=3,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to summarize chunk {index}/{len(chunks)}: {exc}") from exc
        write_text(path, summary)
        return path

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(_process, item): item[0] for item in pending}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            indexed_paths[index - 1] = future.result()

    return [path for path in indexed_paths if path is not None]
