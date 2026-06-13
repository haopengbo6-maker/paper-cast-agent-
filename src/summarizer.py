from __future__ import annotations

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
