from __future__ import annotations

from pathlib import Path

from .llm_client import chat_with_optional_max_tokens, retry_call
from .utils import read_text, write_text


def write_script(
    summary_paths: list[Path],
    script_path: Path,
    reduce_prompt: str,
    llm_client,
    force: bool = False,
) -> Path:
    if script_path.exists() and not force:
        print(f"Script exists, skipping: {script_path}")
        return script_path

    summaries = _read_summaries(summary_paths)
    prompt = reduce_prompt.replace("{summaries}", summaries)
    script = retry_call(
        lambda: chat_with_optional_max_tokens(llm_client, prompt, max_tokens=2048),
        max_attempts=3,
    )
    validate_script_structure(script)
    write_text(script_path, script)
    return script_path


def _read_summaries(summary_paths: list[Path]) -> str:
    if not summary_paths:
        raise RuntimeError("No summary files found for reduce stage")

    texts: list[str] = []
    for path in summary_paths:
        text = read_text(path)
        if not text.strip():
            raise RuntimeError(f"Cannot generate script from empty summary: {path}")
        texts.append(text)
    return "\n\n".join(texts)


def validate_script_structure(script: str) -> None:
    required_sections = [
        "# 播报标题",
        "# 播报脚本",
        "# 关键词",
        "# 适合延伸学习的概念",
    ]
    for section in required_sections:
        if section not in script:
            raise RuntimeError(f"Generated script missing required section: {section}")
