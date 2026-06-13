from __future__ import annotations

from pathlib import Path


def convert_pdf_to_markdown(
    pdf_path: Path,
    markdown_path: Path,
    force: bool = False,
    converter=None,
) -> Path:
    if markdown_path.exists() and not force:
        print(f"Markdown exists, skipping: {markdown_path}")
        return markdown_path

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    if converter is None:
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise RuntimeError(
                "MarkItDown is not installed. Install requirements before converting PDFs."
            ) from exc
        converter = MarkItDown()

    try:
        result = converter.convert(str(pdf_path))
        text = _extract_markdown_text(result)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert PDF to Markdown: {pdf_path}: {exc}") from exc

    if not text.strip():
        raise RuntimeError(f"Failed to convert PDF to Markdown: {pdf_path}: empty Markdown")

    markdown_path.write_text(text, encoding="utf-8")
    return markdown_path


def _extract_markdown_text(result) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text_content", None)
    if text is not None:
        return str(text)
    markdown = getattr(result, "markdown", None)
    if markdown is not None:
        return str(markdown)
    return str(result)
