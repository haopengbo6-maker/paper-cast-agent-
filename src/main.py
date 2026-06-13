from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arxiv_client import resolve_paper_input, resolve_local_pdf
from src.config import load_llm_config
from src.doctor import run_doctor
from src.llm_client import OpenAICompatibleClient
from src.markdown_converter import convert_pdf_to_markdown
from src.pdf_downloader import download_pdf
from src.prompts import load_prompt
from src.script_writer import write_script
from src.splitter import split_markdown
from src.tts import generate_audio
from src.splitter import export_chunks, format_chunk_previews
from src.summarizer import summarize_chunks
from src.utils import (
    CHUNK_DIR,
    MARKDOWN_DIR,
    PDF_DIR,
    PROMPT_DIR,
    SCRIPT_DIR,
    SUMMARY_DIR,
    ensure_project_dirs,
    read_text,
)


VERSION = "PaperCast Agent 0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Chinese podcast script from an arXiv paper.")
    parser.add_argument("--arxiv-id", help="arXiv paper ID, for example 2401.00000")
    parser.add_argument("--pdf-url", help="Direct PDF URL")
    parser.add_argument("--local-pdf", help="Local PDF file path (bypasses download)")
    parser.add_argument("--force", action="store_true", help="Regenerate all cached outputs")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Chunk size, default 3000")
    parser.add_argument("--chunk-overlap", type=int, default=300, help="Chunk overlap, default 300")
    parser.add_argument("--doctor", action="store_true", help="Check local configuration and dependencies")
    parser.add_argument("--tts", action="store_true", help="Generate MP3 audio from the podcast script")
    parser.add_argument("--voice", default="huihui", choices=["huihui"],
                        help="TTS voice, default huihui (Chinese female)")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    sources = [args.arxiv_id, args.pdf_url, args.local_pdf]
    if sum(1 for s in sources if s) > 1:
        raise ValueError("Use either --arxiv-id or --pdf-url, or provide only --local-pdf")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be non-negative")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.doctor:
            from src.utils import PROJECT_ROOT

            report = run_doctor(PROJECT_ROOT)
            print(report.format())
            return 0 if report.ok else 1

        validate_args(args)
        ensure_project_dirs()

        if args.local_pdf:
            paper = resolve_local_pdf(args.local_pdf)
            pdf_path = PDF_DIR / f"{paper.paper_id}.pdf"
            src = Path(args.local_pdf).resolve()
            if src != pdf_path.resolve():
                shutil.copy2(src, pdf_path)
                print(f"Copied local PDF: {pdf_path}")
            else:
                print(f"Using local PDF: {pdf_path}")
        else:
            paper = resolve_paper_input(args.arxiv_id, args.pdf_url)
            pdf_path = PDF_DIR / f"{paper.paper_id}.pdf"
            download_pdf(paper.pdf_url, pdf_path, force=args.force)

        markdown_path = MARKDOWN_DIR / f"{paper.paper_id}.md"
        script_path = SCRIPT_DIR / f"{paper.paper_id}_script.md"

        convert_pdf_to_markdown(pdf_path, markdown_path, force=args.force)

        markdown = read_text(markdown_path)
        chunks = split_markdown(
            markdown,
            paper_id=paper.paper_id,
            source_file=str(markdown_path),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Generated {len(chunks)} chunks")
        chunk_exports = export_chunks(chunks, CHUNK_DIR, force=args.force)
        print(f"Saved chunk metadata: {chunk_exports.metadata_file}")
        for line in format_chunk_previews(chunks, limit=2):
            print(line)

        llm_config = load_llm_config()
        llm_client = OpenAICompatibleClient(llm_config)
        map_prompt = load_prompt(PROMPT_DIR / "map_prompt.txt", required_placeholder="{chunk}")
        reduce_prompt = load_prompt(PROMPT_DIR / "reduce_prompt.txt", required_placeholder="{summaries}")

        summary_paths = summarize_chunks(
            chunks,
            paper_id=paper.paper_id,
            summary_dir=SUMMARY_DIR,
            map_prompt=map_prompt,
            llm_client=llm_client,
            force=args.force,
        )
        write_script(
            summary_paths,
            script_path=script_path,
            reduce_prompt=reduce_prompt,
            llm_client=llm_client,
            force=args.force,
        )

        if args.tts:
            audio_path = generate_audio(script_path)
            print(f"Audio: {audio_path}")

        print(f"Done: {script_path}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
