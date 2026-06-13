# PaperCast Agent MVP Design

## Goal

Build a Python CLI that turns one arXiv paper or PDF URL into a cached Chinese podcast script suitable for TTS playback.

## Scope

The MVP supports a single paper per run. It downloads a PDF, converts it to Markdown, splits long text into chunks, summarizes each chunk with an OpenAI-compatible LLM, and combines the summaries into a final Chinese script with `[uv_break]` pause markers.

Out of scope: audio generation, web UI, login, scheduled fetching, batch paper reports, and deep visual understanding of formulas, tables, or images.

## Architecture

`src/main.py` owns CLI parsing and pipeline orchestration. Each pipeline stage lives in a small module with one responsibility: resolving paper inputs, downloading PDFs, converting Markdown, splitting text, calling the LLM for map summaries, and writing the reduce script.

Local files are the cache and resume mechanism. Existing outputs are skipped unless `--force` is passed.

## Data Flow

1. User passes `--arxiv-id` or `--pdf-url`.
2. The app resolves a stable paper id and PDF URL.
3. The PDF is saved to `data/pdfs/`.
4. Markdown is saved to `data/markdown/`.
5. Chunks are generated in memory from Markdown.
6. Chunk summaries are saved under `data/summaries/`.
7. The final script is saved under `data/scripts/`.

## Error Handling

Missing inputs, missing LLM configuration, failed downloads, failed PDF conversion, and failed LLM calls should produce clear messages. Chunk summaries are written one by one so a later run can continue from the last successful chunk.

## Testing

Unit tests cover input resolution, cache behavior, text splitting, LLM retry behavior, and CLI validation. Integration tests avoid real network and model calls by injecting simple fake clients.
