# PaperCast Agent

PaperCast Agent turns an arXiv paper, PDF URL, or local PDF into a Chinese paper podcast workflow. It can generate the script, optional cover art, and optional voice audio, then present the result in a local art-book style Web UI.

The current visual direction is intentionally not photorealistic: covers are 2D research plates with colored-pencil sketching, messy construction lines, restrained acrylic/oil texture, paper grain, registration marks, and catalogue-like spacing.

## Features

- Resolve arXiv IDs, PDF URLs, or uploaded local PDFs.
- Download and cache PDFs under `data/pdfs/`.
- Convert PDFs to Markdown with MarkItDown.
- Split long papers into resumable chunks.
- Summarize chunks with an OpenAI-compatible LLM endpoint.
- Reduce summaries into a structured Chinese podcast script.
- Run a local Flask Web UI with fast mode and full mode.
- Generate discipline-aware 2D cover images through a running ComfyUI instance.
- Optionally synthesize audio through a local CosyVoice-compatible service.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your OpenAI-compatible LLM settings:

```text
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=deepseek-chat
SUMMARY_MAX_WORKERS=3
```

## Run The Web UI

```bash
python src/web_app.py
```

Open the URL printed by Flask, usually:

```text
http://127.0.0.1:5000
```

The Web UI supports:

- arXiv ID input
- PDF URL input
- local PDF upload
- force regeneration
- fast mode, which skips media generation and uses larger chunks
- full mode, which runs cover and audio generation when providers are enabled

## Run The CLI

```bash
python src/main.py --version
python src/main.py --help
python src/main.py --doctor
python src/main.py --arxiv-id "2401.00000"
python src/main.py --pdf-url "https://arxiv.org/pdf/2401.00000"
python src/main.py --arxiv-id "2401.00000" --force
```

Existing outputs are skipped by default. Use `--force` to regenerate.

## Output Files

Generated files are cached under `data/`:

```text
data/pdfs/        downloaded or uploaded PDFs
data/markdown/    converted Markdown
data/chunks/      split Markdown chunks and JSONL metadata
data/summaries/   per-chunk map summaries
data/scripts/     final Chinese podcast scripts
data/images/      generated cover images
data/audio/       generated audio files
```

## Script Format

The final script is expected to include these sections:

```text
# 播报标题
# 播报脚本
# 关键词
# 适合延伸学习的概念
```

The cover generator also reads the title, keywords, the first script paragraph, and the first summary cue so the visual concept can follow the actual paper instead of falling back to generic AI imagery.

## ComfyUI Cover Generation

PaperCast expects ComfyUI to already be running. It does not install ComfyUI or download checkpoint models.

Enable image generation in `.env`:

```text
MEDIA_IMAGE_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_TIMEOUT_SECONDS=180
```

The current implementation talks to ComfyUI's native endpoints:

```text
GET  /object_info
POST /prompt
GET  /history/{prompt_id}
GET  /view
```

It selects the first available `CheckpointLoaderSimple` checkpoint from ComfyUI, builds a simple SDXL-style workflow, and saves the generated PNG under `data/images/` with a prompt-versioned filename such as:

```text
data/images/{paper_id}_cover_v10.png
```

The generated prompt is discipline-aware. It has explicit visual branches for topics such as:

```text
flow matching / generative modeling
humanoid robotics
computer vision
large language models
medicine and pathology
physics
materials science
biology
social science and economics
climate and earth systems
systems and networking
control and reinforcement learning
chemistry
mathematics
law
energy systems
agriculture
```

This is meant to produce meaningful, topic-specific 2D plates rather than random abstract images. A light Pillow post-process adds subtle paper grain and print finish after ComfyUI returns the image.

## Voice Generation

Enable voice generation in `.env` when you have a compatible local service running:

```text
MEDIA_VOICE_PROVIDER=cosyvoice
COSYVOICE_BASE_URL=http://127.0.0.1:50000
COSYVOICE_TIMEOUT_SECONDS=180
COSYVOICE_VOICE=default
```

Use `MEDIA_IMAGE_PROVIDER=none` or `MEDIA_VOICE_PROVIDER=none` to disable either provider.

If image or audio generation fails during a Web run, PaperCast still returns the generated script and displays a media warning.

## Pipeline Notes

- PDF downloads are cached and include clear errors for failed HTTP requests.
- Markdown conversion fails clearly when MarkItDown is missing or produces empty output.
- Chunk summaries are resumable; completed summary files stay on disk if a later chunk fails.
- Prompt files are validated before model calls:
  - `prompts/map_prompt.txt` must contain `{chunk}`.
  - `prompts/reduce_prompt.txt` must contain `{summaries}`.
- `python src/main.py --doctor` checks local directories, prompt placeholders, LLM environment variables, and dependencies.

## Tests

```bash
python -m unittest discover -s tests -v
python -m unittest tests.test_image_generator -v
python src/main.py --help
```

`tests.test_image_generator` covers the ComfyUI workflow builder, checkpoint selection, prompt-versioned output paths, and the discipline-aware cover prompt mapper.

## Project Structure

```text
src/main.py                CLI and pipeline orchestration
src/web_app.py             Flask Web UI and SSE pipeline runner
src/config.py              .env loading and LLM config validation
src/arxiv_client.py        arXiv ID / PDF URL / local PDF resolution
src/pdf_downloader.py      PDF download and cache skip
src/markdown_converter.py  MarkItDown conversion
src/splitter.py            Markdown chunking
src/llm_client.py          OpenAI-compatible chat client and retry helper
src/summarizer.py          Map-stage chunk summaries
src/script_writer.py       Reduce-stage final script
src/image_generator.py     ComfyUI workflow and discipline-aware cover prompts
src/voice_generator.py     Optional voice generation
src/media_config.py        Media provider configuration
tests/                     Unit tests
```
