# PaperCast Agent

PaperCast Agent is a Python CLI MVP that converts one arXiv paper or PDF URL into a Chinese podcast script.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in an OpenAI-compatible API:

```text
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=deepseek-chat
SUMMARY_MAX_WORKERS=3
```

Optional Web media providers can be enabled in the same `.env` file:

```text
MEDIA_IMAGE_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188

MEDIA_VOICE_PROVIDER=cosyvoice
COSYVOICE_BASE_URL=http://127.0.0.1:50000
COSYVOICE_VOICE=default
```

Use `none` to disable either provider.

The Web UI also includes a fast mode. It uses larger chunks, less overlap, parallel chunk summaries, and skips media generation so you can get a script sooner.

## Usage

```bash
python src/main.py --version
python src/main.py --help
python src/main.py --doctor
python src/main.py --arxiv-id "2401.00000"
python src/main.py --pdf-url "https://arxiv.org/pdf/2401.00000"
python src/main.py --arxiv-id "2401.00000" --force
```

Outputs are cached under `data/`:

```text
data/pdfs/
data/markdown/
data/chunks/
data/summaries/
data/scripts/
data/images/
data/audio/
```

Existing outputs are skipped by default. Use `--force` to regenerate.

## Day 2 PDF And Markdown Handling

PDF files are downloaded to `data/pdfs/{paper_id}.pdf`. If the file already exists, the downloader skips the network request unless `--force` is used. Download failures include the URL, HTTP status when available, and the underlying reason.

Markdown files are written to `data/markdown/{paper_id}.md`. Existing Markdown is skipped unless `--force` is used. The converter uses MarkItDown and fails clearly if the dependency is missing, conversion raises an error, or the generated Markdown is empty.

## Day 3 Chunking

Markdown is split into chunks with metadata for each segment. The CLI prints the chunk count and previews of the first two chunks, then writes debug artifacts to `data/chunks/`:

```text
data/chunks/{paper_id}_chunk_001.md
data/chunks/{paper_id}_chunk_002.md
data/chunks/{paper_id}_chunks.jsonl
```

The JSONL file stores `paper_id`, `chunk_id`, `source_file`, and `char_length` for each chunk.

## Day 4 Map Summaries

Each chunk is summarized independently and saved to `data/summaries/{paper_id}_chunk_001.md`. Existing summary files are skipped by default so failed runs can resume from the next missing chunk. Use `--force` to regenerate all summaries.

If one chunk fails after retries, completed summary files stay on disk and the error message includes the failing chunk number. Prompt files are checked before model calls: `prompts/map_prompt.txt` must contain `{chunk}`, and `prompts/reduce_prompt.txt` must contain `{summaries}`.

## Day 5 Reduce Script

The reduce stage combines all chunk summaries into `data/scripts/{paper_id}_script.md`. Existing scripts are skipped by default and regenerated with `--force`.

Before calling the model, the script writer checks that summary files exist and are not empty. After the model responds, the generated script must contain these sections before it is written:

```text
# 播报标题
# 播报脚本
# 关键词
# 适合延伸学习的概念
```

## Day 6 Doctor Check

Run this before a real paper job:

```bash
python src/main.py --doctor
```

The doctor check verifies local data directories, prompt placeholders, LLM environment variables, and the MarkItDown dependency. It prints a report and exits non-zero if a required item is missing.

## Day 7 Demo

See `docs/demo-day7.md` for the real-paper demo runbook: virtualenv setup, `.env` configuration, doctor checks, sample arXiv command, expected outputs, and resume behavior.

## Web Media Providers

PaperCast can call already-running local ComfyUI and CosyVoice services from the Web pipeline. PaperCast does not install models or start those services.

For the first version, PaperCast expects simple adapter endpoints:

```text
POST {COMFYUI_BASE_URL}/papercast/txt2img
POST {COSYVOICE_BASE_URL}/papercast/tts
```

The ComfyUI adapter endpoint should return PNG bytes. The CosyVoice adapter endpoint should return WAV bytes.

If image or audio generation fails, the Web run still returns the generated script and shows a media warning.

## Test

```bash
python -m unittest discover -s tests -v
python src/main.py --help
```

## Day 1 Project Scaffold

The project is intentionally split into small modules:

```text
src/main.py                CLI and pipeline orchestration
src/config.py              .env loading and LLM config validation
src/arxiv_client.py        arXiv ID / PDF URL resolution
src/pdf_downloader.py      PDF download and cache skip
src/markdown_converter.py  MarkItDown conversion
src/splitter.py            Markdown chunking
src/llm_client.py          OpenAI-compatible chat client and retry helper
src/summarizer.py          Map-stage chunk summaries
src/script_writer.py       Reduce-stage final script
src/utils.py               paths and file helpers
```

CLI validation happens before network or model calls, so setup errors fail quickly.
