# Day 7 Demo Runbook

This runbook prepares a real PaperCast Agent demo from a fresh checkout.

## 1. Create A Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If dependency installation is slow or interrupted, rerun:

```bash
.venv\Scripts\python.exe -m pip install --no-input --disable-pip-version-check -r requirements.txt
```

## 2. Configure LLM Access

Create `.env` from `.env.example`:

```text
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=deepseek-chat
```

The API must support the OpenAI-compatible `/chat/completions` endpoint.

## 3. Run Doctor

```bash
.venv\Scripts\python.exe src/main.py --doctor
```

Expected successful ending:

```text
Status: OK
```

If it reports missing LLM config, check `.env`. If it reports missing MarkItDown, rerun dependency installation.

## 4. Run A Real Paper

Use a small recent arXiv paper first:

```bash
.venv\Scripts\python.exe src/main.py --arxiv-id "2401.00000"
```

Or use a direct PDF URL:

```bash
.venv\Scripts\python.exe src/main.py --pdf-url "https://arxiv.org/pdf/2401.00000"
```

## 5. Expected Outputs

```text
data/pdfs/{paper_id}.pdf
data/markdown/{paper_id}.md
data/chunks/{paper_id}_chunk_001.md
data/chunks/{paper_id}_chunks.jsonl
data/summaries/{paper_id}_chunk_001.md
data/scripts/{paper_id}_script.md
```

## 6. Resume Or Regenerate

Rerun the same command to resume from existing files. Use `--force` to regenerate all cached outputs:

```bash
.venv\Scripts\python.exe src/main.py --arxiv-id "2401.00000" --force
```

## Current Environment Note

During setup in this workspace, dependency installation was interrupted before completion. The project code and tests still run with the system Python because unit tests avoid real MarkItDown and network calls. A real demo needs dependency installation plus `.env` configuration.
