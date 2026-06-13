# PaperCast Agent MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable Python CLI MVP that converts one arXiv ID or PDF URL into cached Markdown summaries and a Chinese podcast script.

**Architecture:** The CLI orchestrates small modules for input resolution, download/cache, Markdown conversion, chunking, map summarization, and reduce script generation. The local filesystem is used for cache and resume behavior.

**Tech Stack:** Python 3.10+, argparse, requests, python-dotenv, MarkItDown, LangChain text splitters, OpenAI-compatible chat completions through HTTP.

---

## File Structure

- `src/main.py`: CLI parsing and pipeline orchestration.
- `src/config.py`: environment loading and LLM settings validation.
- `src/arxiv_client.py`: arXiv ID and PDF URL resolution.
- `src/pdf_downloader.py`: HTTP PDF download with cache skip.
- `src/markdown_converter.py`: MarkItDown PDF to Markdown conversion with cache skip.
- `src/splitter.py`: Markdown chunking and chunk metadata.
- `src/llm_client.py`: OpenAI-compatible chat completion client with retries.
- `src/summarizer.py`: Map-stage chunk summary generation and resume behavior.
- `src/script_writer.py`: Reduce-stage final script generation.
- `src/utils.py`: path setup, stable names, and file helpers.
- `tests/`: unit tests for core behavior.

## Tasks

- [ ] Write failing tests for input resolution, cache skip, splitter, LLM retries, and CLI validation.
- [ ] Implement minimal modules to pass tests.
- [ ] Add prompt files and project metadata.
- [ ] Run `python -m pytest` and `python src/main.py --help`.
- [ ] Fix any failures and document usage in `README.md`.
