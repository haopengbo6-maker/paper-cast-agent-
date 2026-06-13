"""PaperCast Agent — pixel-art podcast web UI with SSE progress."""

from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path

from flask import Flask, Response, render_template, request

# Ensure src/ is importable
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arxiv_client import resolve_paper_input, resolve_local_pdf
from src.config import load_llm_config
from src.llm_client import OpenAICompatibleClient, retry_call
from src.markdown_converter import convert_pdf_to_markdown
from src.pdf_downloader import download_pdf
from src.prompts import load_prompt
from src.script_writer import write_script, validate_script_structure
from src.tts import generate_audio
from src.splitter import split_markdown
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
    write_text,
)

app = Flask(__name__)
UPLOAD_FOLDER = Path.cwd() / "data" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# ─── SSE helpers ────────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _emit(q: queue.Queue, step: str, status: str, message: str, percent: int) -> None:
    q.put({"step": step, "status": status, "message": message, "percent": percent})


# ─── Pipeline runner (runs in background thread) ────────────────────

def _run_pipeline(q: queue.Queue, arxiv_id: str | None, pdf_url: str | None, local_file: Path | None, force: bool) -> None:
    try:
        ensure_project_dirs()
        llm_config = load_llm_config()
        llm_client = OpenAICompatibleClient(llm_config)
        map_prompt = load_prompt(PROMPT_DIR / "map_prompt.txt", required_placeholder="{chunk}")
        reduce_prompt = load_prompt(PROMPT_DIR / "reduce_prompt.txt", required_placeholder="{summaries}")

        # Step 1 — Resolve input
        _emit(q, "输入解析", "running", "正在解析论文输入...", 5)
        if local_file:
            paper = resolve_local_pdf(str(local_file))
            pdf_path = PDF_DIR / f"{paper.paper_id}.pdf"
            import shutil
            src = local_file.resolve()
            if src != pdf_path.resolve():
                shutil.copy2(src, pdf_path)
        else:
            paper = resolve_paper_input(arxiv_id, pdf_url)
            pdf_path = PDF_DIR / f"{paper.paper_id}.pdf"
            _emit(q, "PDF下载", "running", "正在下载论文 PDF...", 15)
            download_pdf(paper.pdf_url, pdf_path, force=force)
        _emit(q, "输入解析", "done", f"论文 ID: {paper.paper_id}", 20)

        # Step 2 — Convert to Markdown
        markdown_path = MARKDOWN_DIR / f"{paper.paper_id}.md"
        _emit(q, "Markdown转换", "running", "正在将 PDF 转换为 Markdown...", 25)
        convert_pdf_to_markdown(pdf_path, markdown_path, force=force)
        _emit(q, "Markdown转换", "done", "Markdown 转换完成", 35)

        # Step 3 — Chunk
        markdown = read_text(markdown_path)
        chunks = split_markdown(markdown, paper_id=paper.paper_id, source_file=str(markdown_path))
        _emit(q, "文本分块", "done", f"切分为 {len(chunks)} 个片段", 45)

        # Step 4 — Summarize (use the existing module)
        _emit(q, "LLM摘要", "running", f"正在逐块生成摘要 (0/{len(chunks)})...", 50)
        summary_paths = summarize_chunks(
            chunks, paper_id=paper.paper_id, summary_dir=SUMMARY_DIR,
            map_prompt=map_prompt, llm_client=llm_client, force=force,
        )
        _emit(q, "LLM摘要", "done", f"全部 {len(chunks)} 个片段摘要完成", 75)

        # Step 5 — Generate script (use the existing module)
        script_path = SCRIPT_DIR / f"{paper.paper_id}_script.md"
        _emit(q, "生成脚本", "running", "正在生成中文播客脚本...", 80)
        write_script(
            summary_paths, script_path=script_path,
            reduce_prompt=reduce_prompt, llm_client=llm_client, force=force,
        )
        _emit(q, "生成脚本", "done", "播客脚本生成完毕!", 90)

        # Step 6 — Generate audio
        _emit(q, "音频合成", "running", "正在合成语音...", 93)
        audio_path = generate_audio(script_path)
        _emit(q, "音频合成", "done", str(audio_path), 97)

        _emit(q, "完成", "done", json.dumps({"script": str(script_path), "audio": str(audio_path)}), 100)
        q.put(None)

    except Exception as exc:
        _emit(q, "错误", "error", str(exc), 0)
        q.put(None)


# ─── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    arxiv_id = request.form.get("arxiv_id", "").strip() or None
    pdf_url = request.form.get("pdf_url", "").strip() or None
    force = request.form.get("force") == "on"

    local_file = None
    uploaded = request.files.get("local_pdf")
    if uploaded and uploaded.filename:
        local_file = UPLOAD_FOLDER / uploaded.filename
        uploaded.save(str(local_file))

    if not arxiv_id and not pdf_url and not local_file:
        return Response(_sse_event({"step": "错误", "status": "error", "message": "请提供 arXiv ID、PDF URL 或上传 PDF 文件", "percent": 0}) + _sse_event({"step": "_done", "status": "done", "message": "", "percent": 0}), mimetype="text/event-stream")

    q: queue.Queue = queue.Queue()

    def generate():
        thread = threading.Thread(target=_run_pipeline, args=(q, arxiv_id, pdf_url, local_file, force), daemon=True)
        thread.start()
        while True:
            try:
                data = q.get(timeout=600)
            except queue.Empty:
                yield _sse_event({"step": "超时", "status": "error", "message": "处理超时", "percent": 0})
                return
            if data is None:
                return
            yield _sse_event(data)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/script")
def api_script():
    path = request.args.get("path", "")
    if not path:
        return "missing path", 400
    p = Path(path)
    if not p.exists():
        return "not found", 404
    return Response(read_text(p), mimetype="text/plain; charset=utf-8")


@app.route("/api/audio")
def api_audio():
    path = request.args.get("path", "")
    if not path:
        return "missing path", 400
    p = Path(path)
    if not p.exists():
        return "not found", 404
    return Response(p.read_bytes(), mimetype="audio/mpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
