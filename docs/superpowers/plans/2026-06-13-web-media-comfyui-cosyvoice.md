# Web Media ComfyUI CosyVoice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Web-first cover image generation through ComfyUI and narration audio generation through CosyVoice while keeping script generation as the core successful output.

**Architecture:** Add focused provider modules for media config, image generation, and voice generation. Integrate them into the existing Flask SSE pipeline after script generation, returning optional media paths plus warnings. Keep real ComfyUI/CosyVoice network calls behind injectable request functions so tests do not require running services.

**Tech Stack:** Python standard library (`urllib.request`, `json`, `mimetypes`, `pathlib`, `unittest`), existing Flask app, existing `.env` loader, existing SSE Web UI.

---

## File Structure

- Create `src/media_config.py`: load and normalize media provider environment variables.
- Create `src/image_generator.py`: build cover prompt, call ComfyUI-compatible HTTP endpoint, save image bytes.
- Create `src/voice_generator.py`: extract speech text, call CosyVoice-compatible HTTP endpoint, save audio bytes.
- Modify `src/utils.py`: add `IMAGE_DIR` and `AUDIO_DIR`, ensure directories exist.
- Modify `src/doctor.py`: report enabled/disabled media provider config without network checks.
- Modify `src/web_app.py`: add Web media pipeline steps, warning handling, image route, safer output path serving.
- Modify `src/templates/index.html`: add cover image card and warning display.
- Modify `.env.example`: add media provider settings.
- Modify `.gitignore`: ignore generated `data/images/*` and `data/audio/*`, keep `.gitkeep`.
- Modify `README.md`: document Web media provider setup and expected endpoints.
- Add tests:
  - `tests/test_media_config.py`
  - `tests/test_image_generator.py`
  - `tests/test_voice_generator.py`
  - extend `tests/test_doctor.py`
  - add focused `tests/test_web_app.py`

---

### Task 1: Media Config

**Files:**
- Create: `src/media_config.py`
- Test: `tests/test_media_config.py`

- [ ] **Step 1: Write failing media config tests**

```python
import os
import tempfile
import unittest
from pathlib import Path

from src.media_config import load_media_config


class MediaConfigTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("", encoding="utf-8")

            config = load_media_config(env_path)

            self.assertEqual(config.image.provider, "none")
            self.assertFalse(config.image.enabled)
            self.assertEqual(config.voice.provider, "none")
            self.assertFalse(config.voice.enabled)

    def test_loads_enabled_comfyui_and_cosyvoice(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "MEDIA_IMAGE_PROVIDER=comfyui\n"
                "COMFYUI_BASE_URL=http://127.0.0.1:8188/\n"
                "COMFYUI_TIMEOUT_SECONDS=12\n"
                "MEDIA_VOICE_PROVIDER=cosyvoice\n"
                "COSYVOICE_BASE_URL=http://127.0.0.1:50000/\n"
                "COSYVOICE_TIMEOUT_SECONDS=34\n"
                "COSYVOICE_VOICE=中文女声\n",
                encoding="utf-8",
            )

            config = load_media_config(env_path)

            self.assertTrue(config.image.enabled)
            self.assertEqual(config.image.base_url, "http://127.0.0.1:8188")
            self.assertEqual(config.image.timeout_seconds, 12)
            self.assertTrue(config.voice.enabled)
            self.assertEqual(config.voice.base_url, "http://127.0.0.1:50000")
            self.assertEqual(config.voice.timeout_seconds, 34)
            self.assertEqual(config.voice.voice, "中文女声")
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m unittest tests.test_media_config -v`

Expected: FAIL or ERROR because `src.media_config` does not exist.

- [ ] **Step 3: Implement media config**

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import load_dotenv_file


@dataclass(frozen=True)
class ImageProviderConfig:
    provider: str
    base_url: str
    timeout_seconds: int

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(frozen=True)
class VoiceProviderConfig:
    provider: str
    base_url: str
    timeout_seconds: int
    voice: str

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(frozen=True)
class MediaConfig:
    image: ImageProviderConfig
    voice: VoiceProviderConfig


def load_media_config(path: Path = Path(".env")) -> MediaConfig:
    load_dotenv_file(path)
    image_provider = _provider("MEDIA_IMAGE_PROVIDER")
    voice_provider = _provider("MEDIA_VOICE_PROVIDER")

    return MediaConfig(
        image=ImageProviderConfig(
            provider=image_provider,
            base_url=_url("COMFYUI_BASE_URL") if image_provider != "none" else "",
            timeout_seconds=_int("COMFYUI_TIMEOUT_SECONDS", 180),
        ),
        voice=VoiceProviderConfig(
            provider=voice_provider,
            base_url=_url("COSYVOICE_BASE_URL") if voice_provider != "none" else "",
            timeout_seconds=_int("COSYVOICE_TIMEOUT_SECONDS", 180),
            voice=os.getenv("COSYVOICE_VOICE", "default").strip() or "default",
        ),
    )


def _provider(name: str) -> str:
    value = os.getenv(name, "").strip().lower()
    return value or "none"


def _url(name: str) -> str:
    return os.getenv(name, "").strip().rstrip("/")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0")
    return value
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m unittest tests.test_media_config -v`

Expected: OK.

---

### Task 2: Output Directories

**Files:**
- Modify: `src/utils.py`
- Test: `tests/test_utils.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing directory test**

Add to `tests/test_utils.py`:

```python
def test_ensure_project_dirs_includes_media_dirs(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = ensure_project_dirs(root)

        expected = {
            root / "data" / "images",
            root / "data" / "audio",
        }
        self.assertTrue(expected.issubset(set(paths)))
        for path in expected:
            self.assertTrue(path.is_dir())
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m unittest tests.test_utils -v`

Expected: FAIL because media directories are not created.

- [ ] **Step 3: Implement media directories**

In `src/utils.py`, add:

```python
IMAGE_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
```

Add to `ensure_project_dirs()`:

```python
root / "data" / "images",
root / "data" / "audio",
```

Update `.gitignore`:

```text
data/images/*
data/audio/*
!data/images/.gitkeep
!data/audio/.gitkeep
```

- [ ] **Step 4: Run test and verify it passes**

Run: `python -m unittest tests.test_utils -v`

Expected: OK.

---

### Task 3: ComfyUI Image Generator

**Files:**
- Create: `src/image_generator.py`
- Test: `tests/test_image_generator.py`

- [ ] **Step 1: Write failing image generator tests**

```python
import tempfile
import unittest
from pathlib import Path

from src.image_generator import build_cover_prompt, generate_cover_image
from src.media_config import ImageProviderConfig


class ImageGeneratorTests(unittest.TestCase):
    def test_build_cover_prompt_uses_script_title_and_keywords(self):
        script = "# 播报标题\nAI 论文收音机\n\n# 播报脚本\n今天我们聊一个 agent 系统。\n\n# 关键词\n- Agent\n- Tool Learning\n"

        prompt = build_cover_prompt(script)

        self.assertIn("AI 论文收音机", prompt)
        self.assertIn("Agent", prompt)
        self.assertIn("podcast cover", prompt)

    def test_disabled_provider_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_cover_image(
                "paper",
                "# 播报标题\n标题",
                Path(tmp),
                ImageProviderConfig("none", "", 1),
            )

            self.assertIsNone(result)

    def test_writes_returned_image_bytes(self):
        calls = []

        def fake_request(url, payload, timeout):
            calls.append((url, payload, timeout))
            return b"\x89PNG\r\n\x1a\nimage"

        with tempfile.TemporaryDirectory() as tmp:
            path = generate_cover_image(
                "paper",
                "# 播报标题\n标题",
                Path(tmp),
                ImageProviderConfig("comfyui", "http://local", 9),
                request_image=fake_request,
                force=True,
            )

            self.assertEqual(path, Path(tmp) / "paper_cover.png")
            self.assertEqual(path.read_bytes(), b"\x89PNG\r\n\x1a\nimage")
            self.assertEqual(calls[0][0], "http://local/papercast/txt2img")
            self.assertEqual(calls[0][2], 9)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m unittest tests.test_image_generator -v`

Expected: FAIL or ERROR because `src.image_generator` does not exist.

- [ ] **Step 3: Implement image generator**

Create `src/image_generator.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from urllib import request

from .media_config import ImageProviderConfig
from .utils import read_text


def build_cover_prompt(script: str) -> str:
    title = _section_first_line(script, "# 播报标题") or "PaperCast AI podcast"
    keywords = _section_lines(script, "# 关键词", limit=6)
    keyword_text = ", ".join(line.lstrip("- ").strip() for line in keywords)
    return (
        f"podcast cover art for an AI research episode, title: {title}, "
        f"keywords: {keyword_text}, clean editorial illustration, readable, high contrast"
    )


def generate_cover_image(
    paper_id: str,
    script: str | Path,
    output_dir: Path,
    config: ImageProviderConfig,
    force: bool = False,
    request_image=None,
) -> Path | None:
    if not config.enabled:
        return None
    if config.provider != "comfyui":
        raise RuntimeError(f"Unsupported image provider: {config.provider}")
    if not config.base_url:
        raise RuntimeError("COMFYUI_BASE_URL is required when MEDIA_IMAGE_PROVIDER=comfyui")

    output = output_dir / f"{paper_id}_cover.png"
    if output.exists() and not force:
        return output

    output_dir.mkdir(parents=True, exist_ok=True)
    script_text = read_text(script) if isinstance(script, Path) else script
    payload = {"prompt": build_cover_prompt(script_text), "paper_id": paper_id}
    requester = request_image or _request_comfyui_image
    image_bytes = requester(f"{config.base_url}/papercast/txt2img", payload, config.timeout_seconds)
    if not image_bytes:
        raise RuntimeError("ComfyUI returned empty image data")
    output.write_bytes(image_bytes)
    return output


def _request_comfyui_image(url: str, payload: dict, timeout: int) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise RuntimeError(f"ComfyUI request failed at {url}: {exc}") from exc


def _section_first_line(text: str, heading: str) -> str:
    lines = _section_lines(text, heading, limit=1)
    return lines[0] if lines else ""


def _section_lines(text: str, heading: str, limit: int) -> list[str]:
    lines = text.splitlines()
    result: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == heading:
            in_section = True
            continue
        if in_section and line.startswith("# "):
            break
        if in_section and line.strip():
            result.append(line.strip())
            if len(result) >= limit:
                break
    return result
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m unittest tests.test_image_generator -v`

Expected: OK.

---

### Task 4: CosyVoice Audio Generator

**Files:**
- Create: `src/voice_generator.py`
- Test: `tests/test_voice_generator.py`

- [ ] **Step 1: Write failing voice generator tests**

```python
import tempfile
import unittest
from pathlib import Path

from src.media_config import VoiceProviderConfig
from src.voice_generator import build_speech_text, generate_voice_audio


class VoiceGeneratorTests(unittest.TestCase):
    def test_build_speech_text_extracts_script_and_strips_markdown(self):
        script = "# 播报脚本\n**第一句**。[uv_break]\n[链接](https://example.com)\n\n# 关键词\n- Agent\n"

        text = build_speech_text(script)

        self.assertEqual(text, "第一句。\n链接")

    def test_disabled_provider_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_voice_audio(
                "paper",
                "# 播报脚本\n正文",
                Path(tmp),
                VoiceProviderConfig("none", "", 1, "default"),
            )

            self.assertIsNone(result)

    def test_writes_returned_audio_bytes(self):
        calls = []

        def fake_request(url, payload, timeout):
            calls.append((url, payload, timeout))
            return b"RIFFaudio"

        with tempfile.TemporaryDirectory() as tmp:
            path = generate_voice_audio(
                "paper",
                "# 播报脚本\n正文",
                Path(tmp),
                VoiceProviderConfig("cosyvoice", "http://voice", 7, "中文女声"),
                request_audio=fake_request,
                force=True,
            )

            self.assertEqual(path, Path(tmp) / "paper_podcast.wav")
            self.assertEqual(path.read_bytes(), b"RIFFaudio")
            self.assertEqual(calls[0][0], "http://voice/papercast/tts")
            self.assertEqual(calls[0][1]["voice"], "中文女声")
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m unittest tests.test_voice_generator -v`

Expected: FAIL or ERROR because `src.voice_generator` does not exist.

- [ ] **Step 3: Implement voice generator**

Create `src/voice_generator.py` with:

```python
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib import request

from .media_config import VoiceProviderConfig
from .tts import _extract_script_body
from .utils import read_text


def build_speech_text(script: str) -> str:
    text = _extract_script_body(script)
    text = text.replace("[uv_break]", "\n")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def generate_voice_audio(
    paper_id: str,
    script: str | Path,
    output_dir: Path,
    config: VoiceProviderConfig,
    force: bool = False,
    request_audio=None,
) -> Path | None:
    if not config.enabled:
        return None
    if config.provider != "cosyvoice":
        raise RuntimeError(f"Unsupported voice provider: {config.provider}")
    if not config.base_url:
        raise RuntimeError("COSYVOICE_BASE_URL is required when MEDIA_VOICE_PROVIDER=cosyvoice")

    output = output_dir / f"{paper_id}_podcast.wav"
    if output.exists() and not force:
        return output

    output_dir.mkdir(parents=True, exist_ok=True)
    script_text = read_text(script) if isinstance(script, Path) else script
    payload = {"text": build_speech_text(script_text), "voice": config.voice}
    requester = request_audio or _request_cosyvoice_audio
    audio_bytes = requester(f"{config.base_url}/papercast/tts", payload, config.timeout_seconds)
    if not audio_bytes:
        raise RuntimeError("CosyVoice returned empty audio data")
    output.write_bytes(audio_bytes)
    return output


def _request_cosyvoice_audio(url: str, payload: dict, timeout: int) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise RuntimeError(f"CosyVoice request failed at {url}: {exc}") from exc
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m unittest tests.test_voice_generator -v`

Expected: OK.

---

### Task 5: Doctor And Env Documentation

**Files:**
- Modify: `src/doctor.py`
- Modify: `.env.example`
- Test: `tests/test_doctor.py`

- [ ] **Step 1: Write failing doctor test**

Add to `tests/test_doctor.py`:

```python
def test_doctor_reports_media_provider_config(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "prompts").mkdir()
        (root / "prompts" / "map_prompt.txt").write_text("{chunk}", encoding="utf-8")
        (root / "prompts" / "reduce_prompt.txt").write_text("{summaries}", encoding="utf-8")
        (root / ".env").write_text(
            "LLM_API_KEY=key\nLLM_BASE_URL=https://example.com/v1\nLLM_MODEL=model\n"
            "MEDIA_IMAGE_PROVIDER=comfyui\nCOMFYUI_BASE_URL=http://127.0.0.1:8188\n"
            "MEDIA_VOICE_PROVIDER=cosyvoice\nCOSYVOICE_BASE_URL=http://127.0.0.1:50000\n",
            encoding="utf-8",
        )

        report = run_doctor(root, check_optional_imports=False)

        self.assertTrue(report.ok)
        self.assertIn("Image provider configured: comfyui", report.messages)
        self.assertIn("Voice provider configured: cosyvoice", report.messages)
```

- [ ] **Step 2: Run test and verify it fails**

Run: `python -m unittest tests.test_doctor -v`

Expected: FAIL because doctor does not report media providers.

- [ ] **Step 3: Implement doctor messages and env example**

In `src/doctor.py`, import `load_media_config` and add:

```python
try:
    media_config = load_media_config(root / ".env")
    if media_config.image.enabled:
        if media_config.image.base_url:
            messages.append(f"Image provider configured: {media_config.image.provider}")
        else:
            errors.append("COMFYUI_BASE_URL is required when MEDIA_IMAGE_PROVIDER=comfyui")
    else:
        messages.append("Image provider disabled")
    if media_config.voice.enabled:
        if media_config.voice.base_url:
            messages.append(f"Voice provider configured: {media_config.voice.provider}")
        else:
            errors.append("COSYVOICE_BASE_URL is required when MEDIA_VOICE_PROVIDER=cosyvoice")
    else:
        messages.append("Voice provider disabled")
except Exception as exc:
    errors.append(str(exc))
```

Append to `.env.example`:

```text

MEDIA_IMAGE_PROVIDER=none
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_TIMEOUT_SECONDS=180

MEDIA_VOICE_PROVIDER=none
COSYVOICE_BASE_URL=http://127.0.0.1:50000
COSYVOICE_TIMEOUT_SECONDS=180
COSYVOICE_VOICE=default
```

- [ ] **Step 4: Run test and verify it passes**

Run: `python -m unittest tests.test_doctor -v`

Expected: OK.

---

### Task 6: Web Pipeline And API Routes

**Files:**
- Modify: `src/web_app.py`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Write focused route tests**

```python
import tempfile
import unittest
from pathlib import Path

from src import web_app


class WebAppTests(unittest.TestCase):
    def test_api_image_serves_image_under_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "data" / "images"
            image_dir.mkdir(parents=True)
            image = image_dir / "cover.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nimage")

            old_image_dir = web_app.IMAGE_DIR
            web_app.IMAGE_DIR = image_dir
            try:
                client = web_app.app.test_client()
                response = client.get(f"/api/image?path={image}")
            finally:
                web_app.IMAGE_DIR = old_image_dir

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, b"\x89PNG\r\n\x1a\nimage")

    def test_api_image_rejects_path_outside_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside.png"
            outside.write_bytes(b"image")

            client = web_app.app.test_client()
            response = client.get(f"/api/image?path={outside}")

            self.assertEqual(response.status_code, 403)
```

- [ ] **Step 2: Run route tests and verify they fail**

Run: `python -m unittest tests.test_web_app -v`

Expected: ERROR because `/api/image` does not exist or `IMAGE_DIR` is not imported.

- [ ] **Step 3: Implement route and media integration helpers**

In `src/web_app.py`, import:

```python
from src.image_generator import generate_cover_image
from src.media_config import load_media_config
from src.voice_generator import generate_voice_audio
from src.utils import AUDIO_DIR, IMAGE_DIR
```

Add:

```python
def _safe_output_path(path_text: str, allowed_dir: Path) -> Path:
    path = Path(path_text).resolve()
    allowed = allowed_dir.resolve()
    if allowed not in path.parents and path != allowed:
        raise PermissionError("path outside allowed output directory")
    if not path.exists():
        raise FileNotFoundError(path)
    return path
```

Add `/api/image`:

```python
@app.route("/api/image")
def api_image():
    path = request.args.get("path", "")
    if not path:
        return "missing path", 400
    try:
        p = _safe_output_path(path, IMAGE_DIR)
    except PermissionError:
        return "forbidden", 403
    except FileNotFoundError:
        return "not found", 404
    return Response(p.read_bytes(), mimetype="image/png")
```

Inside `_run_pipeline`, after script generation:

```python
media_config = load_media_config()
warnings = []
image_path = None
audio_path = None

try:
    _emit(q, "封面生成", "running", "正在生成播客封面...", 92)
    image_path = generate_cover_image(paper.paper_id, script_path, IMAGE_DIR, media_config.image, force=force)
    if image_path:
        _emit(q, "封面生成", "done", str(image_path), 94)
    else:
        _emit(q, "封面生成", "warning", "封面生成已禁用", 94)
except Exception as exc:
    warning = f"封面生成失败: {exc}"
    warnings.append(warning)
    _emit(q, "封面生成", "warning", warning, 94)

try:
    _emit(q, "音频合成", "running", "正在合成播报音频...", 96)
    audio_path = generate_voice_audio(paper.paper_id, script_path, AUDIO_DIR, media_config.voice, force=force)
    if audio_path:
        _emit(q, "音频合成", "done", str(audio_path), 98)
    else:
        _emit(q, "音频合成", "warning", "音频合成已禁用", 98)
except Exception as exc:
    warning = f"音频合成失败: {exc}"
    warnings.append(warning)
    _emit(q, "音频合成", "warning", warning, 98)

payload = {"script": str(script_path), "warnings": warnings}
if image_path:
    payload["image"] = str(image_path)
if audio_path:
    payload["audio"] = str(audio_path)
_emit(q, "完成", "done", json.dumps(payload, ensure_ascii=False), 100)
```

Remove or bypass the old unconditional `generate_audio(script_path)` Web call.

- [ ] **Step 4: Run Web tests and verify they pass**

Run: `python -m unittest tests.test_web_app -v`

Expected: OK.

---

### Task 7: Web UI Output

**Files:**
- Modify: `src/templates/index.html`

- [ ] **Step 1: Add UI placeholders**

Add an image section and warnings section near the existing audio section:

```html
<div id="warning-section"></div>
<div id="image-section">
  <img id="cover-image" alt="播客封面">
  <a id="image-download" href="#" download>下载封面</a>
</div>
```

Use existing pixel-card styling rather than adding a new visual language.

- [ ] **Step 2: Update JS completion handler**

In `handleSSE`, after parsing `paths`, add:

```javascript
if (paths.warnings && paths.warnings.length) {
  const warningSection = document.getElementById('warning-section');
  warningSection.style.display = 'block';
  warningSection.textContent = paths.warnings.join('\n');
}
if (paths.image) {
  document.getElementById('image-section').style.display = 'block';
  document.getElementById('cover-image').src = '/api/image?path=' + encodeURIComponent(paths.image);
  document.getElementById('image-download').href = '/api/image?path=' + encodeURIComponent(paths.image);
}
```

Also update `STEPS` to include `封面生成` and keep `音频合成`.

- [ ] **Step 3: Manually inspect HTML for syntax**

Run: `python -m unittest tests.test_web_app -v`

Expected: OK. If the Flask template fails to render, fix HTML syntax.

---

### Task 8: README And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document media setup**

Add a Web media section:

```markdown
## Web Media Providers

PaperCast can call already-running local ComfyUI and CosyVoice services from the Web pipeline.

Set providers in `.env`:

```text
MEDIA_IMAGE_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188

MEDIA_VOICE_PROVIDER=cosyvoice
COSYVOICE_BASE_URL=http://127.0.0.1:50000
COSYVOICE_VOICE=default
```

For the first version, PaperCast expects simple adapter endpoints:

- `POST {COMFYUI_BASE_URL}/papercast/txt2img` returns PNG bytes.
- `POST {COSYVOICE_BASE_URL}/papercast/tts` returns WAV bytes.

Use `none` to disable either provider.
```

- [ ] **Step 2: Run full test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 3: Check Git status**

Run: `git status --short`

Expected: only intended source, test, doc, and config files modified.

- [ ] **Step 4: Commit implementation**

```bash
git add .env.example .gitignore README.md src tests
git commit -m "Add Web media generation providers"
```

Do not commit `.env`, `.venv`, `.superpowers`, or generated `data/` outputs.

---

## Self-Review

- Spec coverage: config, output directories, provider modules, Web SSE behavior, warnings, `/api/image`, doctor, README, and tests are covered by Tasks 1-8.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: provider dataclasses are introduced in Task 1 and reused in Tasks 3, 4, and 6 with consistent names.
- Scope check: the plan does not start external services, install models, add a queue, or add voice cloning/image editing UI.
