# PaperCast Web Media Generation Design

## Goal

PaperCast should extend the existing Web flow from "paper to Chinese podcast script" into "paper to podcast package" by adding:

- A cover image generated through an already-running ComfyUI service.
- A Chinese narration audio file generated through an already-running CosyVoice service.

The first version is Web-first. The CLI may keep its current behavior except for shared modules that make future CLI support straightforward.

## Confirmed Scope

The Web UI keeps one primary run action. After script generation succeeds, the server attempts media generation as additional SSE progress steps:

1. Resolve arXiv ID, PDF URL, or uploaded PDF.
2. Download or copy PDF.
3. Convert PDF to Markdown.
4. Split Markdown into chunks.
5. Summarize chunks with the configured LLM.
6. Generate the final podcast script.
7. Generate a cover image with ComfyUI.
8. Generate narration audio with CosyVoice.
9. Return script, optional image, optional audio, and any media warnings.

PaperCast does not install models, download checkpoints, or start ComfyUI/CosyVoice processes. It only connects to services the user has already started locally or on another reachable host.

## Out Of Scope For V1

- Automatic ComfyUI or CosyVoice process management.
- Model installation, checkpoint download, GPU detection, or environment setup.
- Voice cloning UI.
- Image editing UI.
- Background job dashboard or persistent queue.
- Multiple cover image candidates.
- CLI flags for media generation unless they fall out naturally from shared code without broadening the implementation.

## Configuration

Add media configuration through `.env` and `.env.example`.

```text
MEDIA_IMAGE_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_TIMEOUT_SECONDS=180

MEDIA_VOICE_PROVIDER=cosyvoice
COSYVOICE_BASE_URL=http://127.0.0.1:50000
COSYVOICE_TIMEOUT_SECONDS=180
COSYVOICE_VOICE=default
```

Provider values may be empty or set to `none` to disable that media type. Missing or disabled media providers should not block script generation.

## Files And Outputs

Add these data directories:

```text
data/images/
data/audio/
```

Expected output names:

```text
data/images/{paper_id}_cover.png
data/audio/{paper_id}_podcast.wav
```

If a provider returns MP3 instead of WAV, the first version may preserve the returned extension, but the Web API must report the correct MIME type.

## Backend Architecture

### `src/media_config.py`

Owns loading and validating media-specific environment variables. It should be independent from `src/config.py` so LLM configuration stays focused.

Responsibilities:

- Read provider names, base URLs, timeouts, and voice settings.
- Normalize disabled providers.
- Expose a small dataclass that Web pipeline code can pass to image/audio generators.
- Avoid network calls during config load.

### `src/image_generator.py`

Owns cover image generation.

Responsibilities:

- Build a concise visual prompt from the generated script, likely using title, keywords, and a short excerpt.
- Call a ComfyUI HTTP API endpoint.
- Save the resulting image to `data/images/{paper_id}_cover.png`.
- Skip existing image output unless force regeneration is requested.
- Raise clear exceptions with endpoint, timeout, and response details when generation fails.

The implementation should isolate ComfyUI request formatting in one small function so the workflow payload can be adjusted without touching the Web pipeline.

### `src/voice_generator.py`

Owns CosyVoice narration generation.

Responsibilities:

- Reuse or share the current script-body extraction logic from `src/tts.py`.
- Strip Markdown formatting and `[uv_break]` markers into speech-friendly text.
- Call a CosyVoice HTTP API endpoint.
- Save the resulting audio to `data/audio/{paper_id}_podcast.wav` or the returned supported extension.
- Skip existing audio output unless force regeneration is requested.
- Raise clear exceptions with endpoint, timeout, and response details when generation fails.

The current `src/tts.py` pyttsx3 implementation can remain as a local fallback, but the Web-first CosyVoice path should live in the new provider module to avoid mixing Windows SAPI and HTTP-service behavior.

## Web Pipeline Behavior

`src/web_app.py` should continue using SSE progress. Add two steps after script generation:

- `封面生成`
- `音频合成`

Script generation failure remains fatal. Media generation failure is non-fatal:

- The failed media step emits an SSE event with status `warning`.
- The final `完成` payload includes a `warnings` array.
- The final payload includes only paths that actually exist.

Example final payload:

```json
{
  "script": "data/scripts/2401.00000_script.md",
  "image": "data/images/2401.00000_cover.png",
  "audio": "data/audio/2401.00000_podcast.wav",
  "warnings": []
}
```

If ComfyUI fails but CosyVoice succeeds:

```json
{
  "script": "data/scripts/2401.00000_script.md",
  "audio": "data/audio/2401.00000_podcast.wav",
  "warnings": ["封面生成失败: connection refused"]
}
```

## Web UI Behavior

The output area should show three result sections:

- Cover image preview when `image` is present.
- Audio player when `audio` is present.
- Script content when `script` is present.

Warnings should be visible near the output area and should not be presented as fatal errors. If image or audio is missing because the provider is disabled, the UI should show a short neutral note rather than an error.

The current Web app uses Chinese labels and a pixel-radio visual style. The media output sections should match that style and avoid introducing a separate design language.

## API Routes

Keep existing routes:

- `/api/run`
- `/api/script`
- `/api/audio`

Add:

- `/api/image`

`/api/image` should mirror `/api/audio`: accept a path query parameter, verify it exists, and return image bytes with a suitable MIME type.

Path handling should not allow arbitrary filesystem reads outside project output directories. If that cannot be fully solved in the first pass, restrict the route to `data/images/` and use resolved-path containment checks.

## Doctor Check

Extend `src/doctor.py` with media checks:

- Report whether image provider is disabled, configured, or missing.
- Report whether voice provider is disabled, configured, or missing.
- Do not require ComfyUI/CosyVoice services to be reachable for a normal OK result unless providers are enabled.

The first version may avoid live network checks to keep `--doctor` fast and deterministic. A future `--doctor --network` style check can probe services explicitly.

## Error Handling

Fatal:

- Invalid input.
- PDF download/conversion failure.
- LLM summary/script failure.
- Generated script missing required sections.

Non-fatal warning:

- ComfyUI unavailable, timed out, or returned invalid image data.
- CosyVoice unavailable, timed out, or returned invalid audio data.
- Media provider disabled.

Warnings should include enough information to fix configuration without exposing secrets.

## Testing Strategy

Use unit tests with fake HTTP clients or injected request functions. Do not require running ComfyUI or CosyVoice in tests.

Coverage should include:

- Media config loads enabled and disabled providers.
- Existing media outputs are skipped unless forced.
- Image generator writes returned bytes to the expected path.
- Voice generator extracts only the script body and writes returned audio bytes.
- Web final payload includes image/audio paths on success.
- Web final payload includes warnings and still includes script when media generation fails.
- `/api/image` refuses missing paths and serves existing image bytes.

Run the full suite after implementation:

```bash
python -m unittest discover -s tests -v
```

## Open Decisions For Implementation

The design intentionally leaves exact ComfyUI and CosyVoice request schemas behind provider modules. During implementation, use the simplest HTTP shape supported by the user's running services, keep it isolated, and document the expected endpoint in README.

If the user's local services use different endpoint paths, the provider modules should make the endpoint configurable without changing Web pipeline code.
