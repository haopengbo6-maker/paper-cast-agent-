from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


URL = "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
OUT = Path(
    r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\checkpoints\sd_xl_base_1.0.safetensors"
)
TOTAL = 6_938_078_334
CHUNK_SIZE = 1024 * 1024


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 21):
        current = OUT.stat().st_size if OUT.exists() else 0
        if current >= TOTAL:
            print(f"done {current}/{TOTAL}")
            return 0

        print(f"attempt {attempt}: resume from {current}/{TOTAL}")
        headers = {"Range": f"bytes={current}-", "User-Agent": "PaperCastDownloader/1.0"}
        req = Request(URL, headers=headers)
        try:
            with urlopen(req, timeout=60) as response, OUT.open("ab") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    current += len(chunk)
                    if current % (128 * CHUNK_SIZE) < CHUNK_SIZE:
                        pct = current * 100 / TOTAL
                        print(f"{current}/{TOTAL} ({pct:.1f}%)", flush=True)
        except (TimeoutError, URLError, OSError) as exc:
            print(f"download interrupted: {exc}")
            time.sleep(10)
            continue

    current = OUT.stat().st_size if OUT.exists() else 0
    print(f"incomplete {current}/{TOTAL}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
