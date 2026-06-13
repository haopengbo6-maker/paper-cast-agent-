from __future__ import annotations

from pathlib import Path


def download_pdf(
    pdf_url: str,
    target_path: Path,
    force: bool = False,
    session=None,
    timeout: int = 30,
) -> Path:
    if target_path.exists() and not force:
        print(f"PDF exists, skipping: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    http = session
    if http is None:
        import requests

        http = requests.Session()

    try:
        response = http.get(pdf_url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        status = getattr(locals().get("response", None), "status_code", "unknown")
        raise RuntimeError(
            f"Failed to download PDF from {pdf_url}: status={status}, reason={exc}"
        ) from exc

    target_path.write_bytes(response.content)
    return target_path
