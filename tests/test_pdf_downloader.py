import tempfile
import unittest
from pathlib import Path

from src.pdf_downloader import download_pdf


class FakeResponse:
    def __init__(self, status_code=200, content=b"%PDF-1.4"):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, response=None):
        self.calls = 0
        self.response = response or FakeResponse()

    def get(self, url, timeout):
        self.calls += 1
        return self.response


class PdfDownloaderTests(unittest.TestCase):
    def test_skips_existing_pdf_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "paper.pdf"
            target.write_bytes(b"existing")
            session = FakeSession()

            result = download_pdf("https://example.com/paper.pdf", target, force=False, session=session)

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertEqual(session.calls, 0)

    def test_downloads_pdf_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "paper.pdf"
            session = FakeSession()

            result = download_pdf("https://example.com/paper.pdf", target, force=False, session=session)

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"%PDF-1.4")
            self.assertEqual(session.calls, 1)

    def test_force_redownloads_existing_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "paper.pdf"
            target.write_bytes(b"old")
            session = FakeSession(FakeResponse(content=b"new-pdf"))

            download_pdf("https://example.com/paper.pdf", target, force=True, session=session)

            self.assertEqual(target.read_bytes(), b"new-pdf")
            self.assertEqual(session.calls, 1)

    def test_download_error_includes_url_status_and_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "paper.pdf"
            session = FakeSession(FakeResponse(status_code=503, content=b"unavailable"))

            with self.assertRaisesRegex(
                RuntimeError,
                "Failed to download PDF from https://example.com/paper.pdf.*status=503",
            ):
                download_pdf("https://example.com/paper.pdf", target, force=False, session=session)


if __name__ == "__main__":
    unittest.main()
