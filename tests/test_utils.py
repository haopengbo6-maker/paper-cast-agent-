import tempfile
import unittest
from pathlib import Path

from src.utils import ensure_project_dirs


class UtilsTests(unittest.TestCase):
    def test_ensure_project_dirs_accepts_custom_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            paths = ensure_project_dirs(root)

            expected = [
                root / "data" / "pdfs",
                root / "data" / "markdown",
                root / "data" / "chunks",
                root / "data" / "summaries",
                root / "data" / "scripts",
                root / "data" / "images",
                root / "data" / "audio",
                root / "prompts",
            ]
            self.assertEqual(paths, expected)
            for path in expected:
                self.assertTrue(path.exists(), f"{path} should exist")

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


if __name__ == "__main__":
    unittest.main()
