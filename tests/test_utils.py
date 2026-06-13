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
                root / "prompts",
            ]
            self.assertEqual(paths, expected)
            for path in expected:
                self.assertTrue(path.exists(), f"{path} should exist")


if __name__ == "__main__":
    unittest.main()
