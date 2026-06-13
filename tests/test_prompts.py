import tempfile
import unittest
from pathlib import Path

from src.prompts import load_prompt


class PromptTests(unittest.TestCase):
    def test_load_prompt_requires_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map_prompt.txt"
            path.write_text("Summarize this text", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, r"must contain \{chunk\}"):
                load_prompt(path, required_placeholder="{chunk}")

    def test_load_prompt_returns_text_when_placeholder_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map_prompt.txt"
            path.write_text("Summarize: {chunk}", encoding="utf-8")

            self.assertEqual(load_prompt(path, required_placeholder="{chunk}"), "Summarize: {chunk}")


if __name__ == "__main__":
    unittest.main()
