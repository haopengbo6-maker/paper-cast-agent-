import os
import tempfile
import unittest
from pathlib import Path

from src.doctor import run_doctor


class DoctorTests(unittest.TestCase):
    def test_doctor_reports_ok_when_required_project_files_and_env_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root / "prompts"
            prompts.mkdir()
            (prompts / "map_prompt.txt").write_text("Map {chunk}", encoding="utf-8")
            (prompts / "reduce_prompt.txt").write_text("Reduce {summaries}", encoding="utf-8")

            old_values = {key: os.environ.get(key) for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")}
            try:
                os.environ["LLM_API_KEY"] = "key"
                os.environ["LLM_BASE_URL"] = "https://example.com/v1"
                os.environ["LLM_MODEL"] = "model"

                report = run_doctor(root=root, check_optional_imports=False)

                self.assertTrue(report.ok)
                self.assertEqual(report.errors, [])
                self.assertIn("data directories", "\n".join(report.messages))
            finally:
                for key, value in old_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_doctor_reports_missing_prompt_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root / "prompts"
            prompts.mkdir()
            (prompts / "map_prompt.txt").write_text("Map missing placeholder", encoding="utf-8")
            (prompts / "reduce_prompt.txt").write_text("Reduce {summaries}", encoding="utf-8")

            report = run_doctor(root=root, check_optional_imports=False)

            self.assertFalse(report.ok)
            self.assertTrue(any("{chunk}" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
