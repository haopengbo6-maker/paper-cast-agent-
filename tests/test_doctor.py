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

    def test_doctor_reports_media_provider_config(self):
        keys = (
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL",
            "MEDIA_IMAGE_PROVIDER",
            "COMFYUI_BASE_URL",
            "MEDIA_VOICE_PROVIDER",
            "COSYVOICE_BASE_URL",
        )
        old_values = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prompts = root / "prompts"
                prompts.mkdir()
                (prompts / "map_prompt.txt").write_text("{chunk}", encoding="utf-8")
                (prompts / "reduce_prompt.txt").write_text("{summaries}", encoding="utf-8")
                (root / ".env").write_text(
                    "LLM_API_KEY=key\nLLM_BASE_URL=https://example.com/v1\nLLM_MODEL=model\n"
                    "MEDIA_IMAGE_PROVIDER=comfyui\nCOMFYUI_BASE_URL=http://127.0.0.1:8188\n"
                    "MEDIA_VOICE_PROVIDER=cosyvoice\nCOSYVOICE_BASE_URL=http://127.0.0.1:50000\n",
                    encoding="utf-8",
                )

                report = run_doctor(root=root, check_optional_imports=False)

                self.assertTrue(report.ok)
                self.assertIn("Image provider configured: comfyui", report.messages)
                self.assertIn("Voice provider configured: cosyvoice", report.messages)
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
