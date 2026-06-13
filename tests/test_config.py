import os
import tempfile
import unittest
from pathlib import Path

from src.config import load_dotenv_file, load_llm_config


class ConfigTests(unittest.TestCase):
    def test_load_dotenv_file_sets_missing_values_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "LLM_API_KEY=from-file\nLLM_BASE_URL=https://example.com/v1\nLLM_MODEL=test-model\n",
                encoding="utf-8",
            )
            old_api_key = os.environ.get("LLM_API_KEY")
            try:
                os.environ["LLM_API_KEY"] = "already-set"
                os.environ.pop("LLM_BASE_URL", None)
                os.environ.pop("LLM_MODEL", None)

                load_dotenv_file(env_path)

                self.assertEqual(os.environ["LLM_API_KEY"], "already-set")
                self.assertEqual(os.environ["LLM_BASE_URL"], "https://example.com/v1")
                self.assertEqual(os.environ["LLM_MODEL"], "test-model")
            finally:
                if old_api_key is None:
                    os.environ.pop("LLM_API_KEY", None)
                else:
                    os.environ["LLM_API_KEY"] = old_api_key
                os.environ.pop("LLM_BASE_URL", None)
                os.environ.pop("LLM_MODEL", None)

    def test_load_llm_config_can_read_explicit_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "LLM_API_KEY=key\nLLM_BASE_URL=https://example.com/v1\nLLM_MODEL=model\n",
                encoding="utf-8",
            )
            keys = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
            old_values = {key: os.environ.get(key) for key in keys}
            try:
                for key in keys:
                    os.environ.pop(key, None)

                config = load_llm_config(env_path)

                self.assertEqual(config.api_key, "key")
                self.assertEqual(config.base_url, "https://example.com/v1")
                self.assertEqual(config.model, "model")
            finally:
                for key, value in old_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
