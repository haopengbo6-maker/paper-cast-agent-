import tempfile
import unittest
from pathlib import Path

from src.media_config import load_media_config


class MediaConfigTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("", encoding="utf-8")

            config = load_media_config(env_path)

            self.assertEqual(config.image.provider, "none")
            self.assertFalse(config.image.enabled)
            self.assertEqual(config.voice.provider, "none")
            self.assertFalse(config.voice.enabled)

    def test_loads_enabled_comfyui_and_cosyvoice(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "MEDIA_IMAGE_PROVIDER=comfyui\n"
                "COMFYUI_BASE_URL=http://127.0.0.1:8188/\n"
                "COMFYUI_TIMEOUT_SECONDS=12\n"
                "MEDIA_VOICE_PROVIDER=cosyvoice\n"
                "COSYVOICE_BASE_URL=http://127.0.0.1:50000/\n"
                "COSYVOICE_TIMEOUT_SECONDS=34\n"
                "COSYVOICE_VOICE=中文女声\n",
                encoding="utf-8",
            )

            config = load_media_config(env_path)

            self.assertTrue(config.image.enabled)
            self.assertEqual(config.image.base_url, "http://127.0.0.1:8188")
            self.assertEqual(config.image.timeout_seconds, 12)
            self.assertTrue(config.voice.enabled)
            self.assertEqual(config.voice.base_url, "http://127.0.0.1:50000")
            self.assertEqual(config.voice.timeout_seconds, 34)
            self.assertEqual(config.voice.voice, "中文女声")


if __name__ == "__main__":
    unittest.main()
