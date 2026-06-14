import os
import tempfile
import unittest
from pathlib import Path

from src.media_config import load_media_config


class MediaConfigTests(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "MEDIA_IMAGE_PROVIDER",
                "COMFYUI_BASE_URL",
                "COMFYUI_TIMEOUT_SECONDS",
                "MEDIA_VOICE_PROVIDER",
                "COSYVOICE_BASE_URL",
                "COSYVOICE_TIMEOUT_SECONDS",
                "COSYVOICE_VOICE",
                "EDGE_TTS_VOICE",
            )
        }
        for key in self._old_env:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

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

    def test_loads_edge_tts_without_base_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "MEDIA_VOICE_PROVIDER=edge_tts\n"
                "EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural\n",
                encoding="utf-8",
            )

            config = load_media_config(env_path)

            self.assertTrue(config.voice.enabled)
            self.assertEqual(config.voice.provider, "edge_tts")
            self.assertEqual(config.voice.base_url, "")
            self.assertEqual(config.voice.voice, "zh-CN-XiaoxiaoNeural")


if __name__ == "__main__":
    unittest.main()
