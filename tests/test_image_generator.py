import tempfile
import unittest
from pathlib import Path

from src.image_generator import build_cover_prompt, generate_cover_image
from src.media_config import ImageProviderConfig


class ImageGeneratorTests(unittest.TestCase):
    def test_build_cover_prompt_uses_script_title_and_keywords(self):
        script = (
            "# 播报标题\n"
            "AI 论文收音机\n\n"
            "# 播报脚本\n"
            "今天我们聊一个 agent 系统。\n\n"
            "# 关键词\n"
            "- Agent\n"
            "- Tool Learning\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("AI 论文收音机", prompt)
        self.assertIn("Agent", prompt)
        self.assertIn("podcast cover", prompt)

    def test_disabled_provider_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_cover_image(
                "paper",
                "# 播报标题\n标题",
                Path(tmp),
                ImageProviderConfig("none", "", 1),
            )

            self.assertIsNone(result)

    def test_writes_returned_image_bytes(self):
        calls = []

        def fake_request(url, payload, timeout):
            calls.append((url, payload, timeout))
            return b"\x89PNG\r\n\x1a\nimage"

        with tempfile.TemporaryDirectory() as tmp:
            path = generate_cover_image(
                "paper",
                "# 播报标题\n标题",
                Path(tmp),
                ImageProviderConfig("comfyui", "http://local", 9),
                request_image=fake_request,
                force=True,
            )

            self.assertEqual(path, Path(tmp) / "paper_cover.png")
            self.assertEqual(path.read_bytes(), b"\x89PNG\r\n\x1a\nimage")
            self.assertEqual(calls[0][0], "http://local/papercast/txt2img")
            self.assertEqual(calls[0][2], 9)


if __name__ == "__main__":
    unittest.main()
