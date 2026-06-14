import tempfile
import unittest
from pathlib import Path

from src.media_config import VoiceProviderConfig
from src.voice_generator import build_speech_text, generate_voice_audio


class VoiceGeneratorTests(unittest.TestCase):
    def test_build_speech_text_extracts_script_and_strips_markdown(self):
        script = (
            "# 播报脚本\n"
            "**第一句**。[uv_break]\n"
            "[链接](https://example.com)\n\n"
            "# 关键词\n"
            "- Agent\n"
        )

        text = build_speech_text(script)

        self.assertEqual(text, "第一句。\n链接")

    def test_disabled_provider_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_voice_audio(
                "paper",
                "# 播报脚本\n正文",
                Path(tmp),
                VoiceProviderConfig("none", "", 1, "default"),
            )

            self.assertIsNone(result)

    def test_writes_returned_audio_bytes(self):
        calls = []

        def fake_request(url, payload, timeout):
            calls.append((url, payload, timeout))
            return b"RIFFaudio"

        with tempfile.TemporaryDirectory() as tmp:
            path = generate_voice_audio(
                "paper",
                "# 播报脚本\n正文",
                Path(tmp),
                VoiceProviderConfig("cosyvoice", "http://voice", 7, "中文女声"),
                request_audio=fake_request,
                force=True,
            )

            self.assertEqual(path, Path(tmp) / "paper_podcast.wav")
            self.assertEqual(path.read_bytes(), b"RIFFaudio")
            self.assertEqual(calls[0][0], "http://voice/papercast/tts")
            self.assertEqual(calls[0][1]["voice"], "中文女声")

    def test_writes_edge_tts_audio_bytes(self):
        calls = []

        def fake_edge_tts(text, output, voice, timeout):
            calls.append((text, output, voice, timeout))
            output.write_bytes(b"edge-audio")

        with tempfile.TemporaryDirectory() as tmp:
            path = generate_voice_audio(
                "paper",
                "# 鎾姤鑴氭湰\n姝ｆ枃",
                Path(tmp),
                VoiceProviderConfig("edge_tts", "", 11, "zh-CN-XiaoxiaoNeural"),
                request_audio=fake_edge_tts,
                force=True,
            )

            self.assertEqual(path, Path(tmp) / "paper_podcast.mp3")
            self.assertEqual(path.read_bytes(), b"edge-audio")
            self.assertEqual(calls[0][2], "zh-CN-XiaoxiaoNeural")


if __name__ == "__main__":
    unittest.main()
