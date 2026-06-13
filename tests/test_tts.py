import unittest

from src.tts import _extract_script_body


class TtsTests(unittest.TestCase):
    def test_extracts_only_podcast_script_section(self):
        text = """# 播报标题
标题

# 播报脚本
第一句。[uv_break]
第二句。

# 关键词
- Agent

# 适合延伸学习的概念
- Tool Learning
"""

        self.assertEqual(_extract_script_body(text), "第一句。[uv_break]\n第二句。")


if __name__ == "__main__":
    unittest.main()
