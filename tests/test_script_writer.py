import tempfile
import unittest
from pathlib import Path

from src.script_writer import write_script


class FakeLlm:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, prompt):
        self.calls.append(prompt)
        return self.response


VALID_SCRIPT = """# 播报标题

# 播报脚本
今天我们聊一篇论文。[uv_break]

# 关键词
- Agent

# 适合延伸学习的概念
- Tool Learning
"""


class ScriptWriterTests(unittest.TestCase):
    def test_skips_existing_script_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "script.md"
            script_path.write_text("existing", encoding="utf-8")
            summary_path = Path(tmp) / "summary.md"
            summary_path.write_text("summary", encoding="utf-8")
            llm = FakeLlm(VALID_SCRIPT)

            result = write_script(
                [summary_path],
                script_path=script_path,
                reduce_prompt="Write script: {summaries}",
                llm_client=llm,
                force=False,
            )

            self.assertEqual(result, script_path)
            self.assertEqual(script_path.read_text(encoding="utf-8"), "existing")
            self.assertEqual(llm.calls, [])

    def test_force_regenerates_existing_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "script.md"
            script_path.write_text("old", encoding="utf-8")
            summary_path = Path(tmp) / "summary.md"
            summary_path.write_text("summary", encoding="utf-8")
            llm = FakeLlm(VALID_SCRIPT)

            write_script(
                [summary_path],
                script_path=script_path,
                reduce_prompt="Write script: {summaries}",
                llm_client=llm,
                force=True,
            )

            self.assertEqual(script_path.read_text(encoding="utf-8"), VALID_SCRIPT)
            self.assertEqual(llm.calls, ["Write script: summary"])

    def test_empty_summary_list_fails_before_llm_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLlm(VALID_SCRIPT)

            with self.assertRaisesRegex(RuntimeError, "No summary files"):
                write_script(
                    [],
                    script_path=Path(tmp) / "script.md",
                    reduce_prompt="Write script: {summaries}",
                    llm_client=llm,
                    force=False,
                )

            self.assertEqual(llm.calls, [])

    def test_empty_summary_file_fails_before_llm_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.md"
            summary_path.write_text("   ", encoding="utf-8")
            llm = FakeLlm(VALID_SCRIPT)

            with self.assertRaisesRegex(RuntimeError, "empty summary"):
                write_script(
                    [summary_path],
                    script_path=Path(tmp) / "script.md",
                    reduce_prompt="Write script: {summaries}",
                    llm_client=llm,
                    force=False,
                )

            self.assertEqual(llm.calls, [])

    def test_generated_script_must_contain_required_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.md"
            summary_path.write_text("summary", encoding="utf-8")
            llm = FakeLlm("# 播报标题\nmissing sections")

            with self.assertRaisesRegex(RuntimeError, "missing required section"):
                write_script(
                    [summary_path],
                    script_path=Path(tmp) / "script.md",
                    reduce_prompt="Write script: {summaries}",
                    llm_client=llm,
                    force=False,
                )

            self.assertFalse((Path(tmp) / "script.md").exists())


if __name__ == "__main__":
    unittest.main()
