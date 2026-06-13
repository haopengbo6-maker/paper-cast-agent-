import tempfile
import unittest
from pathlib import Path

from src.splitter import Chunk
from src.summarizer import summarize_chunks


class FakeLlm:
    def __init__(self, responses=None, fail_on_call=None):
        self.responses = responses or []
        self.fail_on_call = fail_on_call
        self.calls = []

    def chat(self, prompt):
        self.calls.append(prompt)
        if self.fail_on_call is not None and len(self.calls) >= self.fail_on_call:
            raise RuntimeError("model failed")
        if self.responses:
            return self.responses.pop(0)
        return f"summary {len(self.calls)}"


def make_chunk(chunk_id, text):
    return Chunk(
        text=text,
        metadata={
            "paper_id": "2401.00000",
            "chunk_id": chunk_id,
            "source_file": "data/markdown/2401.00000.md",
            "char_length": len(text),
        },
    )


class SummarizerTests(unittest.TestCase):
    def test_skips_existing_summary_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_dir = Path(tmp)
            existing = summary_dir / "2401.00000_chunk_001.md"
            existing.write_text("existing summary", encoding="utf-8")
            llm = FakeLlm()

            paths = summarize_chunks(
                [make_chunk(1, "chunk text")],
                paper_id="2401.00000",
                summary_dir=summary_dir,
                map_prompt="Summarize: {chunk}",
                llm_client=llm,
                force=False,
            )

            self.assertEqual(paths, [existing])
            self.assertEqual(existing.read_text(encoding="utf-8"), "existing summary")
            self.assertEqual(llm.calls, [])

    def test_force_regenerates_existing_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_dir = Path(tmp)
            existing = summary_dir / "2401.00000_chunk_001.md"
            existing.write_text("old summary", encoding="utf-8")
            llm = FakeLlm(responses=["new summary"])

            summarize_chunks(
                [make_chunk(1, "chunk text")],
                paper_id="2401.00000",
                summary_dir=summary_dir,
                map_prompt="Summarize: {chunk}",
                llm_client=llm,
                force=True,
            )

            self.assertEqual(existing.read_text(encoding="utf-8"), "new summary")
            self.assertEqual(llm.calls, ["Summarize: chunk text"])

    def test_failure_keeps_completed_summaries_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_dir = Path(tmp)
            llm = FakeLlm(responses=["first summary"], fail_on_call=2)

            with self.assertRaisesRegex(RuntimeError, "chunk 2/2"):
                summarize_chunks(
                    [make_chunk(1, "first"), make_chunk(2, "second")],
                    paper_id="2401.00000",
                    summary_dir=summary_dir,
                    map_prompt="Summarize: {chunk}",
                    llm_client=llm,
                    force=False,
                )

            self.assertEqual(
                (summary_dir / "2401.00000_chunk_001.md").read_text(encoding="utf-8"),
                "first summary",
            )
            self.assertFalse((summary_dir / "2401.00000_chunk_002.md").exists())


if __name__ == "__main__":
    unittest.main()
