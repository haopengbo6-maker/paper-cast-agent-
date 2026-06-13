import unittest
import tempfile
from pathlib import Path

from src.splitter import export_chunks, format_chunk_previews, split_markdown


class SplitterTests(unittest.TestCase):
    def test_splits_markdown_with_metadata(self):
        text = "# Title\n\n" + ("This is a sentence about agents. " * 40)

        chunks = split_markdown(
            text,
            paper_id="2401.00000",
            source_file="data/markdown/2401.00000.md",
            chunk_size=120,
            chunk_overlap=20,
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["paper_id"], "2401.00000")
        self.assertEqual(chunks[0].metadata["chunk_id"], 1)
        self.assertEqual(chunks[0].metadata["source_file"], "data/markdown/2401.00000.md")
        self.assertEqual(chunks[0].metadata["char_length"], len(chunks[0].text))

    def test_formats_first_two_chunk_previews(self):
        text = "# Title\n\n" + ("This is a sentence about agents. " * 20)
        chunks = split_markdown(
            text,
            paper_id="2401.00000",
            source_file="data/markdown/2401.00000.md",
            chunk_size=100,
            chunk_overlap=10,
        )

        previews = format_chunk_previews(chunks, limit=2, preview_length=60)

        self.assertEqual(len(previews), 2)
        self.assertIn("Chunk 1", previews[0])
        self.assertIn("Chunk 2", previews[1])

    def test_exports_chunk_files_and_metadata_jsonl(self):
        text = "# Title\n\n" + ("This is a sentence about agents. " * 20)
        chunks = split_markdown(
            text,
            paper_id="2401.00000",
            source_file="data/markdown/2401.00000.md",
            chunk_size=100,
            chunk_overlap=10,
        )

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp)
            paths = export_chunks(chunks, export_dir, force=False)

            self.assertEqual(len(paths.chunk_files), len(chunks))
            self.assertTrue(paths.metadata_file.exists())
            self.assertTrue((export_dir / "2401.00000_chunk_001.md").exists())
            metadata_text = paths.metadata_file.read_text(encoding="utf-8")
            self.assertIn('"chunk_id": 1', metadata_text)
            self.assertIn('"paper_id": "2401.00000"', metadata_text)


if __name__ == "__main__":
    unittest.main()
