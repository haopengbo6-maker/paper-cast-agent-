import tempfile
import unittest
from pathlib import Path

from src.markdown_converter import convert_pdf_to_markdown


class FakeConverter:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def convert(self, pdf_path):
        self.calls += 1
        return self.text


class FakeResult:
    def __init__(self, text_content):
        self.text_content = text_content


class MarkdownConverterTests(unittest.TestCase):
    def test_skips_existing_markdown_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "paper.pdf"
            markdown_path = Path(tmp) / "paper.md"
            pdf_path.write_bytes(b"%PDF")
            markdown_path.write_text("existing", encoding="utf-8")
            converter = FakeConverter("new")

            result = convert_pdf_to_markdown(pdf_path, markdown_path, force=False, converter=converter)

            self.assertEqual(result, markdown_path)
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), "existing")
            self.assertEqual(converter.calls, 0)

    def test_converts_pdf_to_markdown_with_injected_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "paper.pdf"
            markdown_path = Path(tmp) / "paper.md"
            pdf_path.write_bytes(b"%PDF")
            converter = FakeConverter(FakeResult("# Paper\n\ncontent"))

            result = convert_pdf_to_markdown(pdf_path, markdown_path, force=False, converter=converter)

            self.assertEqual(result, markdown_path)
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), "# Paper\n\ncontent")
            self.assertEqual(converter.calls, 1)

    def test_empty_markdown_output_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "paper.pdf"
            markdown_path = Path(tmp) / "paper.md"
            pdf_path.write_bytes(b"%PDF")
            converter = FakeConverter(FakeResult("   "))

            with self.assertRaisesRegex(RuntimeError, "empty Markdown"):
                convert_pdf_to_markdown(pdf_path, markdown_path, force=False, converter=converter)


if __name__ == "__main__":
    unittest.main()
