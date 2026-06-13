import unittest

from src.arxiv_client import resolve_paper_input


class ArxivClientTests(unittest.TestCase):
    def test_resolves_arxiv_id_to_pdf_url_and_paper_id(self):
        paper = resolve_paper_input(arxiv_id="2401.00000", pdf_url=None)

        self.assertEqual(paper.paper_id, "2401.00000")
        self.assertEqual(paper.pdf_url, "https://arxiv.org/pdf/2401.00000.pdf")

    def test_resolves_pdf_url_to_stable_paper_id(self):
        paper = resolve_paper_input(
            arxiv_id=None,
            pdf_url="https://arxiv.org/pdf/2401.00000",
        )

        self.assertEqual(paper.paper_id, "2401.00000")
        self.assertEqual(paper.pdf_url, "https://arxiv.org/pdf/2401.00000")

    def test_requires_one_input(self):
        with self.assertRaisesRegex(ValueError, "Provide --arxiv-id or --pdf-url"):
            resolve_paper_input(arxiv_id=None, pdf_url=None)


if __name__ == "__main__":
    unittest.main()
