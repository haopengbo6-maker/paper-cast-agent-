import subprocess
import sys
import unittest


class MainCliTests(unittest.TestCase):
    def test_help_command_runs(self):
        result = subprocess.run(
            [sys.executable, "src/main.py", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--arxiv-id", result.stdout)
        self.assertIn("--pdf-url", result.stdout)

    def test_cli_requires_arxiv_id_or_pdf_url(self):
        result = subprocess.run(
            [sys.executable, "src/main.py"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Provide --arxiv-id or --pdf-url", result.stderr)

    def test_version_command_runs(self):
        result = subprocess.run(
            [sys.executable, "src/main.py", "--version"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("PaperCast Agent", result.stdout)

    def test_doctor_command_runs(self):
        result = subprocess.run(
            [sys.executable, "src/main.py", "--doctor"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertIn("PaperCast Agent doctor", result.stdout)

    def test_cli_rejects_both_input_modes_together(self):
        result = subprocess.run(
            [
                sys.executable,
                "src/main.py",
                "--arxiv-id",
                "2401.00000",
                "--pdf-url",
                "https://arxiv.org/pdf/2401.00000",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Use either --arxiv-id or --pdf-url", result.stderr)

    def test_cli_validates_chunk_overlap(self):
        result = subprocess.run(
            [
                sys.executable,
                "src/main.py",
                "--arxiv-id",
                "2401.00000",
                "--chunk-size",
                "100",
                "--chunk-overlap",
                "100",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--chunk-overlap must be smaller than --chunk-size", result.stderr)


if __name__ == "__main__":
    unittest.main()
