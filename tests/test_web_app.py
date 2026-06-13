import tempfile
import unittest
from pathlib import Path

from src import web_app


class WebAppTests(unittest.TestCase):
    def test_build_run_options_defaults_to_quality_mode(self):
        options = web_app._build_run_options({})

        self.assertEqual(options.chunk_size, 3000)
        self.assertEqual(options.chunk_overlap, 300)
        self.assertFalse(options.skip_media)
        self.assertEqual(options.summary_max_workers, 1)

    def test_build_run_options_uses_fast_mode(self):
        options = web_app._build_run_options({"fast_mode": "on"})

        self.assertEqual(options.chunk_size, 6000)
        self.assertEqual(options.chunk_overlap, 100)
        self.assertTrue(options.skip_media)
        self.assertGreaterEqual(options.summary_max_workers, 2)

    def test_api_image_serves_image_under_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "data" / "images"
            image_dir.mkdir(parents=True)
            image = image_dir / "cover.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\nimage")

            old_image_dir = web_app.IMAGE_DIR
            web_app.IMAGE_DIR = image_dir
            try:
                client = web_app.app.test_client()
                response = client.get(f"/api/image?path={image}")
            finally:
                web_app.IMAGE_DIR = old_image_dir

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, b"\x89PNG\r\n\x1a\nimage")

    def test_api_image_rejects_path_outside_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside.png"
            outside.write_bytes(b"image")

            client = web_app.app.test_client()
            response = client.get(f"/api/image?path={outside}")

            self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
