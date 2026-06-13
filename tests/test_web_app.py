import tempfile
import unittest
from pathlib import Path

from src import web_app


class WebAppTests(unittest.TestCase):
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
