import tempfile
import unittest
from pathlib import Path

from src import web_app


class WebAppTests(unittest.TestCase):
    def test_index_renders_fast_and_full_mode_choices(self):
        client = web_app.app.test_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn('value="fast" checked', text)
        self.assertIn('value="full"', text)

    def test_index_renders_radio_blog_result_view(self):
        client = web_app.app.test_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("broadcast-post", text)
        self.assertIn("player-layout", text)
        self.assertIn("result-title", text)
        self.assertIn("audio-player", text)
        self.assertIn("cover-image", text)
        self.assertIn("scrollToOutput()", text)

    def test_index_warns_when_opened_as_file(self):
        client = web_app.app.test_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("window.location.protocol === 'file:'", text)
        self.assertIn("http://localhost:5000/", text)

    def test_build_run_options_defaults_to_quality_mode(self):
        options = web_app._build_run_options({})

        self.assertEqual(options.output_tag, "full")
        self.assertEqual(options.chunk_size, 3000)
        self.assertEqual(options.chunk_overlap, 300)
        self.assertFalse(options.skip_media)
        self.assertGreaterEqual(options.summary_max_workers, 2)

    def test_build_run_options_uses_fast_mode(self):
        options = web_app._build_run_options({"fast_mode": "on"})

        self.assertEqual(options.output_tag, "fast")
        self.assertEqual(options.chunk_size, 6000)
        self.assertEqual(options.chunk_overlap, 100)
        self.assertTrue(options.skip_media)
        self.assertGreaterEqual(options.summary_max_workers, 2)

    def test_run_options_use_separate_cache_keys_for_modes(self):
        fast = web_app._build_run_options({"fast_mode": "on"})
        full = web_app._build_run_options({})

        self.assertNotEqual(
            web_app._paper_output_id("2401.00000", fast),
            web_app._paper_output_id("2401.00000", full),
        )

    def test_media_disabled_warning_is_recorded(self):
        warnings = []

        web_app._record_warning(warnings, "media disabled")

        self.assertEqual(warnings, ["media disabled"])

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
