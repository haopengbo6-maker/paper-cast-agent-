import tempfile
import unittest
from pathlib import Path

from src.image_generator import (
    build_cover_prompt,
    build_sdxl_workflow,
    cover_image_path,
    generate_cover_image,
    select_comfyui_checkpoint,
)
from src.media_config import ImageProviderConfig


TITLE = "# \u64ad\u62a5\u6807\u9898\n"
SCRIPT = "# \u64ad\u62a5\u811a\u672c\n"
KEYWORDS = "# \u5173\u952e\u8bcd\n"


class ImageGeneratorTests(unittest.TestCase):
    def test_build_cover_prompt_uses_script_title_and_keywords(self):
        script = (
            TITLE
            + "AI \u8bba\u6587\u6536\u97f3\u673a\n\n"
            + SCRIPT
            + "\u4eca\u5929\u6211\u4eec\u804a\u4e00\u4e2a agent \u7cfb\u7edf\u3002\n\n"
            + KEYWORDS
            + "- Agent\n"
            + "- Tool Learning\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("AI \u8bba\u6587\u6536\u97f3\u673a", prompt)
        self.assertIn("Agent", prompt)
        self.assertIn("research plate cover", prompt)

    def test_build_cover_prompt_uses_summary_hint(self):
        script = (
            TITLE
            + "Flow Matching \u8ba9\u751f\u6210\u6a21\u578b\u5b66\u4f1a\u6284\u8fd1\u9053\n\n"
            + KEYWORDS
            + "- Flow Matching\n"
        )

        prompt = build_cover_prompt(script, "\u628a\u566a\u58f0\u5206\u5e03\u6cbf\u6700\u77ed\u8def\u5f84\u642c\u5230\u6570\u636e\u5206\u5e03\u3002")

        self.assertIn("summary cue", prompt)
        self.assertIn("\u566a\u58f0\u5206\u5e03", prompt)
        self.assertIn("data distribution manifold", prompt)

    def test_build_cover_prompt_uses_script_hint(self):
        script = (
            TITLE
            + "\u9898\u76ee\u770b\u8d77\u6765\u5f88\u6cdb\u5316\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u5176\u5b9e\u8ba8\u8bba\u4eba\u5f62\u673a\u5668\u4eba\u7684\u53cc\u8db3\u884c\u8d70\u548c\u5168\u8eab\u63a7\u5236\u3002\n"
            + KEYWORDS
            + "- embodiment\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("script cue", prompt)
        self.assertIn("\u4eba\u5f62\u673a\u5668\u4eba", prompt)
        self.assertIn("actuator joints", prompt)

    def test_build_cover_prompt_keeps_flow_matching_cover_on_topic(self):
        script = (
            TITLE
            + "Flow Matching \u8ba9\u751f\u6210\u6a21\u578b\u5b66\u4f1a\u6284\u8fd1\u9053\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u8ba8\u8bba Continuous Normalizing Flows\u3001\u5411\u91cf\u573a\u3001\u6700\u4f18\u4f20\u8f93\u548c\u6269\u6563\u6a21\u578b\u91c7\u6837\u3002\n"
            + KEYWORDS
            + "- Flow Matching\n"
            + "- Optimal Transport\n"
            + "- vector field\n"
            + "- diffusion models\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("Flow Matching", prompt)
        self.assertIn("Optimal Transport", prompt)
        self.assertIn("vector field", prompt)
        self.assertIn("Gaussian noise cloud", prompt)
        self.assertIn("8 to 12 smooth vector arrows", prompt)
        self.assertIn("meaningful conceptual scene", prompt)
        self.assertIn("messy line art", prompt)
        self.assertIn("colored pencil sketch", prompt)
        self.assertIn("no visible text", prompt)
        self.assertIn("no Chinese characters", prompt)
        self.assertIn("avoid ancient art", prompt)

    def test_build_cover_prompt_recognizes_flow_machine_topic(self):
        script = (
            TITLE
            + "Flow Machine \u8bba\u6587\u91cc\u7684\u751f\u6210\u6a21\u578b\u65b0\u8def\u7ebf\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u8ba8\u8bba\u6982\u7387\u8def\u5f84\u3001ODE \u8f68\u8ff9\u548c\u751f\u6210\u6a21\u578b\u91c7\u6837\u3002\n"
            + KEYWORDS
            + "- Flow Machine\n"
            + "- generative model\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("Flow Machine", prompt)
        self.assertIn("Gaussian noise cloud", prompt)
        self.assertIn("data distribution manifold", prompt)
        self.assertIn("avoid generic scenery", prompt)

    def test_disabled_provider_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_cover_image(
                "paper",
                TITLE + "\u6807\u9898",
                Path(tmp),
                ImageProviderConfig("none", "", 1),
            )

            self.assertIsNone(result)

    def test_writes_returned_image_bytes(self):
        calls = []

        def fake_request(url, payload, timeout):
            calls.append((url, payload, timeout))
            return b"\x89PNG\r\n\x1a\nimage"

        with tempfile.TemporaryDirectory() as tmp:
            path = generate_cover_image(
                "paper",
                TITLE + "\u6807\u9898",
                Path(tmp),
                ImageProviderConfig("comfyui", "http://local", 9),
                request_image=fake_request,
                force=True,
            )

            self.assertEqual(path, Path(tmp) / "paper_cover_v11.png")
            self.assertEqual(path.read_bytes(), b"\x89PNG\r\n\x1a\nimage")
            self.assertEqual(calls[0][0], "http://local")
            self.assertEqual(calls[0][2], 9)

    def test_flow_topic_uses_artistic_comfyui_prompt(self):
        calls = []

        def fake_request(url, payload, timeout):
            calls.append(payload)
            return b"\x89PNG\r\n\x1a\nimage"

        script = (
            TITLE
            + "Flow Matching \u8ba9\u751f\u6210\u6a21\u578b\u5b66\u4f1a\u6284\u8fd1\u9053\n\n"
            + KEYWORDS
            + "- Flow Matching\n"
            + "- vector field\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = generate_cover_image(
                "flow_machine",
                script,
                Path(tmp),
                ImageProviderConfig("comfyui", "http://local", 9),
                request_image=fake_request,
                force=True,
            )

            self.assertEqual(path, Path(tmp) / "flow_machine_cover_v11.png")
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("messy line art", calls[0]["prompt"])
            self.assertIn("colored pencil sketch", calls[0]["prompt"])
            self.assertIn("acrylic gouache blocks", calls[0]["prompt"])
            self.assertIn("oil-paint texture", calls[0]["prompt"])
            self.assertIn("letterpress impression", calls[0]["prompt"])
            self.assertIn("binding gutter", calls[0]["prompt"])

    def test_cover_image_path_includes_prompt_version(self):
        self.assertEqual(
            cover_image_path(Path("images"), "paper"),
            Path("images") / "paper_cover_v11.png",
        )

    def test_selects_first_available_comfyui_checkpoint(self):
        object_info = {
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": [["model-a.safetensors", "model-b.safetensors"], {}]
                    }
                }
            }
        }

        self.assertEqual(select_comfyui_checkpoint(object_info), "model-a.safetensors")

    def test_comfyui_checkpoint_selection_fails_clearly_when_empty(self):
        object_info = {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [[], {}]}}
            }
        }

        with self.assertRaisesRegex(RuntimeError, "No ComfyUI checkpoint"):
            select_comfyui_checkpoint(object_info)

    def test_build_cover_prompt_handles_physics_topic(self):
        script = (
            TITLE
            + "\u91cf\u5b50\u53d8\u5316\u4e0e\u5149\u5b66\u573a\u5206\u6790\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u6587\u7ae0\u8ba8\u8bba\u6ce2\u51fd\u6570\u3001\u76f8\u4e92\u4f5c\u7528\u548c\u5b9e\u9a8c\u66f2\u7ebf\u3002\n"
            + KEYWORDS
            + "- quantum\n"
            + "- optics\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("physics primitives", prompt)
        self.assertIn("wave fronts", prompt)
        self.assertIn("equation fragments", prompt)

    def test_build_cover_prompt_handles_social_science_topic(self):
        script = (
            TITLE
            + "\u793e\u4f1a\u884c\u4e3a\u4e0e\u653f\u7b56\u5f71\u54cd\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u6587\u7ae0\u8ba8\u8bba\u8c03\u67e5\u3001\u57fa\u4e8e\u4eba\u7fa4\u7684\u7edf\u8ba1\u548c\u653f\u7b56\u53cd\u9988\u3002\n"
            + KEYWORDS
            + "- sociology\n"
            + "- survey\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("social-science evidence", prompt)
        self.assertIn("timeline bands", prompt)
        self.assertIn("policy feedback arrows", prompt)

    def test_build_cover_prompt_handles_chinese_medical_topic(self):
        script = (
            TITLE
            + "\u9898\u76ee\u7565\u5199\u5f97\u50cf\u666e\u901a\u7814\u7a76\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u5173\u6ce8\u533b\u5b66\u75c5\u7406\u5207\u7247\u548c\u4e34\u5e8a\u8bca\u65ad\u6d41\u7a0b\u3002\n"
            + KEYWORDS
            + "- representation learning\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("medical research evidence", prompt)
        self.assertIn("diagnostic overlay contours", prompt)
        self.assertIn("organ or tissue silhouette", prompt)

    def test_build_cover_prompt_handles_chinese_materials_topic(self):
        script = (
            TITLE
            + "\u65b0\u578b\u6750\u6599\u4e0e\u7535\u6c60\u754c\u9762\u7814\u7a76\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u5206\u6790\u534a\u5bfc\u4f53\u6676\u4f53\u3001\u7535\u6781\u8868\u9762\u548c\u5fae\u7ed3\u6784\u3002\n"
            + KEYWORDS
            + "- interface\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("materials-science objects", prompt)
        self.assertIn("crystal lattice unit cells", prompt)
        self.assertIn("grain boundaries", prompt)

    def test_build_cover_prompt_handles_math_topic(self):
        script = (
            TITLE
            + "\u7b49\u5f0f\u4e0e\u8bc1\u660e\u7684\u7ed3\u6784\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u8bae\u8bba\u7ebf\u6027\u4ee3\u6570\u3001\u5750\u6807\u51e0\u4f55\u548c\u5b9a\u7406\u53d1\u73b0\u3002\n"
            + KEYWORDS
            + "- theorem\n"
            + "- proof\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("mathematical objects", prompt)
        self.assertIn("theorem-proof blocks", prompt)
        self.assertIn("matrix grid", prompt)

    def test_build_cover_prompt_handles_law_topic(self):
        script = (
            TITLE
            + "\u6cd5\u5f8b\u6587\u4e66\u4e0e\u89c4\u7ae0\u5206\u6790\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u5173\u6ce8\u6cd5\u5f8b\u89c4\u5236\u3001\u5408\u89c4\u4e0e\u53f8\u6cd5\u6848\u4f8b\u3002\n"
            + KEYWORDS
            + "- regulation\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("legal-research evidence", prompt)
        self.assertIn("balance-scale silhouette", prompt)
        self.assertIn("citation bands", prompt)

    def test_build_cover_prompt_handles_energy_topic(self):
        script = (
            TITLE
            + "\u80fd\u6e90\u7cfb\u7edf\u4e0e\u7535\u7f51\u8c03\u5ea6\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u6587\u7ae0\u8ba8\u8bba\u5149\u4f0f\u548c\u98ce\u529b\u7684\u80fd\u6e90\u80fd\u91cf\u6d41\u52a8\u3002\n"
            + KEYWORDS
            + "- solar\n"
            + "- grid\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("energy-system components", prompt)
        self.assertIn("power-flow arrows", prompt)
        self.assertIn("battery/storage block", prompt)

    def test_build_cover_prompt_accepts_english_headings(self):
        script = (
            "# title\n"
            + "Generic paper title\n\n"
            + "# script\n"
            + "This paper studies climate remote sensing, ocean change, and carbon monitoring.\n\n"
            + "# keywords\n"
            + "- earth observation\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("Generic paper title", prompt)
        self.assertIn("earth observation", prompt)
        self.assertIn("earth-system evidence", prompt)
        self.assertIn("topographic bands", prompt)

    def test_build_cover_prompt_prevents_robot_skeleton_mismatch(self):
        script = (
            TITLE
            + "\u4eba\u5f62\u673a\u5668\u4eba\u7684\u5168\u8eab\u63a7\u5236\u7814\u7a76\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u8ba8\u8bba\u4eba\u5f62\u673a\u5668\u4eba\u7684\u53cc\u8db3\u884c\u8d70\u3001\u4f3a\u670d\u5173\u8282\u548c\u5168\u8eab\u5e73\u8861\u63a7\u5236\u3002\n"
            + KEYWORDS
            + "- humanoid robot\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("clearly mechanical humanoid robot", prompt)
        self.assertIn("servo motors", prompt)
        self.assertIn("not a skeleton", prompt)
        self.assertIn("avoid bones", prompt)

    def test_build_cover_prompt_handles_geomagnetic_storm_topic(self):
        script = (
            TITLE
            + "\u5730\u78c1\u66b4\u4e0e\u7a7a\u95f4\u5929\u6c14\u9884\u6d4b\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u5206\u6790\u592a\u9633\u98ce\u3001\u78c1\u5c42\u6270\u52a8\u3001\u7535\u79bb\u5c42\u54cd\u5e94\u548c\u6781\u5149\u6d3b\u52a8\u3002\n"
            + KEYWORDS
            + "- geomagnetic storm\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("space-weather science plate", prompt)
        self.assertIn("solar wind", prompt)
        self.assertIn("magnetosphere", prompt)
        self.assertIn("no houses", prompt)
        self.assertIn("no buildings", prompt)

    def test_build_cover_prompt_discourages_house_for_agriculture(self):
        script = (
            TITLE
            + "\u519c\u4e1a\u4f5c\u7269\u4ea7\u91cf\u4e0e\u571f\u58e4\u4f20\u611f\u7814\u7a76\n\n"
            + SCRIPT
            + "\u8fd9\u7bc7\u8bba\u6587\u7814\u7a76\u4f5c\u7269\u3001\u571f\u58e4\u5206\u5c42\u3001\u704c\u6e89\u4e0e\u4ea7\u91cf\u9884\u6d4b\u3002\n"
            + KEYWORDS
            + "- agriculture\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("crop-row geometry", prompt)
        self.assertIn("soil-layer cross-section", prompt)
        self.assertIn("must not show farmhouse", prompt)
        self.assertIn("rural landscape painting", prompt)

    def test_negative_prompt_blocks_common_mismatches(self):
        workflow = build_sdxl_workflow(
            checkpoint="sd_xl_base_1.0.safetensors",
            positive_prompt="cover",
            output_prefix="out",
        )
        negative = workflow["7"]["inputs"]["text"]

        self.assertIn("human skeleton", negative)
        self.assertIn("house", negative)
        self.assertIn("building", negative)
        self.assertIn("residential architecture", negative)

    def test_build_sdxl_workflow_uses_checkpoint_and_prompt(self):
        workflow = build_sdxl_workflow(
            checkpoint="sd_xl_base_1.0.safetensors",
            positive_prompt="podcast cover",
            output_prefix="papercast_cover",
        )

        self.assertEqual(workflow["4"]["inputs"]["ckpt_name"], "sd_xl_base_1.0.safetensors")
        self.assertIn("podcast cover", workflow["6"]["inputs"]["text"])
        self.assertIn("Chinese calligraphy", workflow["7"]["inputs"]["text"])
        self.assertIn("landscape painting", workflow["7"]["inputs"]["text"])
        self.assertIn("Chinese characters", workflow["7"]["inputs"]["text"])
        self.assertIn("pagoda", workflow["7"]["inputs"]["text"])
        self.assertIn("device interface", workflow["7"]["inputs"]["text"])
        self.assertIn("cassette", workflow["7"]["inputs"]["text"])
        self.assertIn("many spheres", workflow["7"]["inputs"]["text"])
        self.assertIn("maze", workflow["7"]["inputs"]["text"])
        self.assertIn("meaningless abstraction", workflow["7"]["inputs"]["text"])
        self.assertEqual(workflow["8"]["inputs"]["steps"], 32)
        self.assertEqual(workflow["8"]["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertEqual(workflow["8"]["inputs"]["scheduler"], "karras")
        self.assertEqual(workflow["10"]["inputs"]["filename_prefix"], "papercast_cover")


if __name__ == "__main__":
    unittest.main()
