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


class ImageGeneratorTests(unittest.TestCase):
    def test_build_cover_prompt_uses_script_title_and_keywords(self):
        script = (
            "# 播报标题\n"
            "AI 论文收音机\n\n"
            "# 播报脚本\n"
            "今天我们聊一个 agent 系统。\n\n"
            "# 关键词\n"
            "- Agent\n"
            "- Tool Learning\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("AI 论文收音机", prompt)
        self.assertIn("Agent", prompt)
        self.assertIn("podcast cover", prompt)

    def test_build_cover_prompt_keeps_flow_matching_cover_on_topic(self):
        script = (
            "# 播报标题\n"
            "Flow Matching 让生成模型学会抄近道\n\n"
            "# 播报脚本\n"
            "这篇论文讨论 Continuous Normalizing Flows、向量场、最优传输和扩散模型采样。\n\n"
            "# 关键词\n"
            "- Flow Matching\n"
            "- Optimal Transport\n"
            "- vector field\n"
            "- diffusion models\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("Flow Matching", prompt)
        self.assertIn("Optimal Transport", prompt)
        self.assertIn("vector field", prompt)
        self.assertIn("Gaussian noise cloud", prompt)
        self.assertIn("8 to 12 smooth vector arrows", prompt)
        self.assertIn("artistic conceptual scene", prompt)
        self.assertIn("messy line art", prompt)
        self.assertIn("colored pencil sketch", prompt)
        self.assertIn("no visible text", prompt)
        self.assertIn("no Chinese characters", prompt)
        self.assertIn("avoid ancient art", prompt)

    def test_build_cover_prompt_recognizes_flow_machine_topic(self):
        script = (
            "# 播报标题\n"
            "Flow Machine 论文里的生成模型新路线\n\n"
            "# 播报脚本\n"
            "这篇论文讨论概率路径、ODE 轨迹和生成模型采样。\n\n"
            "# 关键词\n"
            "- Flow Machine\n"
            "- generative model\n"
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
                "# 播报标题\n标题",
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
                "# 播报标题\n标题",
                Path(tmp),
                ImageProviderConfig("comfyui", "http://local", 9),
                request_image=fake_request,
                force=True,
            )

            self.assertEqual(path, Path(tmp) / "paper_cover_v6.png")
            self.assertEqual(path.read_bytes(), b"\x89PNG\r\n\x1a\nimage")
            self.assertEqual(calls[0][0], "http://local")
            self.assertEqual(calls[0][2], 9)

    def test_flow_topic_uses_artistic_comfyui_prompt(self):
        calls = []

        def fake_request(url, payload, timeout):
            calls.append(payload)
            return b"\x89PNG\r\n\x1a\nimage"

        script = (
            "# 播报标题\n"
            "Flow Matching 让生成模型学会抄近道\n\n"
            "# 关键词\n"
            "- Flow Matching\n"
            "- vector field\n"
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

            self.assertEqual(path, Path(tmp) / "flow_machine_cover_v6.png")
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("messy line art", calls[0]["prompt"])
            self.assertIn("colored pencil sketch", calls[0]["prompt"])
            self.assertIn("acrylic paint", calls[0]["prompt"])
            self.assertIn("oil-paint texture", calls[0]["prompt"])

    def test_cover_image_path_includes_prompt_version(self):
        self.assertEqual(
            cover_image_path(Path("images"), "paper"),
            Path("images") / "paper_cover_v6.png",
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
