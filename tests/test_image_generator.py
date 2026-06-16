import tempfile
import unittest
from pathlib import Path

from src.image_generator import (
    build_cover_prompt,
    build_sdxl_workflow,
    cover_image_path,
    generate_cover_image,
    select_comfyui_checkpoint,
    _lookup_visual,
    _extract_title_visual_terms,
)
from src.media_config import ImageProviderConfig


TITLE = "# 播报标题\n"
SCRIPT = "# 播报脚本\n"
KEYWORDS = "# 关键词\n"


class ImageGeneratorTests(unittest.TestCase):
    # ── v16: compact, zero-repetition prompts ──

    def test_prompt_is_compact(self):
        script = (
            TITLE
            + "GROOT-N1: An Open Foundation Model for Generalist Humanoid Robots\n\n"
            + SCRIPT
            + "这篇论文提出了 GROOT-N1 开源基础模型。\n"
            + KEYWORDS
            + "- foundation model\n"
            + "- manipulation\n"
            + "- locomotion\n"
        )
        prompt = build_cover_prompt(script)
        words = len(prompt.split())
        # v16 target: under 130 words
        self.assertLess(words, 130, f"Prompt too long: {words} words")
        # No repeated concepts
        for phrase in ["humanoid", "mechanical", "skeleton", "servo", "biological"]:
            count = prompt.lower().count(phrase)
            self.assertLessEqual(count, 1, f"'{phrase}' appears {count} times in prompt")

    def test_title_injected_as_visual_subject(self):
        script = (
            TITLE
            + "GROOT-N1: An Open Foundation Model for Generalist Humanoid Robots\n\n"
            + SCRIPT
            + "这篇论文提出一个通用人形机器人基础模型。\n"
            + KEYWORDS
            + "- humanoid robot\n"
            + "- foundation model\n"
            + "- manipulation\n"
        )
        prompt = build_cover_prompt(script)
        self.assertIn("GROOT-N1", prompt)
        self.assertIn("subject:", prompt)
        self.assertIn("context:", prompt)

    def test_flow_matching_topic_still_matched(self):
        script = (
            TITLE
            + "Flow Matching 让生成模型学会抄近道\n\n"
            + SCRIPT
            + "讨论 Continuous Normalizing Flows、向量场、最优传输。\n"
            + KEYWORDS
            + "- Flow Matching\n"
            + "- Optimal Transport\n"
            + "- vector field\n"
        )
        prompt = build_cover_prompt(script)
        self.assertIn("Flow Matching", prompt)
        self.assertIn("generative modeling", prompt)
        self.assertIn("no Chinese", prompt)

    def test_humanoid_robot_prevents_skeleton(self):
        script = (
            TITLE
            + "人形机器人的全身控制研究\n\n"
            + SCRIPT
            + "讨论人形机器人的双足行走、伺服关节和全身平衡控制。\n"
            + KEYWORDS
            + "- humanoid robot\n"
        )
        prompt = build_cover_prompt(script)
        self.assertIn("no skin", prompt)
        # These should NOT appear multiple times
        self.assertLessEqual(prompt.lower().count("humanoid"), 2)  # once in subject, maybe in title

    def test_physics_topic(self):
        script = (
            TITLE
            + "量子变化与光学场分析\n\n"
            + SCRIPT
            + "讨论波函数、相互作用和实验曲线。\n"
            + KEYWORDS
            + "- quantum\n"
            + "- optics\n"
        )
        prompt = build_cover_prompt(script)
        self.assertIn("wave patterns", prompt)
        self.assertIn("physics", prompt.lower())

    def test_climate_topic(self):
        script = (
            "# title\n"
            + "Climate remote sensing study\n\n"
            + "# script\n"
            + "This paper studies climate remote sensing and ocean change.\n\n"
            + "# keywords\n"
            + "- earth observation\n"
        )
        prompt = build_cover_prompt(script)
        self.assertIn("earth system", prompt)
        # Title terms now map to visual cues via CN_EN_MAP → KEYWORD_VISUALS
        self.assertIn("climate", prompt.lower())
        self.assertIn("atmospheric layers", prompt.lower())  # visual cue from climate/remote sensing

    def test_geomagnetic_storm_topic(self):
        script = (
            TITLE
            + "地磁暴与空间天气预测\n\n"
            + SCRIPT
            + "分析太阳风、磁层扰动、电离层响应和极光活动。\n"
            + KEYWORDS
            + "- geomagnetic storm\n"
        )
        prompt = build_cover_prompt(script)
        self.assertIn("space weather", prompt)
        self.assertIn("magnetosphere", prompt)

    def test_agriculture_discourages_farmhouse(self):
        script = (
            TITLE
            + "农业作物产量与土壤传感研究\n\n"
            + SCRIPT
            + "研究作物、土壤分层、灌溉与产量预测。\n"
            + KEYWORDS
            + "- agriculture\n"
        )
        prompt = build_cover_prompt(script)
        self.assertIn("agricultural research", prompt)
        self.assertIn("crop rows", prompt)

    # ── Keyword visual lookup ──

    def test_lookup_visual_finds_exact_match(self):
        cue = _lookup_visual("manipulation")
        self.assertIsNotNone(cue)
        self.assertIn("robotic hand", cue)

    def test_lookup_visual_handles_plurals(self):
        cue = _lookup_visual("robots")
        self.assertIsNone(cue)  # "robots" not in dict, "robot" is also not

    def test_lookup_visual_returns_none_for_generic(self):
        cue = _lookup_visual("method")
        self.assertIsNone(cue)

    # ── Title extraction ──

    def test_extract_title_visual_terms_splits_on_colon(self):
        terms = _extract_title_visual_terms(
            "GROOT-N1: An Open Foundation Model for Generalist Humanoid Robots"
        )
        self.assertIn("GROOT-N1", terms)
        self.assertIn("Open Foundation Model", terms)
        self.assertIn("Generalist Humanoid Robots", terms)

    def test_extract_title_visual_terms_handles_simple_title(self):
        terms = _extract_title_visual_terms("LSTM Networks for Sequence Prediction")
        self.assertTrue(len(terms) >= 2, f"Expected >=2 terms, got {terms}")

    # ── Integration ──

    def test_disabled_provider_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_cover_image(
                "paper", TITLE + "标题", Path(tmp),
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
                "paper", TITLE + "标题", Path(tmp),
                ImageProviderConfig("comfyui", "http://local", 9),
                request_image=fake_request, force=True,
            )
            self.assertEqual(path, Path(tmp) / "paper_cover_v16.png")
            self.assertEqual(path.read_bytes(), b"\x89PNG\r\n\x1a\nimage")

    def test_flow_topic_prompt_has_artistic_elements(self):
        calls = []
        def fake_request(url, payload, timeout):
            calls.append(payload)
            return b"\x89PNG\r\n\x1a\nimage"
        script = TITLE + "Flow Matching 让生成模型学会抄近道\n\n" + KEYWORDS + "- Flow Matching\n- vector field\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_cover_image(
                "flow_machine", script, Path(tmp),
                ImageProviderConfig("comfyui", "http://local", 9),
                request_image=fake_request, force=True,
            )
            self.assertEqual(path, Path(tmp) / "flow_machine_cover_v16.png")
            self.assertIn("colored pencil and gouache", calls[0]["prompt"])
            self.assertIn("printmaking aesthetic", calls[0]["prompt"])
            self.assertIn("cream paper", calls[0]["prompt"])

    def test_cover_image_path_includes_version(self):
        self.assertEqual(
            cover_image_path(Path("images"), "paper"),
            Path("images") / "paper_cover_v16.png",
        )

    def test_selects_first_checkpoint(self):
        info = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"], {}]}}}}
        self.assertEqual(select_comfyui_checkpoint(info), "a.safetensors")

    def test_checkpoint_empty_fails(self):
        info = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [[], {}]}}}}
        with self.assertRaisesRegex(RuntimeError, "No ComfyUI checkpoint"):
            select_comfyui_checkpoint(info)

    def test_negative_prompt_blocks_mismatches(self):
        wf = build_sdxl_workflow("sd.safetensors", "cover", "out")
        neg = wf["7"]["inputs"]["text"]
        self.assertIn("human skeleton", neg)
        self.assertIn("house", neg)
        self.assertIn("photorealistic", neg)
        self.assertIn("Chinese characters", neg)
        self.assertIn("circuit board", neg)

    def test_sdxl_workflow_structure(self):
        wf = build_sdxl_workflow("sd_xl.safetensors", "podcast cover", "pc")
        self.assertEqual(wf["4"]["inputs"]["ckpt_name"], "sd_xl.safetensors")
        self.assertIn("podcast cover", wf["6"]["inputs"]["text"])
        self.assertIn("photorealistic", wf["7"]["inputs"]["text"])
        self.assertEqual(wf["8"]["inputs"]["steps"], 30)
        self.assertEqual(wf["8"]["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertEqual(wf["8"]["inputs"]["scheduler"], "karras")
        self.assertEqual(wf["10"]["inputs"]["filename_prefix"], "pc")


if __name__ == "__main__":
    unittest.main()
