import tempfile
import unittest
from pathlib import Path

from src.image_generator import (
    build_cover_prompt,
    build_sdxl_workflow,
    cover_image_path,
    generate_cover_image,
    select_comfyui_checkpoint,
    _map_keywords_to_visuals,
)
from src.media_config import ImageProviderConfig


TITLE = "# 播报标题\n"
SCRIPT = "# 播报脚本\n"
KEYWORDS = "# 关键词\n"


class ImageGeneratorTests(unittest.TestCase):
    # ── Prompt structure tests ──

    def test_build_cover_prompt_uses_script_title_and_keywords(self):
        script = (
            TITLE
            + "AI 论文收音机\n\n"
            + SCRIPT
            + "今天我们聊一个 agent 系统。\n\n"
            + KEYWORDS
            + "- Agent\n"
            + "- Tool Learning\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("AI 论文收音机", prompt)
        self.assertIn("Agent", prompt)
        self.assertIn("scientific plate illustration", prompt)

    def test_build_cover_prompt_uses_summary_hint(self):
        script = (
            TITLE
            + "Flow Matching 让生成模型学会抄近道\n\n"
            + KEYWORDS
            + "- Flow Matching\n"
        )

        prompt = build_cover_prompt(script, "把噪声分布沿最短路径搬到数据分布。")

        self.assertIn("research context", prompt)
        self.assertIn("噪声分布", prompt)
        self.assertIn("data-distribution manifold", prompt)

    def test_build_cover_prompt_uses_script_hint(self):
        script = (
            TITLE
            + "题目看起来很泛化\n\n"
            + SCRIPT
            + "这篇论文其实讨论人形机器人的双足行走和全身控制。\n"
            + KEYWORDS
            + "- embodiment\n"
        )

        prompt = build_cover_prompt(script)

        # The script hint triggers topic matching — the scene should reflect humanoid robot concepts
        self.assertIn("humanoid robot", prompt)
        self.assertIn("servo motors", prompt)
        self.assertIn("embodiment", prompt)

    # ── Topic matching tests ──

    def test_build_cover_prompt_keeps_flow_matching_cover_on_topic(self):
        script = (
            TITLE
            + "Flow Matching 让生成模型学会抄近道\n\n"
            + SCRIPT
            + "这篇论文讨论 Continuous Normalizing Flows、向量场、最优传输和扩散模型采样。\n"
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
        self.assertIn("Gaussian noise particles", prompt)
        self.assertIn("vector-field streamlines", prompt)
        self.assertIn("no visible text", prompt)
        self.assertIn("no Chinese characters", prompt)
        self.assertIn("no ancient art", prompt)

    def test_build_cover_prompt_recognizes_flow_machine_topic(self):
        script = (
            TITLE
            + "Flow Machine 论文里的生成模型新路线\n\n"
            + SCRIPT
            + "这篇论文讨论概率路径、ODE 轨迹和生成模型采样。\n"
            + KEYWORDS
            + "- Flow Machine\n"
            + "- generative model\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("Flow Machine", prompt)
        self.assertIn("Gaussian noise particles", prompt)
        self.assertIn("data-distribution manifold", prompt)
        self.assertIn("no generic scenery", prompt)

    def test_build_cover_prompt_handles_physics_topic(self):
        script = (
            TITLE
            + "量子变化与光学场分析\n\n"
            + SCRIPT
            + "这篇文章讨论波函数、相互作用和实验曲线。\n"
            + KEYWORDS
            + "- quantum\n"
            + "- optics\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("wave interference", prompt)
        self.assertIn("particle trajectories", prompt)
        self.assertIn("field lines", prompt)

    def test_build_cover_prompt_handles_social_science_topic(self):
        script = (
            TITLE
            + "社会行为与政策影响\n\n"
            + SCRIPT
            + "这篇文章讨论调查、基于人群的统计和政策反馈。\n"
            + KEYWORDS
            + "- sociology\n"
            + "- survey\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("population nodes", prompt)
        self.assertIn("survey response", prompt)
        self.assertIn("policy intervention", prompt)

    def test_build_cover_prompt_handles_chinese_medical_topic(self):
        script = (
            TITLE
            + "题目略写得像普通研究\n\n"
            + SCRIPT
            + "这篇论文关注医学病理切片和临床诊断流程。\n"
            + KEYWORDS
            + "- representation learning\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("radiology scan", prompt)
        self.assertIn("diagnostic contour", prompt)
        self.assertIn("tissue texture", prompt)

    def test_build_cover_prompt_handles_chinese_materials_topic(self):
        script = (
            TITLE
            + "新型材料与电池界面研究\n\n"
            + SCRIPT
            + "这篇论文分析半导体晶体、电极表面和微结构。\n"
            + KEYWORDS
            + "- interface\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("crystal lattice", prompt)
        self.assertIn("grain boundaries", prompt)
        self.assertIn("cross-section", prompt)

    def test_build_cover_prompt_handles_math_topic(self):
        script = (
            TITLE
            + "等式与证明的结构\n\n"
            + SCRIPT
            + "这篇论文议论线性代数、坐标几何和定理发现。\n"
            + KEYWORDS
            + "- theorem\n"
            + "- proof\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("geometric proof construction", prompt)
        self.assertIn("auxiliary construction lines", prompt)
        self.assertIn("matrix grid", prompt)

    def test_build_cover_prompt_handles_law_topic(self):
        script = (
            TITLE
            + "法律文书与规章分析\n\n"
            + SCRIPT
            + "这篇论文关注法律规制、合规与司法案例。\n"
            + KEYWORDS
            + "- regulation\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("legal documents", prompt)
        self.assertIn("balance scale", prompt)
        self.assertIn("citation threads", prompt)

    def test_build_cover_prompt_handles_energy_topic(self):
        script = (
            TITLE
            + "能源系统与电网调度\n\n"
            + SCRIPT
            + "这篇文章讨论光伏和风力的能源能量流动。\n"
            + KEYWORDS
            + "- solar\n"
            + "- grid\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("power grid", prompt)
        self.assertIn("power-flow arrows", prompt)
        self.assertIn("energy storage", prompt)

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
        self.assertIn("atmospheric layer", prompt)

    def test_build_cover_prompt_prevents_robot_skeleton_mismatch(self):
        script = (
            TITLE
            + "人形机器人的全身控制研究\n\n"
            + SCRIPT
            + "这篇论文讨论人形机器人的双足行走、伺服关节和全身平衡控制。\n"
            + KEYWORDS
            + "- humanoid robot\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("clearly mechanical", prompt)
        self.assertIn("servo motors", prompt)
        self.assertIn("no skin", prompt)
        self.assertIn("no skeleton", prompt)
        self.assertIn("no human anatomy", prompt)

    def test_build_cover_prompt_handles_geomagnetic_storm_topic(self):
        script = (
            TITLE
            + "地磁暴与空间天气预测\n\n"
            + SCRIPT
            + "这篇论文分析太阳风、磁层扰动、电离层响应和极光活动。\n"
            + KEYWORDS
            + "- geomagnetic storm\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("coronal mass ejection", prompt)
        self.assertIn("solar wind", prompt)
        self.assertIn("magnetosphere", prompt)
        self.assertIn("no houses", prompt)
        self.assertIn("no buildings", prompt)

    def test_build_cover_prompt_discourages_house_for_agriculture(self):
        script = (
            TITLE
            + "农业作物产量与土壤传感研究\n\n"
            + SCRIPT
            + "这篇论文研究作物、土壤分层、灌溉与产量预测。\n"
            + KEYWORDS
            + "- agriculture\n"
        )

        prompt = build_cover_prompt(script)

        self.assertIn("crop rows", prompt)
        self.assertIn("soil cross-section cutaway", prompt)
        self.assertIn("no farmhouse", prompt)
        self.assertIn("no barn", prompt)

    # ── Paper-specific keyword-to-visual injection tests ──

    def test_keyword_to_visual_maps_known_terms(self):
        visuals = _map_keywords_to_visuals(["reinforcement learning", "manipulation", "locomotion"])
        self.assertTrue(any("reinforcement learning" in v for v in visuals),
                        f"Expected 'reinforcement learning' in visuals: {visuals}")
        self.assertTrue(any("manipulation" in v for v in visuals),
                        f"Expected 'manipulation' in visuals: {visuals}")

    def test_keyword_to_visual_handles_unknown_terms(self):
        visuals = _map_keywords_to_visuals(["SomeNovelConcept"])
        self.assertTrue(any("SomeNovelConcept" in v for v in visuals),
                        f"Expected unknown term passed through: {visuals}")

    def test_keyword_to_visual_filters_generic_terms(self):
        visuals = _map_keywords_to_visuals(["model", "method", "data", "deep learning"])
        # Generic terms should be filtered out
        self.assertFalse(any("model" == v for v in visuals),
                         f"Generic 'model' should be filtered: {visuals}")
        self.assertFalse(any("method" == v for v in visuals),
                         f"Generic 'method' should be filtered: {visuals}")

    def test_build_cover_prompt_injects_paper_title_as_subject(self):
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

        # Title is the visual subject
        self.assertIn("GROOT-N1", prompt)
        self.assertIn("illustrating this specific research", prompt)
        # Keywords are woven in as visual elements
        self.assertIn("key concepts are visualized", prompt)
        self.assertIn("foundation model", prompt)

    def test_build_cover_prompt_weaves_keywords_into_scene(self):
        script = (
            TITLE
            + "Scaling Cross-Embodied Learning\n\n"
            + SCRIPT
            + "这篇论文讨论跨具身迁移学习、操作和导航。\n"
            + KEYWORDS
            + "- manipulation\n"
            + "- navigation\n"
            + "- transfer learning\n"
        )

        prompt = build_cover_prompt(script)

        # Keywords should appear as visual elements, not just in context brackets
        self.assertIn("key concepts are visualized", prompt)
        # At least one keyword should appear with "shown as"
        self.assertIn("manipulation", prompt)
        self.assertIn("navigation", prompt)

    # ── Integration / unchanged tests ──

    def test_disabled_provider_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_cover_image(
                "paper",
                TITLE + "标题",
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
                TITLE + "标题",
                Path(tmp),
                ImageProviderConfig("comfyui", "http://local", 9),
                request_image=fake_request,
                force=True,
            )
            self.assertEqual(path, Path(tmp) / "paper_cover_v13.png")
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
            + "Flow Matching 让生成模型学会抄近道\n\n"
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

            self.assertEqual(path, Path(tmp) / "flow_machine_cover_v13.png")
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("colored pencil and gouache", calls[0]["prompt"])
            self.assertIn("printmaking aesthetic", calls[0]["prompt"])
            self.assertIn("archival paper", calls[0]["prompt"])
            self.assertIn("letterpress impression", calls[0]["prompt"])
            self.assertIn("editorial art-book composition", calls[0]["prompt"])

    def test_cover_image_path_includes_prompt_version(self):
        self.assertEqual(
            cover_image_path(Path("images"), "paper"),
            Path("images") / "paper_cover_v13.png",
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
        self.assertIn("photorealistic", negative)
        self.assertIn("3d render", negative)
        self.assertIn("Chinese characters", negative)
        self.assertIn("circuit board", negative)

    def test_build_sdxl_workflow_uses_checkpoint_and_prompt(self):
        workflow = build_sdxl_workflow(
            checkpoint="sd_xl_base_1.0.safetensors",
            positive_prompt="podcast cover",
            output_prefix="papercast_cover",
        )

        self.assertEqual(workflow["4"]["inputs"]["ckpt_name"], "sd_xl_base_1.0.safetensors")
        self.assertIn("podcast cover", workflow["6"]["inputs"]["text"])
        self.assertIn("photorealistic", workflow["7"]["inputs"]["text"])
        self.assertIn("Chinese characters", workflow["7"]["inputs"]["text"])
        self.assertIn("ancient Chinese painting", workflow["7"]["inputs"]["text"])
        self.assertIn("human skeleton", workflow["7"]["inputs"]["text"])
        self.assertIn("house", workflow["7"]["inputs"]["text"])
        self.assertEqual(workflow["8"]["inputs"]["steps"], 35)
        self.assertEqual(workflow["8"]["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertEqual(workflow["8"]["inputs"]["scheduler"], "karras")
        self.assertEqual(workflow["10"]["inputs"]["filename_prefix"], "papercast_cover")


if __name__ == "__main__":
    unittest.main()
