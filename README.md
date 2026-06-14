# PaperCast Agent

PaperCast Agent 是一个面向论文阅读与播客创作的本地工具。它可以把 arXiv 论文、PDF URL 或本地 PDF 转换成中文播报脚本，并在 Web 页面中展示处理进度、脚本、封面图和可选音频。

项目的视觉方向不是写实风，而是“艺术书籍封面 / 学术画册”风格：封面生成偏 2D 图版、彩铅素描、凌乱线条、纸张纹理、版心线和克制的科技感，尽量避免随机抽象图或无意义装饰。

## 功能特点

- 支持 arXiv ID、PDF URL、本地 PDF 上传。
- 自动下载并缓存 PDF 文件。
- 使用 MarkItDown 将 PDF 转换为 Markdown。
- 将长论文切分成可恢复的文本块。
- 调用 OpenAI-compatible LLM 对每个文本块进行摘要。
- 将分块摘要整合成结构化中文播报脚本。
- 提供 Flask Web UI，支持快速模式和完整模式。
- 支持通过本地 ComfyUI 生成跨学科、主题相关的 2D 封面图。
- 支持通过本地 CosyVoice-compatible 服务生成音频。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，然后填写你的 LLM 配置：

```text
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=deepseek-chat
SUMMARY_MAX_WORKERS=3
```

## 启动 Web 页面

```bash
python src/web_app.py
```

启动后打开 Flask 输出的本地地址，通常是：

```text
http://127.0.0.1:5000
```

Web 页面支持：

- 输入 arXiv ID
- 输入 PDF URL
- 上传本地 PDF
- 强制重新生成
- 快速模式：使用更大的文本块，并跳过封面和音频生成
- 完整模式：在媒体服务启用时生成封面和音频

## 命令行用法

```bash
python src/main.py --version
python src/main.py --help
python src/main.py --doctor
python src/main.py --arxiv-id "2401.00000"
python src/main.py --pdf-url "https://arxiv.org/pdf/2401.00000"
python src/main.py --arxiv-id "2401.00000" --force
```

默认情况下，已有输出会被跳过。需要重新生成时使用 `--force`。

## 输出目录

生成结果会缓存在 `data/` 目录下：

```text
data/pdfs/        下载或上传的 PDF
data/markdown/    转换后的 Markdown
data/chunks/      切分后的文本块和 JSONL 元数据
data/summaries/   每个文本块的摘要
data/scripts/     最终中文播报脚本
data/images/      生成的封面图
data/audio/       生成的音频文件
```

## 脚本格式

最终生成的播报脚本应包含以下章节：

```text
# 播报标题
# 播报脚本
# 关键词
# 适合延伸学习的概念
```

封面生成模块会读取标题、关键词、脚本开头段落以及摘要线索，让封面图跟随论文主题，而不是退化成通用 AI 插画。

## ComfyUI 封面生成

PaperCast 默认不会安装 ComfyUI，也不会下载模型。你需要先在本机启动 ComfyUI，并准备好可用 checkpoint。

在 `.env` 中启用封面生成：

```text
MEDIA_IMAGE_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_TIMEOUT_SECONDS=180
```

当前实现直接调用 ComfyUI 原生接口：

```text
GET  /object_info
POST /prompt
GET  /history/{prompt_id}
GET  /view
```

程序会从 ComfyUI 的 `CheckpointLoaderSimple` 列表中选择第一个可用 checkpoint，构建一个简洁的 SDXL 风格 workflow，并将 PNG 保存到 `data/images/`，文件名包含 prompt 版本号，例如：

```text
data/images/{paper_id}_cover_v10.png
```

封面 prompt 会根据论文主题选择视觉母题，目前覆盖：

```text
Flow Matching / 生成模型
人形机器人 / 具身智能
计算机视觉
大语言模型
医学 / 病理 / 临床
物理
材料科学
生物 / 基因 / 分子
社会科学 / 经济 / 政策
气候 / 地球系统 / 遥感
系统 / 网络 / 分布式
控制 / 强化学习
化学
数学
法律
能源系统
农业
```

目标是生成“有意义的论文图版”，例如机器人论文应出现清晰的人形机器人、关节和运动分析语义；Flow Matching 论文应出现噪声分布、数据流形、向量场和轨迹，而不是随机风景或无关抽象图。

ComfyUI 返回图片后，程序会使用 Pillow 添加很轻的纸纹和印刷质感，使封面与 Web UI 的艺术画册风格更统一。

## 音频生成

如果本机有兼容的 CosyVoice 服务，可以在 `.env` 中启用音频生成：

```text
MEDIA_VOICE_PROVIDER=cosyvoice
COSYVOICE_BASE_URL=http://127.0.0.1:50000
COSYVOICE_TIMEOUT_SECONDS=180
COSYVOICE_VOICE=default
```

如果不需要封面或音频，可以设置：

```text
MEDIA_IMAGE_PROVIDER=none
MEDIA_VOICE_PROVIDER=none
```

如果 Web 流程中的图片或音频生成失败，系统仍会返回已经生成的脚本，并在页面中显示媒体生成警告。

## 流水线说明

- PDF 下载结果会缓存，下载失败时会给出 URL、HTTP 状态和错误原因。
- Markdown 转换依赖 MarkItDown；如果依赖缺失或转换结果为空，会明确报错。
- 分块摘要支持断点恢复；如果后续块失败，已完成摘要会保留在磁盘中。
- 调用模型前会检查提示词文件：
  - `prompts/map_prompt.txt` 必须包含 `{chunk}`。
  - `prompts/reduce_prompt.txt` 必须包含 `{summaries}`。
- `python src/main.py --doctor` 会检查目录、提示词占位符、LLM 环境变量和依赖。

## 测试

```bash
python -m unittest discover -s tests -v
python -m unittest tests.test_image_generator -v
python src/main.py --help
```

其中 `tests.test_image_generator` 会覆盖 ComfyUI workflow 构建、checkpoint 选择、封面版本路径以及跨学科 prompt 映射逻辑。

## 项目结构

```text
src/main.py                CLI 和主流程编排
src/web_app.py             Flask Web UI 与 SSE 进度推送
src/config.py              .env 加载与 LLM 配置校验
src/arxiv_client.py        arXiv ID / PDF URL / 本地 PDF 解析
src/pdf_downloader.py      PDF 下载与缓存跳过
src/markdown_converter.py  MarkItDown 转换
src/splitter.py            Markdown 文本切分
src/llm_client.py          OpenAI-compatible chat client 与重试逻辑
src/summarizer.py          Map 阶段分块摘要
src/script_writer.py       Reduce 阶段生成最终脚本
src/image_generator.py     ComfyUI workflow 与跨学科封面 prompt
src/voice_generator.py     可选音频生成
src/media_config.py        媒体服务配置
tests/                     单元测试
```
