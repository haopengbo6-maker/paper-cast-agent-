# PaperCast Agent

将学术论文转化为可播、可读、可收藏的中文研究图录。

放入论文 → 生成封面图 + 中文播报脚本 + 语音音频，全部编排在一页研究图录中。

## 界面预览

Web 界面采用"杂志封面 + 研究图录"的双页设计：

- **封面页**：固定全屏封面图（ComfyUI 生成，16:9 横版），模糊背景填充 + 原图居中，向下滚动淡出
- **内容页**：纸面翻页效果，标题区透出封面图，正文区纸色底，音频播放器 + 脚本阅读

## 快速开始

```bash
pip install -r requirements.txt
pip install markitdown[pdf]  # PDF 转换支持
cp .env.example .env          # 配置 LLM API Key
python src/web_app.py         # 启动 http://localhost:5000
```

## 功能

- arXiv ID / PDF URL / 本地 PDF 上传
- **速览模式**：仅生成脚本，快速预览
- **馆藏模式**：脚本 + 封面图 + 音频，完整图录
- ComfyUI SDXL 封面生成（v16 提示词系统）
- CosyVoice 语音合成
- SSE 实时进度推送
- 脚本逐字流式呈现

## 封面生成

封面提示词 v16 系统特点：

- **LLM 驱动**：封面提示词由 LLM 根据论文标题、关键词、摘要生成自然艺术指导语
- **程序化回退**：LLM 不可用时自动切换关键词→视觉元素映射（200+ 条目）
- **动态色彩**：从论文关键词推断色调（天文学→深蓝、生物学→鼠尾草绿、AI→石板蓝+铜色）
- **构图变化**：5 种交替（居中对称、左重右轻、对角线、三角、顶部垂下）
- **氛围光照**：4 种交替（左上柔光、侧光长影、均匀环境光、学术暗调）
- **16:9 横版**：1216×832，适配浏览器全屏
- 中英文关键词桥接（120+ 条目）
- 增强负面提示词（禁止漫画分格、边框、logo、镜头光晕等）

## 配置

```text
# LLM
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=deepseek-chat

# 封面（ComfyUI）
MEDIA_IMAGE_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188

# 音频（可选）
MEDIA_VOICE_PROVIDER=cosyvoice
COSYVOICE_BASE_URL=http://127.0.0.1:50000

# 禁用封面/音频
MEDIA_IMAGE_PROVIDER=none
MEDIA_VOICE_PROVIDER=none
```

## 输出结构

```text
data/
  pdfs/         PDF 缓存
  markdown/     Markdown 转换
  chunks/       文本分块
  summaries/    分块摘要
  scripts/      中文播报脚本
  images/       封面图（{paper_id}_cover_v16.png）
  audio/        语音音频
```

## 脚本格式

```text
# 播报标题
# 播报脚本
# 关键词
# 适合延伸学习的概念
```

## 命令行

```bash
python src/main.py --arxiv-id "2401.00000"
python src/main.py --arxiv-id "2401.00000" --force
python src/main.py --pdf-url "https://arxiv.org/pdf/2401.00000"
python src/main.py --doctor
```

## 项目结构

```text
src/
  web_app.py               Flask Web UI + SSE 进度
  main.py                   CLI 入口
  image_generator.py        封面提示词 v16 + ComfyUI workflow
  voice_generator.py        语音合成
  script_writer.py          脚本生成
  summarizer.py             分块摘要
  splitter.py               文本切分
  llm_client.py             LLM 客户端
  markdown_converter.py     PDF→Markdown
  arxiv_client.py           论文输入解析
  prompts.py                提示词加载
  config.py / media_config.py  配置
prompts/
  cover_prompt.txt          封面 LLM 提示词模板
  map_prompt.txt            摘要提示词
  reduce_prompt.txt         脚本整合提示词
tests/
  test_image_generator.py   封面生成测试
```
