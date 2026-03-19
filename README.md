---
title: Serve AI Tracker
sdk: gradio
sdk_version: 4.x.x
app_file: app.py
python_version: "3.10"
---

# Serve AI Tracker

ModelScope 创空间精简运行版，仅保留在线演示所需文件：

- `app.py`
- `serve_score.py`
- `requirements.txt`

功能说明：

- 上传一段网球发球视频
- 用 MediaPipe 提取人体关键点
- 调用 `serve_score.analyze_serve()` 输出发球评分与关键阶段
- 返回带骨架回放的视频和文字评估结果

如果需要完整的 3D 建模、关键帧报告、HybrIK 工作流和本地开发脚本，请使用主仓库开发分支。
