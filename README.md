---
title: "Serve-AI-Tracker"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "6.2.0"
app_file: "app.py"
pinned: false
python_version: "3.11"
---

# Serve AI Tracker

Serve clip scoring pipeline. The ModelScope space runs the 2D scoring flow only, while local 3D modeling stays available in the Windows `servepose` environment.

## Space Features

- Upload a serve video and generate the 2D pose overlay replay plus a concise Markdown scoring report.
- The space no longer runs HybrIK or keyframe 3D analysis.

## Local CLI

```bash
conda activate servepose
python make_report.py --video D:\Serve_Score\input\serve.mp4 --out_dir D:\Serve_Score\out\serve_report
python run_hybrik_image.py --image D:\Serve_Score\out\serve_report\keyframes\trophy.png --out_dir D:\Serve_Score\out\hybrik_trophy
```

## Runtime Notes

- Space runtime dependencies are intentionally trimmed to the 2D path: `gradio`, `mediapipe`, `opencv-python-headless`, `numpy`, `pandas`.
- Local 3D workflow still depends on the existing `servepose` environment plus the bundled HybrIK files and checkpoint.

## Output Location

- Space runs: `out/space_runs/...`
- Local CLI runs: any folder passed with `--out_dir`, typically under `D:\Serve_Score\out\...`

## Runtime File Set

- Space entrypoint: `app.py`, `serve_score.py`
- Local 3D tools retained in repo: `make_report.py`, `run_hybrik_image.py`, `third_party/HybrIK-main`, `models/`
