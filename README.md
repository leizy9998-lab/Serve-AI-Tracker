---
title: Serve AI Tracker
emoji: 🎾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
python_version: "3.10"
---

# Serve AI Tracker

Serve clip scoring pipeline with optional 3D keyframe modeling.

## Space Features

- `Serve Report`: upload a serve video and generate `metrics.json`, `report.html`, optional `report.pdf`, annotated pose video, keyframes, and optional 3D keyframe overlays.
- `Single Frame 3D`: upload one keyframe image and export the HybrIK overlay, `mesh.obj`, `joints3d.json`, and `result.pkl`.

## Local CLI

```bash
python make_report.py --video D:\Serve_Score\input\serve.mp4 --out_dir D:\Serve_Score\out\serve_report
python run_hybrik_image.py --image D:\Serve_Score\out\serve_report\keyframes\trophy.png --out_dir D:\Serve_Score\out\hybrik_trophy
```

## Runtime Notes

- The 2D report flow requires `gradio`, `mediapipe`, `opencv-python-headless`, `numpy`, `pandas`, and `matplotlib`.
- The 3D workflow also requires `torch`, `torchvision`, `easydict`, the local `third_party/HybrIK-main` checkout, and a checkpoint at `third_party/HybrIK-main/pretrained_models/hybrik_hrnet.pth`.
- PDF export stays optional. If no supported PDF dependency is installed, the pipeline skips `report.pdf` gracefully.

## Output Location

- Space runs: `out/space_runs/...`
- Local CLI runs: any folder passed with `--out_dir`, typically under `D:\Serve_Score\out\...`
