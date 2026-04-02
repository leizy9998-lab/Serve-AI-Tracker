---
title: "Serve-AI-Tracker"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "6.10.0"
app_file: "app.py"
pinned: false
python_version: "3.11"
---

# Serve AI Tracker

Serve AI Tracker is a tennis serve analysis project with two runtime paths:

- a lightweight 2D Gradio app for quick demos and Space deployment
- a local full pipeline that adds keyframe 3D modeling and HTML report export

The repository is intentionally split this way because the 2D flow is cheap enough for hosted demos, while the 3D flow depends on a heavier local environment with PyTorch and HybrIK assets.

## Demo

![Serve AI Tracker demo](./assets/serve-tracker-demo.gif)

The demo above shows the current smooth-pose tracking overlay produced by the local report pipeline.

## Highlights

- single-player serve tracking from broadcast-style tennis clips
- smooth 2D pose overlay export
- serve phase detection: trophy, racket drop, contact, finish
- compact HTML report generation
- optional keyframe 3D modeling through a local HybrIK-based workflow

## Docs

- [README.md](./README.md): public project overview
- [LOCAL_3D_ASSETS.md](./LOCAL_3D_ASSETS.md): practical local 3D asset setup
- [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md): third-party code and model asset licensing notes
- [CONTRIBUTING.md](./CONTRIBUTING.md): contribution guidelines

## What The Project Does

- tracks a single server through a clip
- smooths the 2D pose trajectory
- detects serve phases such as trophy, racket drop, contact, and finish
- scores the serve with simple biomechanical heuristics
- optionally runs per-keyframe 3D modeling with HybrIK
- exports a compact report package with overlay video, metrics, and keyframe comparisons

## Runtime Modes

| Mode | Entry point | Typical use |
| --- | --- | --- |
| Space / demo | `app.py` | upload one video and get a 2D overlay plus a short report |
| Local compact report | `make_report.py` | smooth 2D scoring + keyframe 3D + compact HTML report |
| Compatibility wrappers | `extract_pose_compare.py`, `run_hybrik_image.py` | thin shims kept for older commands |

## Repository Layout

```text
app.py                  Gradio Space entrypoint
serve_score.py          Core 2D serve scoring logic
make_report.py          Main local pipeline: extraction, 3D helpers, report export
extract_pose_compare.py Compatibility wrapper to the extraction subcommand in make_report.py
run_hybrik_image.py     Compatibility wrapper to the HybrIK subcommand in make_report.py
models/                 MediaPipe model assets
third_party/HybrIK-main HybrIK checkout and local resources
```

## Installation

### Option A: lightweight 2D demo install

Use this if you only want to preview the tracker or deploy the small Gradio app.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: full local report install

Use this if you want the full `make_report.py` flow with keyframe 3D.

```bash
conda create -n servepose python=3.11 -y
conda activate servepose
pip install -r requirements-local-3d.txt
```

You will also need local model assets. See [LOCAL_3D_ASSETS.md](./LOCAL_3D_ASSETS.md).

## Running The Project

### Run the lightweight 2D demo

```bash
python app.py
```

What you get:

- 2D pose overlay replay
- concise markdown report
- no HybrIK
- no keyframe 3D bundle

### Run the full local report

```bash
conda activate servepose
python make_report.py --video D:\Serve_Score\input\serve.mp4 --out_dir D:\Serve_Score\out\serve_report
```

What you get by default:

- smooth-only 2D pose extraction
- compact output mode
- English report output
- keyframe 3D bundle when the local 3D runtime is available

### Example full command

```bash
python make_report.py ^
  --video D:\Serve_Score\input\serve.mp4 ^
  --out_dir D:\Serve_Score\out\serve_report ^
  --verify_3d on ^
  --output_mode compact
```

The compact output keeps:

- `report.html`
- `metrics.json`
- `serve_overlay.mp4`
- report-required keyframe 3D comparison images

## Dependency Files

- `requirements.txt`
  Space / 2D runtime only
- `requirements-local-3d.txt`
  validated local full-pipeline stack for `make_report.py`

As of April 2, 2026, the public 2D dependency pins were refreshed to newer PyPI releases for Gradio, MediaPipe, and OpenCV. The local 3D manifest is intentionally pinned to the versions already validated in the local `servepose` workflow instead of blindly jumping to the newest PyTorch stack.

## Important Repo Notes

- the original repository code is now released under the root [MIT License](./LICENSE)
- the 3D stack includes third-party code and model assets that are not all covered by the same terms
- if you plan to publish, fork, redistribute, or commercialize the 3D path, read [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) first

## Repository License Status

The original project code in this repository is covered by the root [MIT License](./LICENSE).

That matters for a public GitHub repository:

- the MIT license applies to the original repo code unless a file or bundled third-party subtree says otherwise
- third-party code and assets keep their own upstream terms regardless of the root MIT license
- the local 3D stack still requires separate care because source code, checkpoints, and body-model assets are not one licensing bucket

## Assets You Still Need

The repo is configured to ignore large runtime assets and generated outputs. Public users should expect to provide these locally:

- `models/pose_landmarker_full.task`
- `third_party/HybrIK-main/pretrained_models/hybrik_hrnet.pth`

The local 3D path may also require SMPL / SMPL-X body-model files under `third_party/HybrIK-main/model_files/`.

See [LOCAL_3D_ASSETS.md](./LOCAL_3D_ASSETS.md) and [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

Large generated artifacts under `input/` and `out/` are also ignored.

## Output Notes

This project currently works best when the input clip contains one serve only.

- single-serve videos usually work well
- compilation videos should be cut into individual serves before running reports
- the local 3D path is heavier and slower than the 2D Space path

## Current Limitations

- the local 3D workflow is still more Windows-oriented than fully cross-platform
- some helper defaults still assume local model paths are available
- the 3D report is keyframe-focused, not a full-body temporal 3D reconstruction across the whole clip
- serve phase detection can degrade on edits, camera cuts, and highlight compilations
- the legal boundary between vendored 3D code and body-model assets must be handled carefully in public distributions

## Public-Facing Notes

- `make_report.py` is now the main local file for future development
- `extract_pose_compare.py` and `run_hybrik_image.py` are compatibility wrappers, not independent feature branches
- the Space path intentionally stays minimal so hosted demos do not need the full 3D dependency stack
- HybrIK source code, pretrained checkpoints, and SMPL / SMPL-X assets should be documented and treated separately when preparing public releases
- the root MIT license does not override third-party license terms inside vendored code or model assets

## Sources

- Gradio PyPI: https://pypi.org/project/gradio/
- MediaPipe PyPI: https://pypi.org/project/mediapipe/
- OpenCV headless PyPI: https://pypi.org/project/opencv-python-headless/
- Pandas PyPI: https://pypi.org/project/pandas/
- NumPy PyPI: https://pypi.org/project/numpy/
- HybrIK official repository: https://github.com/jeffffffli/HybrIK
- SMPL official site: https://smpl.is.tue.mpg.de/
- SMPL-X official site: https://smpl-x.is.tue.mpg.de/
- GitHub licensing docs: https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository
