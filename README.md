---
title: Serve AI Tracker
emoji: 🎾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.x.x
app_file: app.py
pinned: false
python_version: "3.10"
---

# Serve AI Tracker

Tennis serve analysis toolkit built around pose extraction, phase detection, scoring, and report generation.

This repository mixes:

- original project code for serve analysis and reporting
- third-party code vendored under `third_party/`
- downloadable third-party model files
- sample local assets

If you publish or redistribute this repository, read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [PRIVACY_AND_COMPLIANCE.md](PRIVACY_AND_COMPLIANCE.md) first.

## What It Does

- extracts 2D pose landmarks from serve videos
- detects key serve phases and scoring metrics
- generates HTML/PDF-style analysis reports
- optionally runs HybrIK-based 3D verification for trophy-pose analysis

## Repository Layout

- `app.py`: Gradio entry point
- `make_report.py`: end-to-end report generation CLI
- `serve_score.py`: serve scoring and report helpers
- `extract_pose.py`, `extract_pose_compare.py`: MediaPipe-based pose extraction
- `run_hybrik_image.py`, `verify_trophy_3d.py`: HybrIK-based 3D verification
- `third_party/HybrIK-main/`: vendored upstream HybrIK source code
- `小程序/`: local prototype and sample assets; do not assume these are redistributable

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the MediaPipe pose model into `models/`:

```bash
python download_model.py
```

Generate a report from a serve clip:

```bash
python make_report.py --video D:\Serve_Score\input\serve.mp4 --out_dir D:\Serve_Score\out\serve_report
```

Run the Gradio app:

```bash
python app.py
```

## Third-Party And Legal Notes

### 1. Vendored third-party code

This repository includes a copy of `HybrIK` under `third_party/HybrIK-main/`.

- Upstream project: <https://github.com/Jeff-sjtu/HybrIK>
- Upstream license file is preserved at `third_party/HybrIK-main/LICENSE`
- Any redistribution of that directory must keep the original copyright and license notice

### 2. Downloaded model files

The pose landmark model used by this project is not authored by this repository.

- `download_model.py` downloads the MediaPipe Pose Landmarker model from Google's hosted source
- use the original upstream terms for any redistribution or commercial use
- if you are unsure about redistributing model binaries, prefer downloading them locally instead of committing them to your own repo

### 3. Sample data, personal data, and screenshots

Do not publish personal identity documents, user videos, or other sensitive sample materials unless you have explicit authorization.

## License Status Of This Repository

The root [LICENSE](LICENSE) file applies to the original code and documentation in this repository.

That means:

- third-party parts remain under their own upstream licenses or terms
- the root license does not override `third_party/` contents, model terms, dataset terms, or media restrictions
- sensitive sample materials and externally sourced assets still require separate permission and review

## Citation And Attribution

If this repository is used in academic or public work, credit both:

- this repository for the serve-analysis pipeline
- the upstream third-party projects listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Disclaimer

This repository documentation is provided for project hygiene and attribution clarity. It is not legal advice. You are responsible for checking the exact license and usage terms of all third-party code, models, datasets, and media before redistribution.
