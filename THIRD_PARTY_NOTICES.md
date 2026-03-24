# Third-Party Notices

This file documents third-party code, models, templates, and materials that are referenced by or included in this repository.

It is a practical attribution record, not legal advice.

The root `LICENSE` file of this repository does not replace or override the licenses or terms of third-party materials listed here.

## 1. HybrIK

- Local path: `third_party/HybrIK-main/`
- Upstream project: <https://github.com/Jeff-sjtu/HybrIK>
- Copyright notice in bundled license:
  `Copyright (c) 2021 Machine Vision and Intelligence Group, SJTU`
- Bundled upstream license: MIT

Status in this repository:

- a copy of the upstream HybrIK source tree is vendored in `third_party/HybrIK-main/`
- the upstream `LICENSE` file has been preserved
- files in this directory are not claimed as original work of this repository

Redistribution note:

- if you redistribute `third_party/HybrIK-main/` or a modified version of it, keep the original license and copyright notice

Additional caution:

- some HybrIK-related workflows depend on external body-model files, pretrained weights, or datasets that are not bundled here
- those assets may have separate license or access restrictions
- follow the upstream HybrIK documentation and the original asset providers' terms

## 2. MediaPipe Pose Landmarker Model

- Local usage: `extract_pose.py`, `extract_pose_compare.py`, `make_report.py`
- Download helper: `download_model.py`
- Download URL used by this repository:
  <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task>

Status in this repository:

- the pose model is a third-party model file, not original content from this repository
- users should preferably download the model from the original upstream source

Redistribution note:

- before redistributing the `.task` model file, verify the applicable upstream model and platform terms
- if you are unsure, do not republish the binary; use a download script instead

## 3. User-Supplied Or Sample Media

- Local path examples: `小程序/`

Status:

- sample images, prototypes, screenshots, and local media in this repository may be user-supplied or externally sourced
- unless a file is clearly marked as original and redistributable, do not assume you can republish it

Required handling:

- remove personal identity documents and other sensitive images from public releases
- obtain permission before publishing user videos or identifiable personal data
- keep private materials out of commits, releases, and screenshots

## 4. Dependencies Installed Via Package Managers

This repository also relies on third-party Python packages declared in `requirements.txt`, including but not limited to:

- `gradio`
- `mediapipe`
- `opencv-python-headless`
- `numpy`
- `pandas`

These dependencies are governed by their own licenses. Installation of a package does not transfer ownership of that package to this repository.

## Maintainer Guidance

Before making the repository public or tagging a release:

1. confirm every bundled third-party directory has its original license file
2. avoid committing third-party model binaries unless redistribution rights are clear
3. remove personal documents, IDs, test media, and local-only configuration
4. keep attribution links and notices in the root README
