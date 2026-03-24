# Privacy And Compliance

This repository analyzes sports video and may involve third-party code, model files, and sample media. Public release should be handled conservatively.

## Core Rules

1. Only upload, process, or publish videos that you own or are authorized to use.
2. Do not publish identity documents, face images, private chat screenshots, account data, or other personally identifiable information.
3. Do not assume a public GitHub repository gives you the right to republish third-party code, models, templates, logos, or datasets.
4. Keep original license notices intact for any vendored third-party code.

## Sensitive Data

The following material should stay out of a public repository unless there is explicit permission:

- identity cards, passports, student IDs, or licenses
- raw user videos containing identifiable people
- private configuration files and local debug files
- API keys, tokens, cookies, and cloud credentials
- proprietary training data or screenshots taken from paid services

## Third-Party Models And Datasets

Pose estimation and 3D body reconstruction workflows often involve separate restrictions beyond normal source-code licenses.

Examples:

- pretrained model weights
- body-model files such as SMPL or SMPL-X assets
- benchmark datasets such as Human3.6M, 3DPW, or COCO-derived annotations

If an upstream asset has its own access agreement, research-only terms, or click-through license, those terms still apply.

## Before Publishing

Use this checklist:

1. remove sensitive sample files from the tracked repository contents
2. replace local-only binaries with download instructions where possible
3. confirm each bundled third-party directory retains its upstream license
4. make README attribution explicit
5. verify that demo screenshots, icons, and logos can be redistributed
6. review release artifacts, not just source files

## Responsibility Boundary

These notes improve attribution and reduce obvious compliance mistakes, but they do not replace formal legal review. If you plan commercial deployment, institutional distribution, or publication tied to regulated data, get a human legal review.
