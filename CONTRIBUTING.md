# Contributing

Thanks for taking a look at the project.

This repository has two distinct runtime targets:

- a lightweight 2D Gradio / Space path
- a heavier local report path with keyframe 3D modeling

Please keep that split in mind when proposing changes.

## Before You Open A PR

- keep changes focused
- describe which runtime path you touched: 2D demo, local 3D report, or both
- if you changed output structure or CLI behavior, update `README.md`
- if you added a dependency, explain whether it belongs in `requirements.txt` or `requirements-local-3d.txt`
- if you added or changed third-party assets, update `THIRD_PARTY_NOTICES.md`

## Development Guidelines

- `serve_score.py` contains the core 2D scoring logic
- `make_report.py` is the main local pipeline file
- `extract_pose_compare.py` and `run_hybrik_image.py` are compatibility wrappers
- generated inputs and outputs under `input/` and `out/` should not be committed

## Asset And Licensing Rules

- the original repo code is MIT-licensed at the root, but that does not automatically re-license third-party code or model assets
- do not commit pretrained weights or body-model assets unless redistribution rights are clearly documented
- do not assume all 3D-related files are MIT just because HybrIK code is MIT
- check `THIRD_PARTY_NOTICES.md` before adding new 3D assets

## Testing Expectations

At minimum:

- run `python -m py_compile` on changed Python files
- if you touch 2D extraction or scoring, test `app.py` or `make_report.py` on a short serve clip
- if you touch the local 3D path, run a local `make_report.py --verify_3d on` smoke test when the assets are available

## Pull Request Notes

Useful things to include:

- what changed
- why it changed
- what command you ran to validate it
- any known limitations or follow-up work
