# Local 3D Asset Setup

This project can run in two modes:

- lightweight 2D demo mode
- local full report mode with HybrIK-based keyframe 3D modeling

The 3D mode needs extra assets that are intentionally not treated like normal source code dependencies.

## Required Local Assets

### MediaPipe pose model

Expected path:

- `models/pose_landmarker_full.task`

This is used by the pose extraction path in `make_report.py`.

### HybrIK checkpoint

Expected path:

- `third_party/HybrIK-main/pretrained_models/hybrik_hrnet.pth`

This is the default checkpoint path used by the local 3D helper in `make_report.py`.

### SMPL / SMPL-X body model assets

Expected examples:

- `third_party/HybrIK-main/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
- `third_party/HybrIK-main/model_files/smplx/SMPLX_NEUTRAL.pkl`
- `third_party/HybrIK-main/model_files/smplx/SMPLX_MALE.pkl`
- `third_party/HybrIK-main/model_files/smplx/SMPLX_FEMALE.pkl`

## Why These Files Are Not Bundled Like Normal Open-Source Assets

- the HybrIK code and the body-model assets are not governed by the same terms
- SMPL / SMPL-X assets have their own upstream license requirements
- pretrained checkpoints should not be casually mirrored into public Git history without checking redistribution rights

See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) before publishing or redistributing the 3D stack.

## Recommended Setup Flow

1. Clone the repository.
2. Install the local full-pipeline dependencies from `requirements-local-3d.txt`.
3. Obtain the MediaPipe model bundle from official MediaPipe / Google AI Edge sources.
4. Obtain HybrIK code and checkpoint files from the official HybrIK release flow.
5. Obtain SMPL / SMPL-X body model files from the official Max Planck / Meshcapade distribution flow.
6. Place the files into the paths listed above.

## Validation Command

After the assets are in place, validate the local full pipeline with:

```bash
python make_report.py --video path/to/serve.mp4 --out_dir out/test_report --verify_3d on --output_mode compact
```

If the local 3D runtime or assets are incomplete, `make_report.py` will skip the 3D portion instead of producing a full keyframe 3D bundle.

## Public Repo Recommendation

For a public GitHub repo:

- commit the code
- document asset locations
- keep checkpoints and body-model files out of Git
- point users to the official upstream download pages
