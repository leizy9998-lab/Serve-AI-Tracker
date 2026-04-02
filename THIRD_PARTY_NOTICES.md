# Third-Party Notices

This repository mixes original project code with third-party code, model assets, and pretrained weights that do not all share the same license.

Do not assume that everything in this repo is covered by the root MIT license.

This file is an engineering notice, not legal advice.

## Practical Summary

| Component | Local path | Practical rule |
| --- | --- | --- |
| Original repo glue code | `app.py`, `serve_score.py`, `make_report.py`, wrappers | Covered by the root MIT license in `LICENSE`, unless a specific file says otherwise. |
| HybrIK code | `third_party/HybrIK-main/` | HybrIK upstream code is MIT-licensed. Keep the upstream license file and attribution. |
| HybrIK pretrained weights | `third_party/HybrIK-main/pretrained_models/hybrik_hrnet.pth` | Prefer user-side download from the official HybrIK release instructions. Do not mirror or redistribute publicly unless you have verified redistribution rights. |
| SMPL body model assets | `third_party/HybrIK-main/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` | Not covered by HybrIK's MIT license. Follow the official SMPL license terms from Max Planck / Meshcapade. |
| SMPL-X body model assets | `third_party/HybrIK-main/model_files/smplx/*` | Not covered by HybrIK's MIT license. Follow the official SMPL-X license terms from Max Planck / Meshcapade. |
| MediaPipe pose model bundle | `models/pose_landmarker_full.task` | Obtain from official Google AI Edge / MediaPipe distribution channels and comply with their terms. Do not assume the repo's license policy applies to that model file. |

## Why The 3D Stack Needs Extra Care

### 1. HybrIK code and 3D body assets are not the same licensing bucket

The vendored HybrIK repository includes an MIT license file at:

- `third_party/HybrIK-main/LICENSE`

However, the SMPL-related layer code in the vendored tree also contains a Max Planck proprietary notice. For example:

- `third_party/HybrIK-main/hybrik/models/layers/smpl/lbs.py`

That file explicitly states that use requires a valid license from Max Planck or an authorized licensor. This is the main reason the repo should not present the whole 3D stack as simply "MIT".

### 2. SMPL / SMPL-X model files are separate licensed assets

The repository currently expects body-model files such as:

- `third_party/HybrIK-main/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
- `third_party/HybrIK-main/model_files/smplx/SMPLX_NEUTRAL.pkl`
- `third_party/HybrIK-main/model_files/smplx/SMPLX_MALE.pkl`
- `third_party/HybrIK-main/model_files/smplx/SMPLX_FEMALE.pkl`

These files are not ordinary helper resources. They come from the SMPL / SMPL-X model ecosystem and must be handled according to the upstream license terms.

### 3. Pretrained checkpoints should be treated separately from source code

The official HybrIK repository documents pretrained model downloads and expects users to place them under `pretrained_models/`.

For a public GitHub repository, the safest practice is:

- keep checkpoints out of Git history
- document where users should obtain them
- do not re-license or repackage them as if they were part of the repository's original code

## Recommended Public Repo Policy

If you keep this repository public:

- keep the root MIT license scoped to the original project code
- keep third-party notices in this file
- keep large model assets and checkpoints out of Git unless redistribution rights are clear
- when in doubt, link users to the official download pages instead of bundling the files

## Official Sources

- HybrIK official repository: https://github.com/jeffffffli/HybrIK
- SMPL model license: https://smpl.is.tue.mpg.de/modellicense.html
- SMPL body license page: https://smpl.is.tue.mpg.de/license.html
- SMPL-X model license: https://smpl-x.is.tue.mpg.de/modellicense.html
- SMPL-X body license page: https://smpl-x.is.tue.mpg.de/bodylicense.html
- GitHub docs on repositories without a license: https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository
