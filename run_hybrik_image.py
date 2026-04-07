import argparse
import inspect
import json
import os
import pickle
import sys
from contextlib import contextmanager
from collections import namedtuple
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HYBRIK_ROOT = os.path.join(REPO_ROOT, "third_party", "HybrIK-main")
DEFAULT_CONFIG = "configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml"
DEFAULT_CHECKPOINT = os.path.join("pretrained_models", "hybrik_hrnet.pth")
JOINT_NAMES_29 = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "jaw",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_thumb",
    "right_thumb",
    "head",
    "left_middle",
    "right_middle",
    "left_bigtoe",
    "right_bigtoe",
]


if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec_compat(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec_compat

for _alias, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _value)


@contextmanager
def pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_path(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def _xyxy_to_xywh(bbox: List[float]) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def _write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for tri in faces:
            a, b, c = tri.astype(int) + 1
            f.write(f"f {a} {b} {c}\n")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _load_hybrik_modules(hybrik_root: str):
    if hybrik_root not in sys.path:
        sys.path.insert(0, hybrik_root)

    from hybrik.models import builder
    from hybrik.utils.config import update_config
    from hybrik.utils.presets import SimpleTransform3DSMPLCam
    from hybrik.utils.vis import get_one_box, vis_2d, vis_bbox

    return builder, update_config, SimpleTransform3DSMPLCam, get_one_box, vis_2d, vis_bbox


def _build_transform(cfg: edict, transform_cls) -> Any:
    bbox_3d_shape = getattr(cfg.MODEL, "BBOX_3D_SHAPE", (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
    dummy_set = edict(
        {
            "joint_pairs_17": None,
            "joint_pairs_24": None,
            "joint_pairs_29": None,
            "bbox_3d_shape": bbox_3d_shape,
        }
    )
    return transform_cls(
        dummy_set,
        scale_factor=cfg.DATASET.SCALE_FACTOR,
        color_factor=cfg.DATASET.COLOR_FACTOR,
        occlusion=cfg.DATASET.OCCLUSION,
        input_size=cfg.MODEL.IMAGE_SIZE,
        output_size=cfg.MODEL.HEATMAP_SIZE,
        depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
        bbox_3d_shape=bbox_3d_shape,
        rot=cfg.DATASET.ROT_FACTOR,
        sigma=cfg.MODEL.EXTRA.SIGMA,
        train=False,
        add_dpg=False,
        loss_type=cfg.LOSS["TYPE"],
    )


def _load_state_dict(model: torch.nn.Module, checkpoint_path: str) -> None:
    save_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(save_dict, dict) and "model" in save_dict:
        model.load_state_dict(save_dict["model"])
    else:
        model.load_state_dict(save_dict)


def run_inference(
    image_path: str,
    out_dir: str,
    hybrik_root: str,
    config_path: str,
    checkpoint_path: str,
    detector_score_threshold: float,
    device_name: str,
) -> Dict[str, Any]:
    builder, update_config, transform_cls, get_one_box, vis_2d, vis_bbox = _load_hybrik_modules(hybrik_root)
    config_abs = _resolve_path(hybrik_root, config_path)
    checkpoint_abs = _resolve_path(hybrik_root, checkpoint_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(config_abs):
        raise FileNotFoundError(f"Config not found: {config_abs}")
    if not os.path.exists(checkpoint_abs):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_abs}")

    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    device = torch.device(device_name)

    _ensure_dir(out_dir)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with pushd(hybrik_root):
        cfg = update_config(config_abs)
        transform = _build_transform(cfg, transform_cls)

        det_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        det_model.to(device)
        det_model.eval()

        hybrik_model = builder.build_sppe(cfg.MODEL)
        _load_state_dict(hybrik_model, checkpoint_abs)
        hybrik_model.to(device)
        hybrik_model.eval()

        with torch.no_grad():
            det_input = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().div(255.0).to(device)
            det_output = det_model([det_input])[0]
            tight_bbox = get_one_box(det_output, thrd=detector_score_threshold)
            if tight_bbox is None:
                raise RuntimeError("No person bounding box detected in the input image.")

            pose_input, bbox, img_center = transform.test_transform(image_rgb, tight_bbox)
            pose_input = pose_input.to(device)[None, :, :, :]

            pose_output = hybrik_model(
                pose_input,
                flip_test=True,
                bboxes=torch.from_numpy(np.array(bbox)).to(device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(device).unsqueeze(0).float(),
            )

    pred_xyz_17 = pose_output.pred_xyz_jts_17.reshape(17, 3).detach().cpu().numpy()
    pred_xyz_29 = pose_output.pred_xyz_jts_29.reshape(29, 3).detach().cpu().numpy()
    pred_xyz_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(24, 3).detach().cpu().numpy()
    pred_uvd = pose_output.pred_uvd_jts.reshape(-1, 3).detach().cpu().numpy()
    pred_scores = pose_output.maxvals.detach().cpu().reshape(-1).numpy()[:29]
    pred_camera = pose_output.pred_camera.squeeze(0).detach().cpu().numpy()
    pred_betas = pose_output.pred_shape.squeeze(0).detach().cpu().numpy()
    pred_phi = pose_output.pred_phi.squeeze(0).detach().cpu().numpy()
    pred_theta = pose_output.pred_theta_mats.squeeze(0).detach().cpu().numpy().reshape(24, 3, 3)
    pred_vertices = pose_output.pred_vertices.squeeze(0).detach().cpu().numpy()
    transl = pose_output.transl.squeeze(0).detach().cpu().numpy()
    cam_root = pose_output.cam_root.squeeze(0).detach().cpu().numpy()
    faces = hybrik_model.smpl.faces.astype(np.int32)

    bbox_xywh = _xyxy_to_xywh(bbox)
    points_2d = pred_uvd[:, :2].copy() * bbox_xywh[2]
    points_2d[:, 0] += bbox_xywh[0]
    points_2d[:, 1] += bbox_xywh[1]

    bbox_vis = cv2.cvtColor(vis_bbox(image_rgb.copy(), tight_bbox), cv2.COLOR_RGB2BGR)
    joints_vis = cv2.cvtColor(vis_2d(image_rgb.copy(), tight_bbox, points_2d), cv2.COLOR_RGB2BGR)

    overlay = image_bgr.copy()
    for idx, (x, y) in enumerate(points_2d):
        cv2.circle(overlay, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)
        cv2.putText(
            overlay,
            str(idx),
            (int(round(x)) + 4, int(round(y)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(
        overlay,
        (int(round(tight_bbox[0])), int(round(tight_bbox[1]))),
        (int(round(tight_bbox[2])), int(round(tight_bbox[3]))),
        (0, 0, 255),
        2,
    )

    overlay_path = os.path.join(out_dir, "overlay.jpg")
    bbox_path = os.path.join(out_dir, "bbox_2d.jpg")
    joints_2d_path = os.path.join(out_dir, "joints_2d.jpg")
    obj_path = os.path.join(out_dir, "mesh.obj")
    json_path = os.path.join(out_dir, "joints3d.json")
    pkl_path = os.path.join(out_dir, "result.pkl")

    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(bbox_path, bbox_vis)
    cv2.imwrite(joints_2d_path, joints_vis)
    _write_obj(obj_path, pred_vertices, faces)

    json_payload = {
        "image_path": os.path.abspath(image_path),
        "bbox_xyxy": [float(v) for v in bbox],
        "detector_bbox_xyxy": [float(v) for v in tight_bbox],
        "camera": pred_camera,
        "translation": transl,
        "cam_root": cam_root,
        "betas": pred_betas,
        "phi": pred_phi,
        "theta_mats": pred_theta,
        # HybrIK camera model constants – used by serve_score.py to convert
        # pred_xyz_29 (root-relative depth-factor units) to camera-space metres.
        "depth_factor": float(getattr(hybrik_model, "depth_factor", 2.2)),
        "hybrik_focal_length": float(getattr(hybrik_model, "focal_length", 1000.0)),
        "hybrik_input_size": float(getattr(hybrik_model, "input_size", 256.0)),
        "joints_3d_17": pred_xyz_17,
        "joints_3d_24_struct": pred_xyz_24_struct,
        "joints_3d_29": [
            {
                "index": idx,
                "name": JOINT_NAMES_29[idx] if idx < len(JOINT_NAMES_29) else f"joint_{idx}",
                "xyz": pred_xyz_29[idx],
                "uvd": pred_uvd[idx],
                "score": float(pred_scores[idx]) if idx < len(pred_scores) else None,
            }
            for idx in range(pred_xyz_29.shape[0])
        ],
        "vertices_count": int(pred_vertices.shape[0]),
        "faces_count": int(faces.shape[0]),
        "device": str(device),
        "hybrik_root": os.path.abspath(hybrik_root),
        "config_path": os.path.abspath(config_abs),
        "checkpoint_path": os.path.abspath(checkpoint_abs),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(json_payload), f, ensure_ascii=False, indent=2)

    pickle_payload = {
        "bbox_xyxy": np.array(bbox),
        "detector_bbox_xyxy": np.array(tight_bbox),
        "pred_xyz_17": pred_xyz_17,
        "pred_xyz_29": pred_xyz_29,
        "pred_xyz_24_struct": pred_xyz_24_struct,
        "pred_uvd": pred_uvd,
        "pred_scores": pred_scores,
        "pred_camera": pred_camera,
        "pred_betas": pred_betas,
        "pred_phi": pred_phi,
        "pred_theta_mats": pred_theta,
        "pred_vertices": pred_vertices,
        "transl": transl,
        "cam_root": cam_root,
        "joint_names_29": JOINT_NAMES_29,
        "image_path": os.path.abspath(image_path),
        "config_path": os.path.abspath(config_abs),
        "checkpoint_path": os.path.abspath(checkpoint_abs),
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pickle_payload, f)

    return {
        "overlay_path": overlay_path,
        "bbox_path": bbox_path,
        "joints_2d_path": joints_2d_path,
        "obj_path": obj_path,
        "json_path": json_path,
        "pkl_path": pkl_path,
        "bbox": bbox,
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HybrIK on a single image and export 3D joints plus SMPL mesh.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--out_dir",
        default=os.path.join("out", "hybrik_image"),
        help="Output directory for HybrIK results.",
    )
    parser.add_argument(
        "--hybrik_root",
        default=DEFAULT_HYBRIK_ROOT,
        help="Path to the local HybrIK checkout.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Config path relative to HybrIK root, or absolute path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path relative to HybrIK root, or absolute path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use, e.g. auto, cpu, cuda:0.",
    )
    parser.add_argument(
        "--det_score_thr",
        type=float,
        default=0.9,
        help="Initial detector score threshold. HybrIK's helper will back off if needed.",
    )
    args = parser.parse_args()

    result = run_inference(
        image_path=args.image,
        out_dir=args.out_dir,
        hybrik_root=args.hybrik_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        detector_score_threshold=args.det_score_thr,
        device_name=args.device,
    )
    print(f"Device: {result['device']}")
    print(f"BBox: {result['bbox']}")
    print(f"OBJ: {result['obj_path']}")
    print(f"JSON: {result['json_path']}")
    print(f"PKL: {result['pkl_path']}")
    print(f"Overlay: {result['overlay_path']}")


if __name__ == "__main__":
    main()
