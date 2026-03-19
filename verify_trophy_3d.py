import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import serve_score
from analyze_trophy_3d import analyze_3d_joints
from run_hybrik_image import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_HYBRIK_ROOT, run_inference


SKELETON_EDGES = [
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("left_foot", "left_bigtoe"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("right_foot", "right_bigtoe"),
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "jaw"),
    ("jaw", "head"),
    ("spine3", "left_collar"),
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_thumb"),
    ("left_wrist", "left_middle"),
    ("spine3", "right_collar"),
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_thumb"),
    ("right_wrist", "right_middle"),
]

MP_JOINT_MAP = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

SYMMETRY_SEGMENTS = {
    "upper_arm": ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow"),
    "forearm": ("left_elbow", "left_wrist", "right_elbow", "right_wrist"),
    "thigh": ("left_hip", "left_knee", "right_hip", "right_knee"),
    "shank": ("left_knee", "left_ankle", "right_knee", "right_ankle"),
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected top-level JSON object: {path}")
    return data


def _joint_items(data: Dict[str, object]) -> List[Dict[str, object]]:
    items = data.get("joints_3d_29")
    if not isinstance(items, list):
        raise RuntimeError("Expected `joints_3d_29` list in HybrIK JSON.")
    return [item for item in items if isinstance(item, dict)]


def _joint_xyz_map(data: Dict[str, object]) -> Dict[str, np.ndarray]:
    joints: Dict[str, np.ndarray] = {}
    for item in _joint_items(data):
        name = item.get("name")
        xyz = item.get("xyz")
        if isinstance(name, str) and isinstance(xyz, list) and len(xyz) == 3:
            joints[name] = np.array(xyz, dtype=float)
    return joints


def _joint_score_map(data: Dict[str, object]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for item in _joint_items(data):
        name = item.get("name")
        score = item.get("score")
        if isinstance(name, str) and isinstance(score, (int, float)):
            scores[name] = float(score)
    return scores


def _bbox_xywh(bbox_xyxy: Iterable[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def project_hybrik_joints_2d(data: Dict[str, object]) -> Dict[str, np.ndarray]:
    bbox = data.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RuntimeError("Expected `bbox_xyxy` in HybrIK JSON.")
    cx, cy, bw, _ = _bbox_xywh(bbox)

    points: Dict[str, np.ndarray] = {}
    for item in _joint_items(data):
        name = item.get("name")
        uvd = item.get("uvd")
        if not isinstance(name, str) or not isinstance(uvd, list) or len(uvd) != 3:
            continue
        x = float(uvd[0]) * bw + cx
        y = float(uvd[1]) * bw + cy
        points[name] = np.array([x, y], dtype=float)
    return points


def load_pose_frame_points(
    pose_csv: str,
    frame_num: int,
    image_w: int,
    image_h: int,
    use_coords: str = "smooth",
) -> Dict[str, np.ndarray]:
    df = pd.read_csv(pose_csv, encoding="utf-8-sig")
    frame_df = df[df["frame"] == frame_num]
    if frame_df.empty:
        return {}

    x_col = f"{use_coords}_x"
    y_col = f"{use_coords}_y"
    if x_col not in frame_df.columns or y_col not in frame_df.columns:
        raise RuntimeError(f"Missing columns `{x_col}` and `{y_col}` in pose CSV.")

    points: Dict[str, np.ndarray] = {}
    for name, lm in MP_JOINT_MAP.items():
        row = frame_df[frame_df["lm"] == lm]
        if row.empty:
            continue
        row0 = row.iloc[0]
        x = float(row0[x_col]) * image_w
        y = float(row0[y_col]) * image_h
        if np.isfinite(x) and np.isfinite(y):
            points[name] = np.array([x, y], dtype=float)
    return points


def compute_pose_consistency(
    hybrik_2d: Dict[str, np.ndarray],
    pose_2d: Dict[str, np.ndarray],
) -> Dict[str, object]:
    rows = []
    for name, pose_pt in pose_2d.items():
        if name not in hybrik_2d:
            continue
        err = float(np.linalg.norm(hybrik_2d[name] - pose_pt))
        rows.append(
            {
                "joint": name,
                "error_px": err,
                "hybrik_xy": hybrik_2d[name].tolist(),
                "pose_xy": pose_pt.tolist(),
            }
        )

    if not rows:
        return {"num_compared": 0, "per_joint": []}

    errors = np.array([row["error_px"] for row in rows], dtype=float)
    return {
        "num_compared": int(len(rows)),
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "max_error_px": float(np.max(errors)),
        "per_joint": rows,
    }


def compute_confidence_stats(data: Dict[str, object]) -> Dict[str, float]:
    scores = np.array(list(_joint_score_map(data).values()), dtype=float)
    if scores.size == 0:
        return {}
    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }


def compute_coverage_stats(
    points_2d: Dict[str, np.ndarray],
    data: Dict[str, object],
    image_w: int,
    image_h: int,
) -> Dict[str, float]:
    if not points_2d:
        return {}

    bbox = data.get("detector_bbox_xyxy") or data.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {}
    x1, y1, x2, y2 = [float(v) for v in bbox]

    inside_image = 0
    inside_bbox = 0
    for pt in points_2d.values():
        x, y = float(pt[0]), float(pt[1])
        if 0 <= x < image_w and 0 <= y < image_h:
            inside_image += 1
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside_bbox += 1
    total = max(len(points_2d), 1)
    return {
        "inside_image_ratio": inside_image / total,
        "inside_detector_bbox_ratio": inside_bbox / total,
    }


def compute_symmetry_stats(joints: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, (la, lb, ra, rb) in SYMMETRY_SEGMENTS.items():
        if la not in joints or lb not in joints or ra not in joints or rb not in joints:
            continue
        left_len = float(np.linalg.norm(joints[lb] - joints[la]))
        right_len = float(np.linalg.norm(joints[rb] - joints[ra]))
        out[f"{label}_left_len"] = left_len
        out[f"{label}_right_len"] = right_len
        if min(left_len, right_len) > 1e-8:
            out[f"{label}_symmetry_ratio"] = max(left_len, right_len) / min(left_len, right_len)
    return out


def _plot_joint_map(joints: Dict[str, np.ndarray], y_axis_down: bool) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, xyz in joints.items():
        x, y, z = xyz.tolist()
        out[name] = np.array([x, z, -y if y_axis_down else y], dtype=float)
    return out


def _set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_3d_views(joints: Dict[str, np.ndarray], y_axis_down: bool, out_path: str) -> None:
    plot_joints = _plot_joint_map(joints, y_axis_down)
    pts = np.stack(list(plot_joints.values()))
    fig = plt.figure(figsize=(13, 4.5))
    views = [
        ("Front", dict(elev=0, azim=-90)),
        ("Side", dict(elev=0, azim=0)),
        ("Top", dict(elev=90, azim=-90)),
    ]

    for idx, (title, kwargs) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        for a, b in SKELETON_EDGES:
            if a not in plot_joints or b not in plot_joints:
                continue
            seg = np.vstack([plot_joints[a], plot_joints[b]])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#2a6f97", linewidth=2)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="#d62828", s=18)
        ax.view_init(**kwargs)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Depth(Z)")
        ax.set_zlabel("Up(-Y)")
        _set_equal_axes(ax, pts)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_reprojection_compare(
    image_path: str,
    hybrik_2d: Dict[str, np.ndarray],
    pose_2d: Dict[str, np.ndarray],
    out_path: str,
    consistency: Optional[Dict[str, object]] = None,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    for name, pt in hybrik_2d.items():
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(image, (x, y), 4, (255, 255, 0), -1)

    for name, pt in pose_2d.items():
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.drawMarker(image, (x, y), (0, 80, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
        if name in hybrik_2d:
            hx, hy = hybrik_2d[name]
            cv2.line(image, (int(round(hx)), int(round(hy))), (x, y), (0, 255, 0), 1, cv2.LINE_AA)

    if consistency and consistency.get("num_compared", 0) > 0:
        mean_err = consistency.get("mean_error_px", float("nan"))
        max_err = consistency.get("max_error_px", float("nan"))
        text = f"mean_err={mean_err:.1f}px max_err={max_err:.1f}px"
        cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, image)


def save_mesh_overlay_compare(
    image_path: str,
    json_path: str,
    overlay_path: str,
    compare_path: str,
) -> Dict[str, object]:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    overlay, render_debug = serve_score.render_hybrik_mesh_overlay(
        image_or_path=image_path,
        data_or_path=json_path,
    )
    cv2.imwrite(overlay_path, overlay)

    gap = 18
    h, w = image.shape[:2]
    compare = np.full((h, w * 2 + gap, 3), 255, dtype=np.uint8)
    compare[:, :w] = image
    compare[:, w + gap :] = overlay

    for origin_x, text in [(18, "Original"), (w + gap + 18, "3D Mesh Overlay")]:
        cv2.putText(compare, text, (origin_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(compare, text, (origin_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (44, 44, 44), 1, cv2.LINE_AA)

    cv2.imwrite(compare_path, compare)
    render_debug["overlay_path"] = os.path.abspath(overlay_path)
    render_debug["compare_path"] = os.path.abspath(compare_path)
    return render_debug


def _parse_frame_num(explicit: Optional[int], image_path: str, json_path: str) -> Optional[int]:
    if explicit is not None:
        return int(explicit)
    for source in [image_path, json_path]:
        name = os.path.basename(source)
        match = re.search(r"(\d+)(?=\.[^.]+$)", name)
        if match:
            return int(match.group(1))
    return None


def _extract_video_frame(video_path: str, frame_num: int, out_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_num} from {video_path}")
        cv2.imwrite(out_path, frame)
    finally:
        cap.release()


def _neighbor_json_path(neighbor_root: str, frame_num: int) -> str:
    return os.path.join(neighbor_root, f"frame_{frame_num:04d}", "joints3d.json")


def generate_neighbor_outputs(
    video_path: str,
    center_frame: int,
    radius: int,
    neighbor_root: str,
    hybrik_root: str,
    config_path: str,
    checkpoint_path: str,
    device: str,
    force: bool,
) -> List[Tuple[int, str]]:
    outputs: List[Tuple[int, str]] = []
    _ensure_dir(neighbor_root)
    for frame_num in range(center_frame - radius, center_frame + radius + 1):
        frame_dir = os.path.join(neighbor_root, f"frame_{frame_num:04d}")
        json_path = os.path.join(frame_dir, "joints3d.json")
        if os.path.exists(json_path) and not force:
            outputs.append((frame_num, json_path))
            continue

        _ensure_dir(frame_dir)
        image_path = os.path.join(frame_dir, f"frame_{frame_num:04d}.jpg")
        _extract_video_frame(video_path, frame_num, image_path)
        run_inference(
            image_path=image_path,
            out_dir=frame_dir,
            hybrik_root=hybrik_root,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            detector_score_threshold=0.9,
            device_name=device,
        )
        outputs.append((frame_num, json_path))
    return outputs


def compute_neighbor_stability(neighbor_jsons: List[Tuple[int, str]]) -> Dict[str, object]:
    if len(neighbor_jsons) < 2:
        return {}

    per_frame = []
    all_joint_arrays: List[np.ndarray] = []
    joint_order: Optional[List[str]] = None

    for frame_num, json_path in neighbor_jsons:
        metrics, _debug = analyze_3d_joints(json_path)
        data = _load_json(json_path)
        joints = _joint_xyz_map(data)
        if joint_order is None:
            joint_order = sorted(joints.keys())
        arr = np.stack([joints[name] for name in joint_order], axis=0)
        all_joint_arrays.append(arr)
        per_frame.append({"frame": frame_num, **metrics})

    disp = []
    for prev, curr in zip(all_joint_arrays[:-1], all_joint_arrays[1:]):
        disp.append(float(np.mean(np.linalg.norm(curr - prev, axis=1))))

    metric_keys = [
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "trunk_inclination_deg",
        "shoulder_tilt_deg",
        "right_elbow_angle_deg",
    ]
    stability = {
        "frames": [row["frame"] for row in per_frame],
        "per_frame_metrics": per_frame,
        "mean_consecutive_joint_disp": float(np.mean(disp)),
        "max_consecutive_joint_disp": float(np.max(disp)),
    }
    for key in metric_keys:
        values = np.array([row[key] for row in per_frame], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            stability[f"{key}_std"] = float(np.std(finite))
            stability[f"{key}_range"] = float(np.max(finite) - np.min(finite))
    return stability


def save_neighbor_plot(stability: Dict[str, object], out_path: str) -> None:
    rows = stability.get("per_frame_metrics")
    if not isinstance(rows, list) or not rows:
        return

    frames = [row["frame"] for row in rows]
    metric_specs = [
        ("right_elbow_angle_deg", "Right Elbow"),
        ("left_knee_angle_deg", "Left Knee"),
        ("right_knee_angle_deg", "Right Knee"),
        ("trunk_inclination_deg", "Trunk"),
        ("shoulder_tilt_deg", "Shoulder Tilt"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metric_specs):
        values = [row.get(key, float("nan")) for row in rows]
        ax.plot(frames, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    axes[-1].text(
        0.05,
        0.7,
        f"mean consecutive joint disp: {stability.get('mean_consecutive_joint_disp', float('nan')):.4f}\n"
        f"max consecutive joint disp: {stability.get('max_consecutive_joint_disp', float('nan')):.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_warnings(
    confidence: Dict[str, float],
    consistency: Dict[str, object],
    symmetry: Dict[str, float],
    stability: Dict[str, object],
) -> List[str]:
    warnings: List[str] = []
    mean_score = confidence.get("mean_score")
    if mean_score is not None and mean_score < 0.03:
        warnings.append("HybrIK joint confidence is low; inspect overlay and mesh before trusting angles.")

    mean_err = consistency.get("mean_error_px")
    if isinstance(mean_err, (int, float)) and mean_err > 40:
        warnings.append("HybrIK 2D reprojection differs noticeably from pose CSV on the selected frame.")

    for key, value in symmetry.items():
        if key.endswith("_symmetry_ratio") and value > 1.25:
            warnings.append(f"Left/right limb length asymmetry looks high for {key}.")

    elbow_std = stability.get("right_elbow_angle_deg_std")
    if isinstance(elbow_std, (int, float)) and elbow_std > 10:
        warnings.append("Neighboring-frame elbow angle varies a lot around the target frame.")

    return warnings


def write_text_report(summary: Dict[str, object], out_path: str) -> None:
    lines = []
    lines.append("HybrIK verification summary")
    lines.append("")
    if "angles_3d" in summary:
        lines.append("3D angles:")
        for key, value in summary["angles_3d"].items():
            if isinstance(value, (int, float)):
                lines.append(f"  {key}: {value:.3f}")
    if "pose_consistency" in summary and summary["pose_consistency"].get("num_compared", 0):
        pc = summary["pose_consistency"]
        lines.append("")
        lines.append("2D pose consistency:")
        lines.append(f"  num_compared: {pc['num_compared']}")
        lines.append(f"  mean_error_px: {pc['mean_error_px']:.3f}")
        lines.append(f"  median_error_px: {pc['median_error_px']:.3f}")
        lines.append(f"  max_error_px: {pc['max_error_px']:.3f}")
    if "warnings" in summary:
        lines.append("")
        lines.append("Warnings:")
        if summary["warnings"]:
            for item in summary["warnings"]:
                lines.append(f"  - {item}")
        else:
            lines.append("  none")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify HybrIK trophy-pose output with 3D views, 2D consistency, and neighbor stability.")
    parser.add_argument("--json_path", default=os.path.join("out", "hybrik_trophy_125", "joints3d.json"))
    parser.add_argument("--image_path", default=None, help="Source image path. Defaults to `image_path` inside the JSON.")
    parser.add_argument("--pose_csv", default=None, help="Optional pose CSV for 2D consistency checks.")
    parser.add_argument("--frame_num", type=int, default=None, help="Video frame number used for the pose CSV lookup.")
    parser.add_argument("--use_coords", choices=["smooth", "raw"], default="smooth")
    parser.add_argument("--video", default=None, help="Optional source video path for neighboring-frame validation.")
    parser.add_argument("--neighbor_radius", type=int, default=0, help="Neighbor radius for stability validation when `--video` is provided.")
    parser.add_argument("--force_neighbors", action="store_true", help="Re-run HybrIK on neighbors even if cached JSON exists.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to `<json_dir>/verify`.")
    parser.add_argument("--device", default="auto", help="Torch device passed to HybrIK when neighbors are generated.")
    parser.add_argument("--hybrik_root", default=DEFAULT_HYBRIK_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    data = _load_json(args.json_path)
    image_path = args.image_path or data.get("image_path")
    if not isinstance(image_path, str):
        raise RuntimeError("Image path not provided and missing from HybrIK JSON.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.json_path)), "verify")
    _ensure_dir(out_dir)

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_h, image_w = image.shape[:2]

    metrics_3d, debug_3d = analyze_3d_joints(args.json_path)
    joints = _joint_xyz_map(data)
    hybrik_2d = project_hybrik_joints_2d(data)
    confidence = compute_confidence_stats(data)
    coverage = compute_coverage_stats(hybrik_2d, data, image_w, image_h)
    symmetry = compute_symmetry_stats(joints)

    frame_num = _parse_frame_num(args.frame_num, image_path, args.json_path)
    pose_points = {}
    pose_consistency: Dict[str, object] = {"num_compared": 0, "per_joint": []}
    if args.pose_csv and frame_num is not None:
        pose_points = load_pose_frame_points(args.pose_csv, frame_num, image_w, image_h, use_coords=args.use_coords)
        pose_consistency = compute_pose_consistency(hybrik_2d, pose_points)

    save_3d_views(joints, bool(debug_3d.get("y_axis_down", True)), os.path.join(out_dir, "3d_views.png"))
    mesh_overlay_debug = save_mesh_overlay_compare(
        image_path=image_path,
        json_path=args.json_path,
        overlay_path=os.path.join(out_dir, "mesh_overlay.png"),
        compare_path=os.path.join(out_dir, "mesh_overlay_compare.png"),
    )
    save_reprojection_compare(
        image_path,
        hybrik_2d,
        pose_points,
        os.path.join(out_dir, "reprojection_compare.png"),
        consistency=pose_consistency,
    )

    stability: Dict[str, object] = {}
    if args.video and frame_num is not None and args.neighbor_radius > 0:
        neighbor_root = os.path.join(out_dir, "neighbors")
        neighbor_jsons = generate_neighbor_outputs(
            video_path=args.video,
            center_frame=frame_num,
            radius=args.neighbor_radius,
            neighbor_root=neighbor_root,
            hybrik_root=args.hybrik_root,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            force=args.force_neighbors,
        )
        if not any(frame == frame_num for frame, _ in neighbor_jsons):
            neighbor_jsons.append((frame_num, args.json_path))
            neighbor_jsons.sort(key=lambda x: x[0])
        else:
            neighbor_jsons = sorted(neighbor_jsons, key=lambda x: x[0])
        stability = compute_neighbor_stability(neighbor_jsons)
        save_neighbor_plot(stability, os.path.join(out_dir, "neighbor_metrics.png"))

    warnings = build_warnings(confidence, pose_consistency, symmetry, stability)
    verdict = "good"
    if warnings:
        verdict = "caution"
    if any("differs noticeably" in item or "varies a lot" in item for item in warnings):
        verdict = "needs_review"

    summary = {
        "verdict": verdict,
        "angles_3d": metrics_3d,
        "angles_debug": debug_3d,
        "confidence": confidence,
        "coverage": coverage,
        "symmetry": symmetry,
        "mesh_overlay": mesh_overlay_debug,
        "pose_consistency": pose_consistency,
        "stability": stability,
        "warnings": warnings,
        "inputs": {
            "json_path": os.path.abspath(args.json_path),
            "image_path": os.path.abspath(image_path),
            "pose_csv": None if not args.pose_csv else os.path.abspath(args.pose_csv),
            "frame_num": frame_num,
            "video": None if not args.video else os.path.abspath(args.video),
        },
        "artifacts": {
            "mesh_overlay_png": os.path.abspath(os.path.join(out_dir, "mesh_overlay.png")),
            "mesh_overlay_compare_png": os.path.abspath(os.path.join(out_dir, "mesh_overlay_compare.png")),
            "3d_views_png": os.path.abspath(os.path.join(out_dir, "3d_views.png")),
            "reprojection_compare_png": os.path.abspath(os.path.join(out_dir, "reprojection_compare.png")),
            "neighbor_metrics_png": None if not stability else os.path.abspath(os.path.join(out_dir, "neighbor_metrics.png")),
        },
    }

    summary_path = os.path.join(out_dir, "verification_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_text_report(summary, os.path.join(out_dir, "verification_report.txt"))

    print(f"Saved verification summary to: {summary_path}")
    print(f"Verdict: {verdict}")
    if pose_consistency.get("num_compared", 0):
        print(
            "Pose consistency: "
            f"mean={pose_consistency['mean_error_px']:.1f}px "
            f"median={pose_consistency['median_error_px']:.1f}px "
            f"max={pose_consistency['max_error_px']:.1f}px"
        )
    if stability:
        print(
            "Neighbor stability: "
            f"mean_consecutive_joint_disp={stability.get('mean_consecutive_joint_disp', float('nan')):.4f} "
            f"right_elbow_std={stability.get('right_elbow_angle_deg_std', float('nan')):.2f}"
        )
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
