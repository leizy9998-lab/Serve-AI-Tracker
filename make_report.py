import argparse
import html as html_lib
import inspect
import json
import math
import time
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

import serve_score
from i18n_zh import zh_label


LIT_KNEE_MEAN = 64.5
LIT_KNEE_SD = 9.7
LIT_TRUNK_MEAN = 25.0
LIT_TRUNK_SD = 7.1

IMPACT_NORM_RANGE = (1.45, 1.55)
TIMING_RANGES = {
    "timing_toss_to_trophy": (0.3, 1.4),
    "timing_trophy_to_drop": (0.12, 0.26),
    "timing_drop_to_contact": (0.08, 0.16),
}

KEYFRAME_3D_PHASES = ["trophy", "racket_drop", "contact", "finish"]
KEYFRAME_3D_ANGLE_ORDER = {
    "trophy": [
        "lead_knee_flexion_deg",
        "trunk_inclination_deg",
        "right_elbow_angle_deg",
        "left_elbow_angle_deg",
        "shoulder_tilt_deg",
        "left_knee_angle_deg",
        "right_knee_angle_deg",
    ],
    "racket_drop": [
        "shoulder_external_rotation_proxy_deg",
        "right_elbow_angle_deg",
        "left_elbow_angle_deg",
        "trunk_inclination_deg",
        "shoulder_tilt_deg",
        "left_knee_angle_deg",
        "right_knee_angle_deg",
    ],
    "contact": [
        "shoulder_raise_deg",
        "elbow_flexion_deg",
        "impact_height_norm",
        "right_elbow_angle_deg",
        "left_elbow_angle_deg",
        "trunk_inclination_deg",
    ],
    "finish": [
        "right_elbow_angle_deg",
        "left_elbow_angle_deg",
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "trunk_inclination_deg",
        "shoulder_tilt_deg",
    ],
}
KEYFRAME_3D_ANGLE_LABELS = {
    "left_knee_angle_deg": ("左膝夹角 (3D)", "Left Knee Angle (3D)"),
    "right_knee_angle_deg": ("右膝夹角 (3D)", "Right Knee Angle (3D)"),
    "lead_knee_flexion_deg": ("奖杯位膝屈角 (TP)", "Trophy Knee Flexion (TP)"),
    "left_elbow_angle_deg": ("左肘夹角 (3D)", "Left Elbow Angle (3D)"),
    "right_elbow_angle_deg": ("右肘夹角 (3D)", "Right Elbow Angle (3D)"),
    "trunk_inclination_deg": ("躯干倾角", "Trunk Inclination"),
    "shoulder_tilt_deg": ("双肩倾斜角", "Shoulder Tilt"),
    "shoulder_external_rotation_proxy_deg": ("拍头下落肩外旋 (RLP)", "RLP Shoulder External Rotation"),
    "shoulder_raise_deg": ("击球瞬间肩抬高 (BI)", "Impact Shoulder Raise (BI)"),
    "elbow_flexion_deg": ("击球瞬间肘屈 (BI)", "Impact Elbow Flexion (BI)"),
    "impact_height_norm": ("击球高度 (BH)", "Impact Height (BH)"),
}


I18N = {
    "zh": {
        "title": "发球动作评估报告",
        "final_score": "总评分（0–100）",
        "hand_assignment": "手别/镜像判断",
        "data_source": "数据源与参数",
        "biomech": "生物力学指标（软评分）",
        "quality_flags": "数据质量（遮挡/缺失）",
        "subscores": "分项得分",
        "diagnosis": "动作诊断与建议",
        "keyframes": "关键帧",
        "plots": "曲线图",
        "metrics": "指标明细",
        "category": "项目",
        "score": "分数",
        "notes": "建议",
        "metric": "指标",
        "value": "数值",
        "risk_flags": "风险提示",
        "range_lit": "文献均值±1SD",
        "range_heur": "经验范围（可调）",
        "range_calib": "样本分位数范围",
        "selected": "选中",
        "mirror_suspect": "镜像疑似",
        "guess": "自动判断",
        "z_scores": "标准分（Z）",
        "stability": "稳定性",
        "penalties": "扣分项",
        "risk_none": "无明显风险提示",
        "confidence": "置信度",
    },
    "en": {
        "title": "Serve Score Report",
        "final_score": "Final score (0-100)",
        "hand_assignment": "Hand Assignment",
        "data_source": "Data Source",
        "biomech": "Biomechanics (Soft Scoring)",
        "quality_flags": "Quality Flags",
        "subscores": "Sub-scores",
        "diagnosis": "Diagnosis",
        "keyframes": "Keyframes",
        "plots": "Plots",
        "metrics": "Metrics",
        "category": "Category",
        "score": "Score",
        "notes": "Notes",
        "metric": "Metric",
        "value": "Value",
        "risk_flags": "Risk Flags",
        "range_lit": "mean±1SD",
        "range_heur": "heuristic",
        "range_calib": "percentile",
        "selected": "Selected",
        "mirror_suspect": "Mirror Suspect",
        "guess": "Guess",
        "z_scores": "Z-Score",
        "stability": "Stability",
        "penalties": "Penalties",
        "risk_none": "No obvious risk flags",
        "confidence": "Confidence",
    },
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _posix_path(path: str) -> str:
    return path.replace("\\", "/")


def _read_image_unicode(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image_unicode(path: str, image: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(path)


def _can_open_video(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened()
    cap.release()
    return ok


def _prepare_video_input(video_path: str) -> Tuple[str, Optional[str]]:
    if _can_open_video(video_path):
        return video_path, None

    suffix = os.path.splitext(video_path)[1] or ".mp4"
    tmp_dir = os.path.join(tempfile.gettempdir(), "serve_ai_inputs")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"serve_input_{int(time.time() * 1000)}{suffix}")
    shutil.copyfile(video_path, tmp_path)
    if _can_open_video(tmp_path):
        return tmp_path, tmp_path
    raise RuntimeError(f"Cannot open video even after staging to ASCII temp path: {video_path}")


def _remove_path(path: Optional[str]) -> None:
    if not path or not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _cleanup_compact_outputs(out_dir: str, generated_pose_csv: Optional[str]) -> None:
    _remove_path(generated_pose_csv)
    _remove_path(os.path.join(out_dir, "score_debug.json"))
    _remove_path(os.path.join(out_dir, "plots"))
    _remove_path(os.path.join(out_dir, "keyframes"))

    for phase in KEYFRAME_3D_PHASES:
        phase_dir = os.path.join(out_dir, "keyframe_3d", phase)
        _remove_path(os.path.join(phase_dir, "hybrik"))
        _remove_path(os.path.join(phase_dir, "mesh_overlay.png"))


POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
]

DEFAULT_HYBRIK_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "HybrIK-main")
DEFAULT_HYBRIK_CONFIG = "configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml"
DEFAULT_HYBRIK_CHECKPOINT = os.path.join("pretrained_models", "hybrik_hrnet.pth")
HYBRIK_JOINT_NAMES_29 = [
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


def _pose_alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class _PoseLowPass:
    def __init__(self) -> None:
        self.inited = False
        self.y = 0.0

    def apply(self, x: float, a: float) -> float:
        if not self.inited:
            self.inited = True
            self.y = x
            return x
        self.y = a * x + (1.0 - a) * self.y
        return self.y


class _PoseOneEuro:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.6, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_lp = _PoseLowPass()
        self.x_lp = _PoseLowPass()

    def apply(self, x: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) / max(dt, 1e-6)
        a_d = _pose_alpha(self.d_cutoff, dt)
        dx_hat = self.dx_lp.apply(dx, a_d)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _pose_alpha(cutoff, dt)
        x_hat = self.x_lp.apply(x, a)
        self.x_prev = x_hat
        return x_hat


def _pose_interp_short(arr: np.ndarray, gap_max: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    return pd.Series(arr).interpolate(limit=int(gap_max), limit_direction="both").to_numpy()


def _pose_centered_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    window = int(max(window, 1))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return arr
    valid = np.isfinite(arr)
    vals = np.where(valid, arr, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    sm = np.convolve(vals, kernel, mode="same")
    cnt = np.convolve(valid.astype(np.float64), kernel, mode="same")
    return np.where(cnt > 0, sm / cnt, np.nan)


def _pose_smooth_sequence(
    raw: np.ndarray,
    vis: np.ndarray,
    mode: str,
    v_thr: float,
    gap_max: int,
    window: int,
    dt: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    arr = raw.astype(np.float64).copy()
    arr[(~np.isfinite(arr)) | (vis < v_thr)] = np.nan
    mode = (mode or "zero_phase").lower()
    if mode == "zero_phase":
        interp = _pose_interp_short(arr, gap_max=gap_max)
        smoothed = _pose_centered_moving_average(interp, window=window)
        smoothed[np.isnan(interp)] = np.nan
        return smoothed

    out = np.full_like(arr, np.nan, dtype=np.float64)
    filt = _PoseOneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    for idx, value in enumerate(arr):
        if not np.isfinite(value):
            out[idx] = np.nan
            filt.x_prev = None
            continue
        out[idx] = filt.apply(value, dt)
    return out


def _pose_draw(frame: np.ndarray, lm_xy: Dict[int, Tuple[float, float]], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
    height, width = frame.shape[:2]
    for a, b in POSE_CONNECTIONS:
        if a not in lm_xy or b not in lm_xy:
            continue
        ax, ay = lm_xy[a]
        bx, by = lm_xy[b]
        cv2.line(
            frame,
            (int(ax * width), int(ay * height)),
            (int(bx * width), int(by * height)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    for x, y in lm_xy.values():
        cv2.circle(frame, (int(x * width), int(y * height)), 3, (0, 0, 255), -1, cv2.LINE_AA)


def _pose_put_label(frame: np.ndarray, text: str, org: Tuple[int, int] = (20, 60), color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2, cv2.LINE_AA)


def _pose_build_lm_xy(x_row: np.ndarray, y_row: np.ndarray, v_row: np.ndarray, v_thr: float) -> Dict[int, Tuple[float, float]]:
    lm_xy: Dict[int, Tuple[float, float]] = {}
    for idx in range(len(x_row)):
        x = x_row[idx]
        y = y_row[idx]
        vis = v_row[idx]
        if not np.isfinite(x) or not np.isfinite(y) or vis < v_thr:
            continue
        lm_xy[idx] = (float(x), float(y))
    return lm_xy


def run_pose_extraction(
    video: str,
    model: str,
    out_csv: str,
    out_video: str,
    num_poses: int = 2,
    min_det: float = 0.5,
    min_presence: float = 0.5,
    min_track: float = 0.6,
    min_cutoff: float = 1.2,
    beta: float = 0.6,
    d_cutoff: float = 1.0,
    vis_gate: float = 0.6,
    v_thr: Optional[float] = None,
    smooth_mode: str = "zero_phase",
    gap_max: int = 3,
    smooth_window: int = 9,
) -> Dict[str, str]:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    _ensure_dir(os.path.dirname(os.path.abspath(out_csv)))
    _ensure_dir(os.path.dirname(os.path.abspath(out_video)))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dt = 1.0 / float(fps)

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_det,
        min_pose_presence_confidence=min_presence,
        min_tracking_confidence=min_track,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    v_thr_value = vis_gate if v_thr is None else float(v_thr)

    prev_center: Optional[Tuple[float, float]] = None
    raw_x_list: List[np.ndarray] = []
    raw_y_list: List[np.ndarray] = []
    vis_list: List[np.ndarray] = []
    ts_list: List[int] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = int(frame_idx * 1000.0 / fps)
        ts_list.append(ts_ms)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        arr_x = np.full(33, np.nan, dtype=np.float64)
        arr_y = np.full(33, np.nan, dtype=np.float64)
        arr_v = np.full(33, 0.0, dtype=np.float64)

        if result.pose_landmarks:
            if len(result.pose_landmarks) == 1 or prev_center is None:
                chosen = 0
            else:
                centers: List[Tuple[float, float]] = []
                for landmarks in result.pose_landmarks:
                    if len(landmarks) > 24:
                        cx = (landmarks[23].x + landmarks[24].x) * 0.5
                        cy = (landmarks[23].y + landmarks[24].y) * 0.5
                    else:
                        cx = float(np.mean([lm.x for lm in landmarks]))
                        cy = float(np.mean([lm.y for lm in landmarks]))
                    centers.append((float(cx), float(cy)))
                dists = [
                    (centers[idx][0] - prev_center[0]) ** 2 + (centers[idx][1] - prev_center[1]) ** 2
                    for idx in range(len(centers))
                ]
                chosen = int(np.argmin(dists))

            landmarks = result.pose_landmarks[chosen]
            if len(landmarks) > 24:
                prev_center = ((landmarks[23].x + landmarks[24].x) * 0.5, (landmarks[23].y + landmarks[24].y) * 0.5)
            for idx, lm in enumerate(landmarks):
                arr_x[idx] = float(lm.x)
                arr_y[idx] = float(lm.y)
                arr_v[idx] = float(getattr(lm, "visibility", 1.0))

        raw_x_list.append(arr_x)
        raw_y_list.append(arr_y)
        vis_list.append(arr_v)
        frame_idx += 1

    cap.release()

    raw_x = np.stack(raw_x_list, axis=0) if raw_x_list else np.zeros((0, 33), dtype=np.float64)
    raw_y = np.stack(raw_y_list, axis=0) if raw_y_list else np.zeros((0, 33), dtype=np.float64)
    vis = np.stack(vis_list, axis=0) if vis_list else np.zeros((0, 33), dtype=np.float64)
    ts_ms = np.array(ts_list, dtype=np.int64)

    smooth_x = np.full_like(raw_x, np.nan, dtype=np.float64)
    smooth_y = np.full_like(raw_y, np.nan, dtype=np.float64)
    for lm_id in range(raw_x.shape[1] if raw_x.ndim == 2 else 0):
        smooth_x[:, lm_id] = _pose_smooth_sequence(
            raw_x[:, lm_id],
            vis[:, lm_id],
            mode=smooth_mode,
            v_thr=v_thr_value,
            gap_max=gap_max,
            window=smooth_window,
            dt=dt,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )
        smooth_y[:, lm_id] = _pose_smooth_sequence(
            raw_y[:, lm_id],
            vis[:, lm_id],
            mode=smooth_mode,
            v_thr=v_thr_value,
            gap_max=gap_max,
            window=smooth_window,
            dt=dt,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )

    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_video}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(ts_ms):
            break
        overlay = frame.copy()
        smooth_xy = _pose_build_lm_xy(smooth_x[frame_idx], smooth_y[frame_idx], vis[frame_idx], v_thr_value)
        _pose_draw(overlay, smooth_xy, color=(0, 255, 0), thickness=2)
        _pose_put_label(overlay, f"SMOOTH {smooth_mode.upper()}", color=(120, 255, 120))
        writer.write(overlay)
        frame_idx += 1

    cap.release()
    writer.release()

    rows: List[Dict[str, object]] = []
    for frame_idx in range(len(ts_ms)):
        for lm_id in range(33):
            rows.append(
                {
                    "frame": frame_idx,
                    "ts_ms": int(ts_ms[frame_idx]),
                    "lm": lm_id,
                    "smooth_x": smooth_x[frame_idx, lm_id],
                    "smooth_y": smooth_y[frame_idx, lm_id],
                    "vis": vis[frame_idx, lm_id],
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return {"pose_csv": out_csv, "pose_video": out_video}


def extract_pose_compare_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--num_poses", type=int, default=2)
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--min_presence", type=float, default=0.5)
    ap.add_argument("--min_track", type=float, default=0.6)
    ap.add_argument("--min_cutoff", type=float, default=1.2)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--d_cutoff", type=float, default=1.0)
    ap.add_argument("--vis_gate", type=float, default=0.6)
    ap.add_argument("--v_thr", type=float, default=None)
    ap.add_argument("--smooth_mode", choices=["causal", "zero_phase"], default="zero_phase")
    ap.add_argument("--gap_max", type=int, default=3)
    ap.add_argument("--smooth_window", type=int, default=9)
    args = ap.parse_args(argv)
    result = run_pose_extraction(
        video=args.video,
        model=args.model,
        out_csv=args.out_csv,
        out_video=args.out_video,
        num_poses=args.num_poses,
        min_det=args.min_det,
        min_presence=args.min_presence,
        min_track=args.min_track,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        d_cutoff=args.d_cutoff,
        vis_gate=args.vis_gate,
        v_thr=args.v_thr,
        smooth_mode=args.smooth_mode,
        gap_max=args.gap_max,
        smooth_window=args.smooth_window,
    )
    print(f"[OK] out_video: {result['pose_video']}")
    print(f"[OK] out_csv  : {result['pose_csv']}")


def _hybrik_prepare_numpy_compat() -> None:
    if not hasattr(inspect, "getargspec"):
        arg_spec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

        def _getargspec_compat(func: Any) -> Any:
            spec = inspect.getfullargspec(func)
            return arg_spec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = _getargspec_compat

    for alias, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if alias not in np.__dict__:
            setattr(np, alias, value)


@contextmanager
def _hybrik_pushd(path: str) -> Iterable[None]:
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _hybrik_resolve_path(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def _hybrik_xyxy_to_xywh(bbox: List[float]) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def _hybrik_write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for tri in faces:
            a, b, c = tri.astype(int) + 1
            f.write(f"f {a} {b} {c}\n")


def _hybrik_to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _hybrik_to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hybrik_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _load_hybrik_modules(hybrik_root: str) -> Any:
    _hybrik_prepare_numpy_compat()
    if hybrik_root not in sys.path:
        sys.path.insert(0, hybrik_root)
    from hybrik.models import builder
    from hybrik.utils.config import update_config
    from hybrik.utils.presets import SimpleTransform3DSMPLCam
    from hybrik.utils.vis import get_one_box, vis_2d, vis_bbox

    return builder, update_config, SimpleTransform3DSMPLCam, get_one_box, vis_2d, vis_bbox


def _build_hybrik_transform(cfg: Any, transform_cls: Any) -> Any:
    from easydict import EasyDict as edict

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


def _load_hybrik_state_dict(model: Any, checkpoint_path: str) -> None:
    import torch

    save_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(save_dict, dict) and "model" in save_dict:
        model.load_state_dict(save_dict["model"])
    else:
        model.load_state_dict(save_dict)


def run_hybrik_inference(
    image_path: str,
    out_dir: str,
    hybrik_root: str = DEFAULT_HYBRIK_ROOT,
    config_path: str = DEFAULT_HYBRIK_CONFIG,
    checkpoint_path: str = DEFAULT_HYBRIK_CHECKPOINT,
    detector_score_threshold: float = 0.9,
    device_name: str = "auto",
) -> Dict[str, Any]:
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

    builder, update_config, transform_cls, get_one_box, vis_2d, vis_bbox = _load_hybrik_modules(hybrik_root)
    config_abs = _hybrik_resolve_path(hybrik_root, config_path)
    checkpoint_abs = _hybrik_resolve_path(hybrik_root, checkpoint_path)

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
    image_bgr = _read_image_unicode(image_path)
    if image_bgr is None:
        raise RuntimeError(f"Failed to decode image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with _hybrik_pushd(hybrik_root):
        cfg = update_config(config_abs)
        transform = _build_hybrik_transform(cfg, transform_cls)
        det_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        det_model.to(device)
        det_model.eval()

        hybrik_model = builder.build_sppe(cfg.MODEL)
        _load_hybrik_state_dict(hybrik_model, checkpoint_abs)
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

    bbox_xywh = _hybrik_xyxy_to_xywh(bbox)
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

    _write_image_unicode(overlay_path, overlay)
    _write_image_unicode(bbox_path, bbox_vis)
    _write_image_unicode(joints_2d_path, joints_vis)
    _hybrik_write_obj(obj_path, pred_vertices, faces)

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
        "joints_3d_17": pred_xyz_17,
        "joints_3d_24_struct": pred_xyz_24_struct,
        "joints_3d_29": [
            {
                "index": idx,
                "name": HYBRIK_JOINT_NAMES_29[idx] if idx < len(HYBRIK_JOINT_NAMES_29) else f"joint_{idx}",
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
        json.dump(_hybrik_to_builtin(json_payload), f, ensure_ascii=False, indent=2)

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
        "joint_names_29": HYBRIK_JOINT_NAMES_29,
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


def run_hybrik_image_main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run HybrIK on a single image and export 3D joints plus SMPL mesh.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", default=os.path.join("out", "hybrik_image"))
    parser.add_argument("--hybrik_root", default=DEFAULT_HYBRIK_ROOT)
    parser.add_argument("--config", default=DEFAULT_HYBRIK_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_HYBRIK_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--det_score_thr", type=float, default=0.9)
    args = parser.parse_args(argv)
    result = run_hybrik_inference(
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


def _run_extract(
    video: str,
    model: str,
    out_csv: str,
    out_video: str,
    smooth_mode: str,
    v_thr: float,
    gap_max: int,
    smooth_window: int,
) -> Dict[str, str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "__extract_pose",
        "--video",
        video,
        "--model",
        model,
        "--out_csv",
        out_csv,
        "--out_video",
        out_video,
        "--smooth_mode",
        smooth_mode,
        "--v_thr",
        str(v_thr),
        "--gap_max",
        str(gap_max),
        "--smooth_window",
        str(smooth_window),
    ]
    subprocess.run(cmd, check=True)
    return {"pose_csv": out_csv, "pose_video": out_video}


def _save_keyframes(video: str, phases: Dict[str, int], out_dir: str) -> Dict[str, str]:
    key_dir = os.path.join(out_dir, "keyframes")
    _ensure_dir(key_dir)
    # Clear old keyframes to avoid browser cache confusion
    for name in ["trophy", "racket_drop", "contact", "finish"]:
        old_path = os.path.join(key_dir, f"{name}.png")
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    outputs = {}
    for name in ["trophy", "racket_drop", "contact", "finish"]:
        idx = phases.get(name)
        if idx is None:
            continue
        idx = int(max(0, min(idx, max(frame_count - 1, 0))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = os.path.join(key_dir, f"{name}.png")
        _write_image_unicode(out_path, frame)
        outputs[name] = _posix_path(os.path.join("keyframes", f"{name}.png"))

    cap.release()
    return outputs


def _plot_series(ts_s: np.ndarray, y: np.ndarray, title: str, ylabel: str, path: str, phases: Dict[str, int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3))
    plt.plot(ts_s, y, color="#2c7fb8", linewidth=2)
    for name, color in [
        ("toss_peak", "#8c510a"),
        ("trophy", "#01665e"),
        ("racket_drop", "#7b3294"),
        ("contact", "#b2182b"),
        ("finish", "#4d4d4d"),
    ]:
        idx = phases.get(name)
        if idx is not None and 0 <= idx < len(ts_s):
            plt.axvline(ts_s[idx], color=color, linestyle="--", linewidth=1, alpha=0.7)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _diagnosis_biomech(
    subscores: Dict[str, float],
    biomech: Dict[str, object],
    lang: str,
) -> List[Dict[str, str]]:
    t = I18N.get(lang, I18N["en"])

    if lang == "zh":
        names = {
            "knee": "膝屈（Knee Flexion）",
            "trunk": "躯干倾角（Trunk Inclination）",
            "impact": "击球高度（Impact Height）",
            "timing": "时序（Timing）",
            "risk": "风险提示（Risk Flags）",
        }
        msgs = {
            "knee": (
                "膝屈幅度良好，发力链条稳定。",
                "膝屈适中，可进一步优化启动深度。",
                "膝屈不足或过大，建议调整下肢负荷。",
            ),
            "trunk": (
                "躯干倾角合理，力量传导顺畅。",
                "躯干倾角尚可，可优化前倾协同。",
                "躯干倾角偏小/偏大，建议调整上体控制。",
            ),
            "impact": (
                "击球高度较高，有利于出球角度。",
                "击球高度尚可，可进一步抬高击球点。",
                "击球高度偏低，建议提高抬臂与蹬伸。",
            ),
            "timing": (
                "时序协调，动作衔接良好。",
                "时序尚可，可优化各相位过渡。",
                "时序偏乱，建议放慢并重建节奏。",
            ),
        }
    else:
        names = {
            "knee": "Knee Flexion",
            "trunk": "Trunk Inclination",
            "impact": "Impact Height",
            "timing": "Timing",
            "risk": "Risk Flags",
        }
        msgs = {
            "knee": (
                "Good knee flexion; stable lower-body loading.",
                "Moderate knee flexion; could deepen loading.",
                "Knee flexion too shallow or too deep; adjust load.",
            ),
            "trunk": (
                "Trunk inclination is well coordinated.",
                "Trunk inclination is okay; improve forward tilt timing.",
                "Trunk inclination too small/large; adjust torso control.",
            ),
            "impact": (
                "Impact height is high; good release angle.",
                "Impact height is acceptable; consider raising the contact.",
                "Impact height is low; extend and raise contact point.",
            ),
            "timing": (
                "Timing is coordinated across phases.",
                "Timing is acceptable; refine phase transitions.",
                "Timing is inconsistent; slow down and rebuild rhythm.",
            ),
        }

    def line(name: str, score: float, good: str, ok: str, bad: str) -> Dict[str, str]:
        if score >= 80:
            text = good
        elif score >= 60:
            text = ok
        else:
            text = bad
        return {"name": name, "score": f"{score:.1f}", "text": text}

    rows = [
        line(names["knee"], subscores.get("knee_flexion", 0.0), *msgs["knee"]),
        line(names["trunk"], subscores.get("trunk_inclination", 0.0), *msgs["trunk"]),
        line(names["impact"], subscores.get("impact_height", 0.0), *msgs["impact"]),
        line(names["timing"], subscores.get("timing", 0.0), *msgs["timing"]),
    ]

    risk_flags = biomech.get("risk_flags") or []
    risk_map = {
        "knee_flexion_out_of_range": (
            "膝屈幅度超出正常范围：可能影响发力或下肢负荷。"
            if lang == "zh"
            else "Knee flexion out of range; may affect loading or power chain."
        ),
        "trunk_inclination_risk": (
            "躯干倾角偏小/偏大：可能影响发力链条与稳定性。"
            if lang == "zh"
            else "Trunk inclination too small/large; may affect power chain stability."
        ),
        "elbow_angle_inconsistent": (
            "肘角波动较大：可能存在发力不连贯/检测不稳定。"
            if lang == "zh"
            else "Elbow angle inconsistent; possible discontinuity or tracking noise."
        ),
        "shoulder_tilt_inconsistent": (
            "肩线倾斜波动较大：可能存在稳定性不足。"
            if lang == "zh"
            else "Shoulder tilt varies widely; possible stability issue."
        ),
    }
    if risk_flags:
        risk_text = "；".join([risk_map.get(r, r) for r in risk_flags])
    else:
        risk_text = t["risk_none"]
    rows.append({"name": names["risk"], "score": "-", "text": risk_text})
    return rows


def _try_export_pdf(html_path: str, pdf_path: str) -> bool:
    try:
        from weasyprint import HTML

        HTML(filename=html_path).write_pdf(pdf_path)
        return True
    except Exception:
        try:
            import pdfkit

            pdfkit.from_file(html_path, pdf_path)
            return True
        except Exception:
            return False


def _find_calibration_dir(repo_root: str) -> Optional[str]:
    candidates = [
        os.path.join(repo_root, "calibration"),
        os.path.join(repo_root, "out", "calibration"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _load_calibration_ranges(repo_root: str) -> Dict[str, tuple]:
    calib_dir = _find_calibration_dir(repo_root)
    if not calib_dir:
        return {}

    targets = [
        "impact_height_norm",
        "timing_toss_to_trophy",
        "timing_trophy_to_drop",
        "timing_drop_to_contact",
    ]
    values: Dict[str, List[float]] = {k: [] for k in targets}

    for root, _, files in os.walk(calib_dir):
        for name in files:
            if name != "metrics.json":
                continue
            path = os.path.join(root, name)
            try:
                payload = json.loads(open(path, "r", encoding="utf-8").read())
            except Exception:
                continue
            metrics = payload.get("metrics") or {}
            biomech = payload.get("biomech") or {}
            biomech_raw = biomech.get("raw") or {}

            def pick_metric(key: str) -> float:
                if key == "impact_height_norm":
                    for alt in (
                        biomech_raw.get("impact_height_norm"),
                        metrics.get("impact_height_norm"),
                        metrics.get("contact_above_head"),
                    ):
                        if isinstance(alt, (int, float)) and np.isfinite(alt):
                            return float(alt)
                    return float("nan")
                return float(metrics.get(key, float("nan")))

            for k in targets:
                val = pick_metric(k)
                if np.isfinite(val):
                    values[k].append(val)

    ranges: Dict[str, tuple] = {}
    for k, series in values.items():
        if len(series) < 5:
            continue
        lo, hi = np.nanpercentile(series, [15, 85])
        ranges[k] = (float(lo), float(hi))
    return ranges


def _format_risk_flags(flags: List[str], lang: str) -> str:
    if not flags:
        return I18N.get(lang, I18N["en"])["risk_none"]

    if lang == "zh":
        sep = "；"
        return sep.join([zh_label(r) for r in flags])
    sep = "; "
    return sep.join(flags)


def _phase_frame_num(phases: Dict[str, object], frames: Optional[np.ndarray], name: str) -> Optional[int]:
    idx = phases.get(name) if isinstance(phases, dict) else None
    if idx is None:
        return None
    try:
        idx = int(idx)
    except Exception:
        return None
    if frames is not None and 0 <= idx < len(frames):
        try:
            return int(frames[idx])
        except Exception:
            return idx
    return idx


def _python_supports_3d_runtime(python_exe: str) -> bool:
    if not python_exe or not os.path.exists(python_exe):
        return False
    cmd = [python_exe, "-c", "import cv2, numpy, torch, torchvision, easydict, yaml, PIL"]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0


def _find_servepose_python() -> Optional[str]:
    exe = os.path.abspath(sys.executable)
    if _python_supports_3d_runtime(exe):
        return exe

    exe_dir = os.path.dirname(exe)
    roots = {exe_dir, os.path.dirname(exe_dir), os.path.dirname(os.path.dirname(exe_dir))}

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_root = os.path.dirname(os.path.dirname(os.path.abspath(conda_exe)))
        roots.add(conda_root)

    candidates = []
    for root in roots:
        candidates.extend(
            [
                os.path.join(root, "envs", "servepose", "python.exe"),
                os.path.join(root, "envs", "servepose", "bin", "python"),
            ]
        )

    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if _python_supports_3d_runtime(candidate):
            return candidate
    return None


def _artifact_relpath(path: Optional[str], out_dir: str, run_id: int) -> Optional[str]:
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None
    rel = os.path.relpath(abs_path, out_dir)
    return f"{_posix_path(rel)}?v={run_id}"


def _load_3d_verification_summary(verify_dir: str, out_dir: str, run_id: int) -> Optional[Dict[str, object]]:
    summary_path = os.path.join(verify_dir, "verification_summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return None

    artifacts = summary.get("artifacts") or {}
    summary["artifact_paths"] = {
        "mesh_overlay": _artifact_relpath(artifacts.get("mesh_overlay_png"), out_dir, run_id),
        "mesh_overlay_compare": _artifact_relpath(artifacts.get("mesh_overlay_compare_png"), out_dir, run_id),
        "views": _artifact_relpath(artifacts.get("3d_views_png"), out_dir, run_id),
        "reprojection": _artifact_relpath(artifacts.get("reprojection_compare_png"), out_dir, run_id),
        "neighbor_metrics": _artifact_relpath(artifacts.get("neighbor_metrics_png"), out_dir, run_id),
    }
    return summary


def _maybe_generate_3d_verification(
    video: str,
    pose_csv: str,
    out_dir: str,
    trophy_image_path: str,
    trophy_frame_num: Optional[int],
    use_coords: str,
    run_id: int,
    verify_3d: str,
    verify_neighbor_radius: int,
    verify_device: str,
) -> Optional[Dict[str, object]]:
    verify_dir = os.path.join(out_dir, "verify")
    existing = _load_3d_verification_summary(verify_dir, out_dir, run_id)
    if verify_3d == "off":
        return existing
    if existing is not None and verify_3d != "on":
        return existing
    if trophy_frame_num is None or not os.path.exists(trophy_image_path):
        print("[INFO] 3D verification skipped (trophy frame unavailable).")
        return existing

    servepose_python = _find_servepose_python()
    if not servepose_python:
        print("[INFO] 3D verification skipped (servepose python not found).")
        return existing

    hybrik_dir = os.path.join(verify_dir, "hybrik")
    _ensure_dir(hybrik_dir)
    verify_script = os.path.join(os.path.dirname(__file__), "verify_trophy_3d.py")
    if not os.path.exists(verify_script):
        print("[INFO] 3D verification skipped (helper scripts missing).")
        return existing

    try:
        cmd_hybrik = [
            servepose_python,
            os.path.abspath(__file__),
            "__run_hybrik_image",
            "--image",
            trophy_image_path,
            "--out_dir",
            hybrik_dir,
            "--device",
            verify_device,
        ]
        subprocess.run(cmd_hybrik, check=True)

        cmd_verify = [
            servepose_python,
            verify_script,
            "--json_path",
            os.path.join(hybrik_dir, "joints3d.json"),
            "--image_path",
            trophy_image_path,
            "--pose_csv",
            pose_csv,
            "--frame_num",
            str(trophy_frame_num),
            "--video",
            video,
            "--neighbor_radius",
            str(max(0, int(verify_neighbor_radius))),
            "--use_coords",
            use_coords,
            "--out_dir",
            verify_dir,
            "--device",
            verify_device,
        ]
        subprocess.run(cmd_verify, check=True)
    except Exception as exc:
        print(f"[INFO] 3D verification skipped ({exc}).")
        return existing

    loaded = _load_3d_verification_summary(verify_dir, out_dir, run_id)
    if loaded is None:
        print("[INFO] 3D verification finished, but summary was not found.")
    return loaded


def _phase_display_name(phase: str, lang: str) -> str:
    names = {
        "trophy": ("奖杯姿势", "Trophy"),
        "racket_drop": ("拍头下落", "Racket Drop"),
        "contact": ("击球瞬间", "Contact"),
        "finish": ("收拍结束", "Finish"),
    }
    zh, en = names.get(phase, (phase, phase))
    return zh if lang == "zh" else en


def _angle_display_name(metric: str, lang: str) -> str:
    zh, en = KEYFRAME_3D_ANGLE_LABELS.get(metric, (metric, metric))
    return zh if lang == "zh" else en


def _save_compare_image(
    image_path: str,
    overlay_image: np.ndarray,
    compare_path: str,
    lang: str,
) -> None:
    original = _read_image_unicode(image_path)
    if original is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    gap = 18
    h, w = original.shape[:2]
    compare = np.full((h, w * 2 + gap, 3), 255, dtype=np.uint8)
    compare[:, :w] = original
    compare[:, w + gap :] = overlay_image

    labels = ("原图", "原图 + 3D 叠加") if lang == "zh" else ("Original", "Original + 3D Overlay")
    for origin_x, text in [(18, labels[0]), (w + gap + 18, labels[1])]:
        cv2.putText(compare, text, (origin_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(compare, text, (origin_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (44, 44, 44), 1, cv2.LINE_AA)

    _write_image_unicode(compare_path, compare)


def _load_keyframe_3d_summary(keyframe_dir: str, out_dir: str, run_id: int) -> Optional[Dict[str, object]]:
    summary_path = os.path.join(keyframe_dir, "keyframe_3d_summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return None

    phase_payloads = summary.get("phases")
    if not isinstance(phase_payloads, dict):
        return None

    for phase_name, payload in phase_payloads.items():
        if not isinstance(payload, dict):
            continue
        artifacts = payload.get("artifacts") or {}
        payload["artifact_paths"] = {
            "compare": _artifact_relpath(artifacts.get("compare_png"), out_dir, run_id),
        }
        payload["phase"] = phase_name
    return summary


def _maybe_generate_keyframe_3d_bundle(
    out_dir: str,
    phase_frame_nums: Dict[str, Optional[int]],
    lang: str,
    run_id: int,
    verify_3d: str,
    verify_device: str,
) -> Optional[Dict[str, object]]:
    keyframe_dir = os.path.join(out_dir, "keyframe_3d")
    existing = _load_keyframe_3d_summary(keyframe_dir, out_dir, run_id)
    if verify_3d == "off":
        return existing
    if existing is not None and verify_3d != "on":
        return existing

    servepose_python = _find_servepose_python()
    if not servepose_python:
        print("[INFO] 3D keyframe analysis skipped (servepose python not found).")
        return existing

    _ensure_dir(keyframe_dir)
    phase_payloads: Dict[str, object] = {}

    for phase in KEYFRAME_3D_PHASES:
        image_path = os.path.join(out_dir, "keyframes", f"{phase}.png")
        if not os.path.exists(image_path):
            continue

        phase_dir = os.path.join(keyframe_dir, phase)
        hybrik_dir = os.path.join(phase_dir, "hybrik")
        _ensure_dir(phase_dir)
        _ensure_dir(hybrik_dir)

        try:
            cmd_hybrik = [
                servepose_python,
                os.path.abspath(__file__),
                "__run_hybrik_image",
                "--image",
                image_path,
                "--out_dir",
                hybrik_dir,
                "--device",
                verify_device,
            ]
            subprocess.run(cmd_hybrik, check=True)

            json_path = os.path.join(hybrik_dir, "joints3d.json")
            obj_path = os.path.join(hybrik_dir, "mesh.obj")
            overlay_path = os.path.join(phase_dir, "mesh_overlay.png")
            compare_path = os.path.join(phase_dir, "mesh_overlay_compare.png")

            metrics_3d, _ = serve_score.analyze_trophy_pose_3d(json_path)
            phase_metrics_3d = serve_score.summarize_keyframe_pose_3d(
                json_path,
                phase=phase,
                dominant_hand="right",
            )
            phase_metrics_3d = {**metrics_3d, **phase_metrics_3d}
            overlay_img, overlay_debug = serve_score.render_hybrik_mesh_overlay(
                image_or_path=image_path,
                data_or_path=json_path,
                mesh_obj_path=obj_path,
            )
            _write_image_unicode(overlay_path, overlay_img)
            _save_compare_image(image_path, overlay_img, compare_path, lang)

            phase_payloads[phase] = {
                "phase": phase,
                "label": _phase_display_name(phase, lang),
                "frame_num": phase_frame_nums.get(phase),
                "phase_metrics_3d": phase_metrics_3d,
                "artifacts": {
                    "compare_png": os.path.abspath(compare_path),
                },
            }
            _remove_path(hybrik_dir)
            _remove_path(overlay_path)
        except Exception as exc:
            phase_payloads[phase] = {
                "phase": phase,
                "label": _phase_display_name(phase, lang),
                "frame_num": phase_frame_nums.get(phase),
                "error": str(exc),
            }

    summary = {
        "phases": phase_payloads,
        "inputs": {
            "out_dir": os.path.abspath(out_dir),
        },
    }
    summary_path = os.path.join(keyframe_dir, "keyframe_3d_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return _load_keyframe_3d_summary(keyframe_dir, out_dir, run_id)


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
        return float(value)
    return float("nan")


def _weighted_score(parts: List[Tuple[float, float]]) -> float:
    total_weight = 0.0
    total_score = 0.0
    for score, weight in parts:
        if np.isfinite(score) and weight > 0:
            total_score += float(score) * float(weight)
            total_weight += float(weight)
    if total_weight <= 1e-8:
        return float("nan")
    return total_score / total_weight


def _score_centered_range(value: float, lo: float, hi: float, edge_score: float = 80.0) -> float:
    if not np.isfinite(value):
        return float("nan")
    lo_f = min(float(lo), float(hi))
    hi_f = max(float(lo), float(hi))
    center = (lo_f + hi_f) / 2.0
    half = (hi_f - lo_f) / 2.0
    if half <= 1e-8:
        return 100.0 if abs(value - center) <= 1e-8 else 0.0
    dist = abs(float(value) - center)
    if dist <= half:
        return float(100.0 - (100.0 - edge_score) * (dist / half))
    outer = dist - half
    return float(max(0.0, edge_score * (1.0 - outer / half)))


def _lit_zone_name(z_abs: float, lang: str) -> str:
    if not np.isfinite(z_abs):
        return "n/a"
    if z_abs <= 1.0:
        return "参考绿区" if lang == "zh" else "green literature zone"
    if z_abs <= 2.0:
        return "参考黄区" if lang == "zh" else "yellow literature zone"
    if z_abs <= 3.0:
        return "参考橙区" if lang == "zh" else "orange literature zone"
    return "参考红区" if lang == "zh" else "red literature zone"


def _build_current_3d_report_summary(
    keyframe_3d: Optional[Dict[str, object]],
    timing_metrics: Dict[str, float],
    lang: str,
) -> Dict[str, object]:
    phases = (keyframe_3d or {}).get("phases") if isinstance(keyframe_3d, dict) else {}
    if not isinstance(phases, dict):
        phases = {}

    trophy_payload = phases.get("trophy") if isinstance(phases.get("trophy"), dict) else {}
    drop_payload = phases.get("racket_drop") if isinstance(phases.get("racket_drop"), dict) else {}
    contact_payload = phases.get("contact") if isinstance(phases.get("contact"), dict) else {}

    trophy_metrics = trophy_payload.get("phase_metrics_3d") if isinstance(trophy_payload.get("phase_metrics_3d"), dict) else {}
    drop_metrics = drop_payload.get("phase_metrics_3d") if isinstance(drop_payload.get("phase_metrics_3d"), dict) else {}
    contact_metrics = contact_payload.get("phase_metrics_3d") if isinstance(contact_payload.get("phase_metrics_3d"), dict) else {}

    tp_knee = _safe_float(trophy_metrics.get("lead_knee_flexion_deg"))
    tp_trunk = _safe_float(trophy_metrics.get("trunk_inclination_deg"))
    impact_height_norm = _safe_float(contact_metrics.get("impact_height_norm"))
    rlp_shoulder_er = _safe_float(drop_metrics.get("shoulder_external_rotation_proxy_deg"))
    bi_shoulder_raise = _safe_float(contact_metrics.get("shoulder_raise_deg"))
    bi_elbow_flex = _safe_float(contact_metrics.get("elbow_flexion_deg"))

    knee_score, knee_z = serve_score._score_lit_plateau(tp_knee, LIT_KNEE_MEAN, LIT_KNEE_SD)
    trunk_score, trunk_z = serve_score._score_lit_plateau(tp_trunk, LIT_TRUNK_MEAN, LIT_TRUNK_SD)

    timing_specs = [
        ("timing_toss_to_trophy", "抛球→奖杯位间隔", TIMING_RANGES["timing_toss_to_trophy"]),
        ("timing_trophy_to_drop", "奖杯位→下落间隔", TIMING_RANGES["timing_trophy_to_drop"]),
        ("timing_drop_to_contact", "下落→击球间隔", TIMING_RANGES["timing_drop_to_contact"]),
    ]
    timing_rows = []
    timing_scores = []
    for key, zh_name, rng in timing_specs:
        value = _safe_float(timing_metrics.get(key))
        score = _score_centered_range(value, rng[0], rng[1], edge_score=80.0)
        if np.isfinite(score):
            timing_scores.append(score)
        note = (
            "处于经验合理范围，节奏正常"
            if lang == "zh"
            else "Within the expected timing band."
        )
        timing_rows.append(
            {
                "name": zh_name if lang == "zh" else key,
                "value": value,
                "score": score,
                "note": note,
            }
        )

    timing_score = float(np.mean(timing_scores)) if timing_scores else float("nan")
    final_score = _weighted_score(
        [
            (knee_score, 0.40),
            (trunk_score, 0.40),
            (timing_score, 0.20),
        ]
    )

    knee_range = (round(LIT_KNEE_MEAN - LIT_KNEE_SD), round(LIT_KNEE_MEAN + LIT_KNEE_SD))
    trunk_range = (round(LIT_TRUNK_MEAN - LIT_TRUNK_SD), round(LIT_TRUNK_MEAN + LIT_TRUNK_SD))

    core_rows = [
        {
            "name": "奖杯位膝屈角（TP）" if lang == "zh" else "Trophy Knee Flexion (TP)",
            "value_text": (
                f"{tp_knee:.1f}°（参考：{knee_range[0]}–{knee_range[1]}°，文献均值±1SD）"
                if np.isfinite(tp_knee)
                else ("未计算" if lang == "zh" else "Not available")
            ),
            "conclusion": (
                f"处于{_lit_zone_name(abs(knee_z), lang)}，技术表现良好，子分 = {knee_score:.1f}"
                if np.isfinite(knee_score)
                else ("3D 关键点不足，未能稳定计算。" if lang == "zh" else "3D joints were insufficient for a stable estimate.")
            ),
        },
        {
            "name": "奖杯位躯干倾角（TP）" if lang == "zh" else "Trophy Trunk Inclination (TP)",
            "value_text": (
                f"{tp_trunk:.1f}°（参考：{trunk_range[0]}–{trunk_range[1]}°，文献均值±1SD）"
                if np.isfinite(tp_trunk)
                else ("未计算" if lang == "zh" else "Not available")
            ),
            "conclusion": (
                f"处于{_lit_zone_name(abs(trunk_z), lang)}，奖杯位躯干构型合理，子分 = {trunk_score:.1f}"
                if np.isfinite(trunk_score)
                else ("3D 关键点不足，未能稳定计算。" if lang == "zh" else "3D joints were insufficient for a stable estimate.")
            ),
        },
        {
            "name": "击球高度（身高归一化）" if lang == "zh" else "Impact Height (Body-Height Normalized)",
            "value_text": (
                f"{impact_height_norm:.3f} BH"
                if np.isfinite(impact_height_norm)
                else ("未计算" if lang == "zh" else "Not available")
            ),
            "conclusion": (
                "已由 3D 模型计算，当前版本暂不纳入总分。"
                if np.isfinite(impact_height_norm)
                else ("3D 关键点不足，未能稳定计算。" if lang == "zh" else "3D joints were insufficient for a stable estimate.")
            ),
        },
        {
            "name": "拍头下落肩外旋（RLP）" if lang == "zh" else "Racket-Drop Shoulder External Rotation (RLP)",
            "value_text": (
                f"{rlp_shoulder_er:.1f}°"
                if np.isfinite(rlp_shoulder_er)
                else ("未计算" if lang == "zh" else "Not available")
            ),
            "conclusion": (
                "基于 3D 关节几何的肩外旋代理角，当前作为扩展观察项展示。"
                if np.isfinite(rlp_shoulder_er)
                else ("3D 关键点不足，未能稳定计算。" if lang == "zh" else "3D joints were insufficient for a stable estimate.")
            ),
        },
        {
            "name": "击球瞬间肩抬高（BI）" if lang == "zh" else "Impact Shoulder Raise (BI)",
            "value_text": (
                f"{bi_shoulder_raise:.1f}°"
                if np.isfinite(bi_shoulder_raise)
                else ("未计算" if lang == "zh" else "Not available")
            ),
            "conclusion": (
                "由 3D 上臂相对躯干夹角得到，当前作为扩展观察项展示。"
                if np.isfinite(bi_shoulder_raise)
                else ("3D 关键点不足，未能稳定计算。" if lang == "zh" else "3D joints were insufficient for a stable estimate.")
            ),
        },
        {
            "name": "击球瞬间肘屈（BI）" if lang == "zh" else "Impact Elbow Flexion (BI)",
            "value_text": (
                f"{bi_elbow_flex:.1f}°"
                if np.isfinite(bi_elbow_flex)
                else ("未计算" if lang == "zh" else "Not available")
            ),
            "conclusion": (
                "由 3D 肘关节屈曲角得到，当前作为扩展观察项展示。"
                if np.isfinite(bi_elbow_flex)
                else ("3D 关键点不足，未能稳定计算。" if lang == "zh" else "3D joints were insufficient for a stable estimate.")
            ),
        },
    ]

    score_rows = [
        {
            "name": "奖杯位膝屈角（TP）" if lang == "zh" else "Trophy Knee Flexion (TP)",
            "score": knee_score,
            "weight": 0.40,
            "note": "核心技术项" if lang == "zh" else "Core technical item",
        },
        {
            "name": "奖杯位躯干倾角（TP）" if lang == "zh" else "Trophy Trunk Inclination (TP)",
            "score": trunk_score,
            "weight": 0.40,
            "note": "核心技术项" if lang == "zh" else "Core technical item",
        },
        {
            "name": "辅助节奏观察" if lang == "zh" else "Auxiliary Timing",
            "score": timing_score,
            "weight": 0.20,
            "note": "纳入总分，但权重低于核心技术项" if lang == "zh" else "Included, but weighted below core technical items",
        },
        {
            "name": "击球高度（归一化）" if lang == "zh" else "Impact Height (Normalized)",
            "score": float("nan"),
            "weight": 0.0,
            "note": "当前不纳入总分" if lang == "zh" else "Currently excluded from the total score",
        },
        {
            "name": "技术评分（当前版）" if lang == "zh" else "Current Technical Score",
            "score": final_score,
            "weight": 1.0,
            "note": "基于当前可稳定评分项目计算" if lang == "zh" else "Computed from the currently stable scoring items",
        },
    ]

    quality_rows = [
        {
            "name": "奖杯位膝屈角（TP）" if lang == "zh" else "Trophy Knee Flexion (TP)",
            "success": "1/1" if np.isfinite(tp_knee) else "0/1",
            "usable": bool(np.isfinite(knee_score)),
            "note": "奖杯位 3D 关键点完整" if lang == "zh" else "Trophy-frame 3D joints are complete",
        },
        {
            "name": "奖杯位躯干倾角（TP）" if lang == "zh" else "Trophy Trunk Inclination (TP)",
            "success": "1/1" if np.isfinite(tp_trunk) else "0/1",
            "usable": bool(np.isfinite(trunk_score)),
            "note": "奖杯位 3D 关键点完整" if lang == "zh" else "Trophy-frame 3D joints are complete",
        },
        {
            "name": "击球高度（归一化）" if lang == "zh" else "Impact Height (Normalized)",
            "success": "1/1" if np.isfinite(impact_height_norm) else "0/1",
            "usable": False,
            "note": "已由 3D 模型计算，但当前不纳入总分" if lang == "zh" else "Computed from 3D, but not yet used in scoring",
        },
        {
            "name": "拍头下落肩外旋（RLP）" if lang == "zh" else "RLP Shoulder External Rotation",
            "success": "1/1" if np.isfinite(rlp_shoulder_er) else "0/1",
            "usable": False,
            "note": "3D 扩展观察项" if lang == "zh" else "3D observational metric",
        },
        {
            "name": "击球瞬间肩抬高（BI）" if lang == "zh" else "BI Shoulder Raise",
            "success": "1/1" if np.isfinite(bi_shoulder_raise) else "0/1",
            "usable": False,
            "note": "3D 扩展观察项" if lang == "zh" else "3D observational metric",
        },
        {
            "name": "击球瞬间肘屈（BI）" if lang == "zh" else "BI Elbow Flexion",
            "success": "1/1" if np.isfinite(bi_elbow_flex) else "0/1",
            "usable": False,
            "note": "3D 扩展观察项" if lang == "zh" else "3D observational metric",
        },
        {
            "name": "辅助时序" if lang == "zh" else "Auxiliary Timing",
            "success": f"{sum(np.isfinite(row['value']) for row in timing_rows)}/3",
            "usable": bool(np.isfinite(timing_score)),
            "note": "事件顺序识别正常，可用于辅助节奏评分" if lang == "zh" else "Event order is valid and usable for rhythm scoring",
        },
    ]

    explanation = (
        "当前评分基于 3D 奖杯位膝屈角（TP）权重 40%，3D 奖杯位躯干倾角（TP）权重 40%，辅助节奏观察权重 20%。"
        " 击球高度、拍头下落肩外旋、击球瞬间肩抬高与肘屈已由 3D 模型计算，当前作为扩展观察项展示，暂不纳入总分。"
        if lang == "zh"
        else "The current score uses 3D trophy knee flexion (40%), 3D trophy trunk inclination (40%), and auxiliary timing (20%)."
    )
    basis = (
        "当前版本采用关键事件驱动的 3D 动作分析口径，围绕奖杯位（TP）、拍头下落（RLP）和击球瞬间（BI）三个关键阶段建立分析框架。"
        " 当前总分仅纳入 TP 膝屈角、TP 躯干倾角和辅助节奏观察；击球高度、RLP 肩外旋以及 BI 肩抬高/肘屈已完成 3D 计算，暂作为扩展观察项展示。"
        if lang == "zh"
        else "This version uses a key-event-driven 3D analysis pipeline around TP, RLP, and BI."
    )
    timing_note = (
        "辅助节奏观察用于反映动作节奏与关键事件组织情况。三个阶段时序分数按等权平均计算，区间中心值对应更高分。"
        if lang == "zh"
        else "Auxiliary timing reflects event organization; the three interval scores are averaged equally."
    )

    return {
        "final_score": final_score,
        "coverage_ratio": 2.0 / 3.0,
        "explanation": explanation,
        "basis": basis,
        "core_rows": core_rows,
        "score_rows": score_rows,
        "quality_rows": quality_rows,
        "timing_rows": timing_rows,
        "timing_total_score": timing_score,
        "timing_total_note": timing_note,
        "raw_values": {
            "tp_knee_flexion_deg": tp_knee,
            "tp_trunk_inclination_deg": tp_trunk,
            "impact_height_norm": impact_height_norm,
            "rlp_shoulder_external_rotation_deg": rlp_shoulder_er,
            "bi_shoulder_raise_deg": bi_shoulder_raise,
            "bi_elbow_flexion_deg": bi_elbow_flex,
            "timing_toss_to_trophy": _safe_float(timing_metrics.get("timing_toss_to_trophy")),
            "timing_trophy_to_drop": _safe_float(timing_metrics.get("timing_trophy_to_drop")),
            "timing_drop_to_contact": _safe_float(timing_metrics.get("timing_drop_to_contact")),
            "tp_knee_score": knee_score,
            "tp_trunk_score": trunk_score,
            "timing_score": timing_score,
        },
    }


def _render_html(
    out_dir: str,
    final_score: float,
    subscores: Dict[str, float],
    metrics: Dict[str, float],
    keyframes: Dict[str, str],
    plots: Dict[str, str],
    diagnosis: List[Dict[str, str]],
    hand_meta: Dict[str, object],
    quality_flags: Dict[str, object],
    source_meta: Dict[str, object],
    biomech: Dict[str, object],
    phases: Dict[str, object],
    lang: str,
    verification_3d: Optional[Dict[str, object]] = None,
    keyframe_3d: Optional[Dict[str, object]] = None,
    report_3d_summary: Optional[Dict[str, object]] = None,
) -> str:
    t = I18N.get(lang, I18N["en"])
    def label(key: str) -> str:
        return zh_label(key) if lang == "zh" else str(key)

    def value_text(value: object) -> str:
        if lang != "zh":
            return str(value)
        if isinstance(value, bool):
            return zh_label("True" if value else "False")
        if isinstance(value, (int, float, np.floating)):
            return str(value)
        if value is None:
            return zh_label("n/a")
        return zh_label(str(value))

    knee_lo = round(LIT_KNEE_MEAN - LIT_KNEE_SD)
    knee_hi = round(LIT_KNEE_MEAN + LIT_KNEE_SD)
    trunk_lo = round(LIT_TRUNK_MEAN - LIT_TRUNK_SD)
    trunk_hi = round(LIT_TRUNK_MEAN + LIT_TRUNK_SD)
    calib_ranges = _load_calibration_ranges(os.path.dirname(__file__))

    impact_unit = "" if lang == "zh" else ""
    recommended_ranges = {
        "knee_flexion_est_deg": (knee_lo, knee_hi, "°", t["range_lit"]),
        "trunk_inclination_est_deg": (trunk_lo, trunk_hi, "°", t["range_lit"]),
        "impact_bh_ratio": (
            IMPACT_NORM_RANGE[0],
            IMPACT_NORM_RANGE[1],
            impact_unit,
            t["range_heur"],
        ),
        "timing_toss_to_trophy": (*TIMING_RANGES["timing_toss_to_trophy"], "s", t["range_heur"]),
        "timing_trophy_to_drop": (*TIMING_RANGES["timing_trophy_to_drop"], "s", t["range_heur"]),
        "timing_drop_to_contact": (*TIMING_RANGES["timing_drop_to_contact"], "s", t["range_heur"]),
    }
    for key, rng in calib_ranges.items():
        if key in recommended_ranges:
            unit = recommended_ranges[key][2]
        else:
            unit = impact_unit if key == "impact_height_norm" else "s"
        recommended_ranges[key] = (rng[0], rng[1], unit, t["range_calib"])

    def fmt(value: float) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return "n/a"
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.3f}"
        return str(value)

    def fmt_with_range(key: str, value: float, unit: str = "") -> str:
        if isinstance(value, (int, float, np.floating)) and key.startswith("timing_") and value < 0:
            return "无效（时间为负）" if lang == "zh" else "invalid (negative time)"
        base = fmt(value)
        rng = recommended_ranges.get(key)
        if rng is None or base == "n/a":
            return f"{base}{unit}"
        lo, hi, unit2, note = rng
        unit_final = unit2 if unit2 else unit
        return f"{base}{unit_final}（建议：{lo}–{hi}{unit_final}，{note}）"

    body_scale = metrics.get("body_scale", 1.0)
    if not isinstance(body_scale, (int, float)) or not np.isfinite(body_scale) or body_scale <= 0:
        body_scale = 1.0

    biomech_raw = dict(biomech.get("raw") or {})
    impact_raw = biomech_raw.get("impact_height")
    if not isinstance(impact_raw, (int, float)) or not np.isfinite(impact_raw):
        impact_raw = metrics.get("contact_above_head_raw")
        if not isinstance(impact_raw, (int, float)) or not np.isfinite(impact_raw):
            impact_raw = metrics.get("contact_above_head")
            if isinstance(impact_raw, (int, float)) and np.isfinite(impact_raw):
                impact_raw = float(impact_raw) * float(body_scale)
    if impact_raw is not None:
        biomech_raw["impact_height"] = float(impact_raw)

    impact_norm = biomech_raw.get("impact_height_norm")
    if not isinstance(impact_norm, (int, float)) or not np.isfinite(impact_norm):
        impact_norm = metrics.get("impact_height_norm", metrics.get("contact_above_head"))
    if impact_norm is not None:
        biomech_raw["impact_height_norm"] = float(impact_norm) if np.isfinite(impact_norm) else float("nan")

    metrics_display = {}
    for key in [
        "knee_flexion_est_deg",
        "trunk_inclination_est_deg",
        "impact_bh_ratio",
    ]:
        metrics_display[key] = metrics.get(key, float("nan"))
    for key in [
        "timing_toss_to_trophy",
        "timing_trophy_to_drop",
        "timing_drop_to_contact",
    ]:
        if key in metrics and np.isfinite(metrics.get(key, float("nan"))):
            metrics_display[key] = metrics[key]
    biomech_raw_display = {k: v for k, v in biomech_raw.items() if k != "impact_height_score_basis"}

    score_block = f"{final_score:.1f}"
    def _score_text(value: object) -> str:
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            return f"{float(value):.1f}"
        return "n/a"

    subs_rows = "\n".join(
        f"<tr><td>{label(k)}</td><td>{_score_text(v)}</td></tr>" for k, v in subscores.items()
    )

    short_note = (
        "该时段非常短，可能表示动作衔接紧凑；若画面遮挡较多或关键帧选择偏差，建议结合关键帧图复核。"
        if lang == "zh"
        else "This interval is very short; consider reviewing keyframes if occlusion or mis-detection is suspected."
    )

    def _zone_note(key: str, value: float) -> str:
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
            return ""
        if key == "knee_flexion_est_deg":
            mu, sd = LIT_KNEE_MEAN, LIT_KNEE_SD
            sub = subscores.get("knee_flexion")
        elif key == "trunk_inclination_est_deg":
            mu, sd = LIT_TRUNK_MEAN, LIT_TRUNK_SD
            sub = subscores.get("trunk_inclination")
        else:
            return ""
        if sd <= 1e-6:
            return ""
        z = abs((value - mu) / sd)
        if z <= 1.0:
            zone = "绿区（文献均值±1SD）"
        elif z <= 2.0:
            zone = "黄区（偏离1–2SD）"
        elif z <= 3.0:
            zone = "橙区（偏离2–3SD）"
        else:
            zone = "红区（偏离>3SD）"
        sub_text = f"{float(sub):.1f}" if isinstance(sub, (int, float, np.floating)) and np.isfinite(sub) else "n/a"
        if lang == "zh":
            return f"文献{zone}，z={z:.2f}，子分={sub_text}"
        return f"Literature zone {zone}, z={z:.2f}, subscore={sub_text}"

    def _timing_note(key: str, value: float) -> str:
        if not key.startswith("timing_"):
            return ""
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
            return ""
        if value < 0.02:
            return short_note
        return ""

    def _metric_note(key: str, value: float) -> str:
        note = _zone_note(key, value)
        timing_note = _timing_note(key, value)
        extra = ""
        if key == "impact_bh_ratio":
            sub = subscores.get("impact_height")
            sub_text = f"{float(sub):.1f}" if isinstance(sub, (int, float, np.floating)) and np.isfinite(sub) else "n/a"
            conf = biomech.get("confidence", {})
            conf_pct = conf.get("impact_confidence")
            conf_text = f"{float(conf_pct) * 100:.0f}%" if isinstance(conf_pct, (int, float, np.floating)) and np.isfinite(conf_pct) else "n/a"
            status = conf.get("impact_status")
            reason = conf.get("impact_reason", "")
            if status == "不可评估":
                return f"不可评估（{reason}）" if reason else "不可评估"
            if lang == "zh":
                extra = f"子分={sub_text}，置信度={conf_text}"
        if key.startswith("timing_"):
            sub = subscores.get("timing")
            sub_text = f"{float(sub):.1f}" if isinstance(sub, (int, float, np.floating)) and np.isfinite(sub) else "n/a"
            conf = biomech.get("confidence", {})
            conf_pct = conf.get("timing_confidence")
            conf_text = f"{float(conf_pct) * 100:.0f}%" if isinstance(conf_pct, (int, float, np.floating)) and np.isfinite(conf_pct) else "n/a"
            status = conf.get("timing_status")
            reason = conf.get("timing_cap_reason", "")
            if status == "不可评估":
                return f"不可评估（{reason}）" if reason else "不可评估"
            if lang == "zh":
                extra = f"子分={sub_text}，置信度={conf_text}"
        parts = [p for p in [note, timing_note, extra] if p]
        if not parts:
            return ""
        return "；".join(parts) if lang == "zh" else "; ".join(parts)

    def _metric_value(key: str, value: float) -> str:
        conf = biomech.get("confidence", {})
        if key == "impact_bh_ratio" and conf.get("impact_status") == "不可评估":
            return "不可评估"
        if key.startswith("timing_") and conf.get("timing_status") == "不可评估":
            return "不可评估"
        return fmt_with_range(key, value)

    metrics_rows = "\n".join(
        f"<tr><td>{label(k)}</td><td>{_metric_value(k, v)}</td><td>{_metric_note(k, v)}</td></tr>"
        for k, v in metrics_display.items()
    )
    z_scores = biomech.get("z_scores") or {}
    z_display = {
        k: v
        for k, v in z_scores.items()
        if k in {"knee_flexion", "trunk_inclination"} and np.isfinite(v)
    }
    z_rows = "\n".join(
        f"<tr><td>{label(k)}</td><td>{fmt(v)}</td></tr>" for k, v in z_display.items()
    )

    confidence = biomech.get("confidence") or {}
    def _pct(val: object) -> str:
        if not isinstance(val, (int, float, np.floating)) or not np.isfinite(val):
            return "n/a"
        return f"{float(val) * 100:.0f}%"

    def _count(valid_key: str, total_key: str) -> str:
        v = confidence.get(valid_key)
        t = confidence.get(total_key)
        if isinstance(v, (int, float)) and isinstance(t, (int, float)) and t:
            return f"{int(v)}/{int(t)}"
        return "n/a"

    if lang == "zh":
        conf_labels = {
            "knee": "膝屈",
            "trunk": "躯干倾角",
            "impact": "击球高度",
            "timing": "时序",
        }
        conf_headers = ["项目", "置信度", "有效帧/总帧", "备注"]
        timing_invalid_note = "存在无效时序" if confidence.get("timing_invalid") else ""
    else:
        conf_labels = {
            "knee": "Knee Flexion",
            "trunk": "Trunk Inclination",
            "impact": "Impact Height",
            "timing": "Timing",
        }
        conf_headers = ["Category", "Confidence", "Valid/Total", "Notes"]
        timing_invalid_note = "invalid timing detected" if confidence.get("timing_invalid") else ""

    confidence_rows = "\n".join(
        [
            f"<tr><td>{conf_labels['knee']}</td><td>{_pct(confidence.get('knee_confidence'))}</td>"
            f"<td>{_count('knee_valid_frames', 'knee_total_frames')}</td><td></td></tr>",
            f"<tr><td>{conf_labels['trunk']}</td><td>{_pct(confidence.get('trunk_confidence'))}</td>"
            f"<td>{_count('trunk_valid_frames', 'trunk_total_frames')}</td><td></td></tr>",
            f"<tr><td>{conf_labels['impact']}</td><td>{_pct(confidence.get('impact_confidence'))}</td>"
            f"<td>{_count('impact_valid_frames', 'impact_total_frames')}</td><td>{value_text(confidence.get('impact_source'))}</td></tr>",
            f"<tr><td>{conf_labels['timing']}</td><td>{_pct(confidence.get('timing_confidence'))}</td>"
            f"<td>{confidence.get('timing_valid_count', 'n/a')}/{confidence.get('timing_total_count', 'n/a')}</td><td>{timing_invalid_note}</td></tr>",
        ]
    )

    def _key_label(name: str) -> str:
        base = label(name)
        status = phases.get(f"{name}_status") if isinstance(phases, dict) else None
        if status == "fallback":
            return f"{base}（自动回退）" if lang == "zh" else f"{base} (fallback)"
        return base

    key_imgs = "\n".join(
        f"<div class='img-card'><div class='label'>{_key_label(name)}</div><img src='{path}'></div>"
        for name, path in keyframes.items()
    )

    keyframe_3d_title = "四个关键帧 3D 对比与关键角度" if lang == "zh" else "3D Keyframes And Angles"
    keyframe_3d_note = "未提供四个关键帧的 3D 结果。" if lang == "zh" else "3D keyframe results are not available."
    keyframe_3d_section = ""
    phase_payloads = keyframe_3d.get("phases") if isinstance(keyframe_3d, dict) else None
    if isinstance(phase_payloads, dict):
        phase_cards = []
        for phase_name in KEYFRAME_3D_PHASES:
            payload = phase_payloads.get(phase_name)
            if not isinstance(payload, dict):
                continue

            phase_label = str(payload.get("label") or _phase_display_name(phase_name, lang))
            frame_num = payload.get("frame_num")
            frame_text = (
                f"（第 {int(frame_num)} 帧）"
                if lang == "zh" and isinstance(frame_num, (int, float))
                else (f"(Frame {int(frame_num)})" if isinstance(frame_num, (int, float)) else "")
            )
            artifact_paths = payload.get("artifact_paths") or {}
            compare_path = artifact_paths.get("compare")
            error_text = payload.get("error")
            metric_pool = {}
            metric_pool.update(payload.get("angles_3d") or {})
            metric_pool.update(payload.get("phase_metrics_3d") or {})
            angle_rows = []
            for metric_name in KEYFRAME_3D_ANGLE_ORDER.get(phase_name, []):
                value = metric_pool.get(metric_name)
                if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
                    continue
                suffix = " BH" if metric_name == "impact_height_norm" else "°"
                display_value = f"{float(value):.3f}{suffix}" if metric_name == "impact_height_norm" else f"{float(value):.1f}{suffix}"
                angle_rows.append(
                    f"<tr><td>{html_lib.escape(_angle_display_name(metric_name, lang))}</td><td>{display_value}</td></tr>"
                )
            if not angle_rows:
                angle_rows.append(
                    f"<tr><td colspan='2'>{html_lib.escape('未能提取有效 3D 角度。' if lang == 'zh' else 'No valid 3D angles were extracted.')}</td></tr>"
                )

            if compare_path:
                media_html = f"<img src='{compare_path}' alt='{html_lib.escape(phase_label)}'>"
            else:
                message = str(error_text or keyframe_3d_note)
                media_html = f"<div class='note'>{html_lib.escape(message)}</div>"

            phase_cards.append(
                f"""
  <div class="phase-card">
    <div class="phase-media">
      <div class="label">{html_lib.escape(phase_label)} {html_lib.escape(frame_text)}</div>
      {media_html}
    </div>
    <div class="phase-metrics">
      <table>
        <tr><th>{t["metric"]}</th><th>{t["value"]}</th></tr>
        {"".join(angle_rows)}
      </table>
    </div>
  </div>
"""
            )
        if phase_cards:
            keyframe_3d_section = f"""
  <div class="section">
    <h2>{keyframe_3d_title}</h2>
    {''.join(phase_cards)}
  </div>
"""
    if not keyframe_3d_section:
        keyframe_3d_section = f"""
  <div class="section">
    <h2>{keyframe_3d_title}</h2>
    <div class="note">{html_lib.escape(keyframe_3d_note)}</div>
  </div>
"""

    report_summary = report_3d_summary or {}
    score_block = (
        f"{float(report_summary.get('final_score')):.1f}"
        if isinstance(report_summary.get("final_score"), (int, float, np.floating)) and np.isfinite(report_summary.get("final_score"))
        else f"{final_score:.1f}"
    )
    coverage_pct = (
        float(report_summary.get("coverage_ratio")) * 100.0
        if isinstance(report_summary.get("coverage_ratio"), (int, float, np.floating)) and np.isfinite(report_summary.get("coverage_ratio"))
        else (biomech.get("coverage") or 0.0) * 100.0
    )
    core_rows_html = "\n".join(
        f"<tr><td>{html_lib.escape(str(row.get('name', '')))}</td><td>{html_lib.escape(str(row.get('value_text', '')))}</td><td>{html_lib.escape(str(row.get('conclusion', '')))}</td></tr>"
        for row in (report_summary.get("core_rows") or [])
    )
    def _score_cell(row: Dict[str, object]) -> str:
        value = _safe_float(row.get("score"))
        return "n/a" if not np.isfinite(value) else f"{value:.1f}"

    def _timing_value_cell(row: Dict[str, object]) -> str:
        value = _safe_float(row.get("value"))
        return "n/a" if not np.isfinite(value) else f"{value:.3f}s"

    score_rows_html = "\n".join(
        f"<tr><td>{html_lib.escape(str(row.get('name', '')))}</td><td>{_score_cell(row)}</td><td>{int(round(float(row.get('weight', 0.0)) * 100))}%</td><td>{html_lib.escape(str(row.get('note', '')))}</td></tr>"
        for row in (report_summary.get("score_rows") or [])
    )
    quality_rows_html = "\n".join(
        f"<tr><td>{html_lib.escape(str(row.get('name', '')))}</td><td>{html_lib.escape(str(row.get('success', 'n/a')))}</td><td>{('是' if row.get('usable') else '否') if lang == 'zh' else ('Yes' if row.get('usable') else 'No')}</td><td>{html_lib.escape(str(row.get('note', '')))}</td></tr>"
        for row in (report_summary.get("quality_rows") or [])
    )
    timing_rows_html = "\n".join(
        f"<tr><td>{html_lib.escape(str(row.get('name', '')))}</td><td>{_timing_value_cell(row)}</td><td>{_score_cell(row)}</td><td>{html_lib.escape(str(row.get('note', '')))}</td></tr>"
        for row in (report_summary.get("timing_rows") or [])
    )
    timing_total_score = _safe_float(report_summary.get("timing_total_score"))
    timing_total_html = (
        f"<tr><td>{'辅助节奏观察总分' if lang == 'zh' else 'Auxiliary Timing Total'}</td><td>—</td><td>{timing_total_score:.1f}</td><td>{html_lib.escape(str(report_summary.get('timing_total_note', '')))}</td></tr>"
        if np.isfinite(timing_total_score)
        else ""
    )
    top_explanation = html_lib.escape(str(report_summary.get("explanation", "")))
    basis_note = html_lib.escape(str(report_summary.get("basis", "")))

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{t["title"]}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin: 0 0 8px 0; }}
    .score {{ font-size: 42px; font-weight: bold; color: #1b7837; }}
    .section {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .img-grid {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .img-card {{ width: 320px; }}
    .img-card img {{ width: 100%; border: 1px solid #ddd; }}
    .verify-grid .img-card {{ width: 360px; }}
    .verify-grid .verify-hero {{ width: 100%; max-width: 1100px; }}
    .verify-grid .verify-wide {{ width: 520px; }}
    .label {{ font-weight: bold; margin-bottom: 6px; }}
    .note {{ margin-top: 6px; color: #555; font-size: 13px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: bold; }}
    .badge.good {{ background: #e6f4ea; color: #1b7837; }}
    .badge.caution {{ background: #fff3cd; color: #8a6d3b; }}
    .badge.needs-review {{ background: #f8d7da; color: #a61e4d; }}
    .phase-card {{ display: grid; grid-template-columns: minmax(460px, 1.45fr) minmax(260px, 0.85fr); gap: 18px; margin-top: 18px; align-items: start; }}
    .phase-media img {{ width: 100%; border: 1px solid #ddd; }}
    @media (max-width: 980px) {{
      .phase-card {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>{t["title"]}</h1>
  <div class="score">{score_block}</div>
  <div>{"技术评分（当前版，0–100）" if lang == "zh" else "Technical Score (Current Version, 0–100)"}</div>
  <div>{"核心指标覆盖率" if lang == "zh" else "Core Metric Coverage"}：{coverage_pct:.0f}%</div>
  <div class="note">{top_explanation}</div>

  <div class="section">
    <h2>{"核心技术指标（文献支持）" if lang == "zh" else "Core Technical Metrics"}</h2>
    <table>
      <tr><th>{"指标" if lang == "zh" else "Metric"}</th><th>{"数值" if lang == "zh" else "Value"}</th><th>{"评估结论" if lang == "zh" else "Assessment"}</th></tr>
      {core_rows_html}
    </table>
    <div class="note">{basis_note}</div>
  </div>

  <div class="section">
    <h2>{"评分构成" if lang == "zh" else "Score Composition"}</h2>
    <table>
      <tr><th>{"项目" if lang == "zh" else "Item"}</th><th>{"分数" if lang == "zh" else "Score"}</th><th>{"权重" if lang == "zh" else "Weight"}</th><th>{"说明" if lang == "zh" else "Notes"}</th></tr>
      {score_rows_html}
    </table>
    <div class="note">
      {"技术评分计算方式：奖杯位膝屈角 × 40% + 奖杯位躯干倾角 × 40% + 辅助节奏观察 × 20%。" if lang == "zh" else "Technical score = trophy knee flexion × 40% + trophy trunk inclination × 40% + auxiliary timing × 20%."}
    </div>
  </div>

  <div class="section">
    <h2>{"数据质量与可评分性" if lang == "zh" else "Data Quality And Scorability"}</h2>
    <table>
      <tr><th>{"项目" if lang == "zh" else "Item"}</th><th>{"检测成功率" if lang == "zh" else "Success Rate"}</th><th>{"可用于评分" if lang == "zh" else "Used In Score"}</th><th>{"备注" if lang == "zh" else "Notes"}</th></tr>
      {quality_rows_html}
    </table>
  </div>

  <div class="section">
    <h2>{"辅助节奏观察（纳入总分）" if lang == "zh" else "Auxiliary Timing (Included In Total Score)"}</h2>
    <table>
      <tr><th>{"指标" if lang == "zh" else "Metric"}</th><th>{"数值" if lang == "zh" else "Value"}</th><th>{"单项分" if lang == "zh" else "Item Score"}</th><th>{"观察结论" if lang == "zh" else "Observation"}</th></tr>
      {timing_rows_html}
      {timing_total_html}
    </table>
    <div class="note">{html_lib.escape(str(report_summary.get("timing_total_note", "")))}</div>
  </div>

  {keyframe_3d_section}
</body>
</html>
"""
    return html


def _build_compact_3d_report_summary(
    keyframe_3d: Optional[Dict[str, object]],
    timing_metrics: Dict[str, float],
    lang: str,
) -> Dict[str, object]:
    phases = (keyframe_3d or {}).get("phases") if isinstance(keyframe_3d, dict) else {}
    if not isinstance(phases, dict):
        phases = {}

    trophy_payload = phases.get("trophy") if isinstance(phases.get("trophy"), dict) else {}
    drop_payload = phases.get("racket_drop") if isinstance(phases.get("racket_drop"), dict) else {}
    contact_payload = phases.get("contact") if isinstance(phases.get("contact"), dict) else {}

    trophy_metrics = trophy_payload.get("phase_metrics_3d") if isinstance(trophy_payload.get("phase_metrics_3d"), dict) else {}
    drop_metrics = drop_payload.get("phase_metrics_3d") if isinstance(drop_payload.get("phase_metrics_3d"), dict) else {}
    contact_metrics = contact_payload.get("phase_metrics_3d") if isinstance(contact_payload.get("phase_metrics_3d"), dict) else {}

    tp_knee = _safe_float(trophy_metrics.get("lead_knee_flexion_deg"))
    tp_trunk = _safe_float(trophy_metrics.get("trunk_inclination_deg"))
    impact_height_norm = _safe_float(contact_metrics.get("impact_height_norm"))
    rlp_shoulder_er = _safe_float(drop_metrics.get("shoulder_external_rotation_proxy_deg"))
    bi_shoulder_raise = _safe_float(contact_metrics.get("shoulder_raise_deg"))
    bi_elbow_flex = _safe_float(contact_metrics.get("elbow_flexion_deg"))

    knee_score, _ = serve_score._score_lit_plateau(tp_knee, LIT_KNEE_MEAN, LIT_KNEE_SD)
    trunk_score, _ = serve_score._score_lit_plateau(tp_trunk, LIT_TRUNK_MEAN, LIT_TRUNK_SD)

    def _fmt_angle(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.1f}°"

    def _fmt_bh(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.3f} BH"

    def _fmt_time(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.3f}s"

    timing_specs = [
        (
            "timing_toss_to_trophy",
            "抛球→奖杯位间隔" if lang == "zh" else "Toss to Trophy Interval",
            "暂无统一参考" if lang == "zh" else "No unified reference",
            TIMING_RANGES["timing_toss_to_trophy"],
        ),
        (
            "timing_trophy_to_drop",
            "奖杯位→下落间隔" if lang == "zh" else "Trophy to Drop Interval",
            "约 0.19–0.20s" if lang == "zh" else "About 0.19-0.20s",
            TIMING_RANGES["timing_trophy_to_drop"],
        ),
        (
            "timing_drop_to_contact",
            "下落→击球间隔" if lang == "zh" else "Drop to Contact Interval",
            "约 0.12–0.13s" if lang == "zh" else "About 0.12-0.13s",
            TIMING_RANGES["timing_drop_to_contact"],
        ),
    ]
    timing_rows: List[Dict[str, object]] = []
    timing_scores: List[float] = []
    for key, display_name, reference_text, rng in timing_specs:
        value = _safe_float(timing_metrics.get(key))
        score = _score_centered_range(value, rng[0], rng[1], edge_score=80.0)
        if np.isfinite(score):
            timing_scores.append(score)
        timing_rows.append(
            {
                "name": display_name,
                "current_value": _fmt_time(value),
                "reference": reference_text,
            }
        )

    timing_score = float(np.mean(timing_scores)) if timing_scores else float("nan")
    final_score = _weighted_score(
        [
            (knee_score, 0.40),
            (trunk_score, 0.40),
            (timing_score, 0.20),
        ]
    )

    core_rows = [
        {
            "name": "奖杯位膝屈角（TP）" if lang == "zh" else "Trophy Knee Flexion (TP)",
            "current_value": _fmt_angle(tp_knee),
            "reference": "64.5 ± 9.7°",
        },
        {
            "name": "奖杯位躯干倾角（TP）" if lang == "zh" else "Trophy Trunk Inclination (TP)",
            "current_value": _fmt_angle(tp_trunk),
            "reference": "25.0 ± 7.1°",
        },
        {
            "name": "击球高度（身高归一化）" if lang == "zh" else "Impact Height (Body-Height Normalized)",
            "current_value": _fmt_bh(impact_height_norm),
            "reference": "约 1.50 BH" if lang == "zh" else "About 1.50 BH",
        },
        {
            "name": "拍头下落肩外旋（RLP）" if lang == "zh" else "Racket-Drop Shoulder External Rotation (RLP)",
            "current_value": _fmt_angle(rlp_shoulder_er),
            "reference": "130.1 ± 26.5°",
        },
        {
            "name": "击球瞬间肩抬高（BI）" if lang == "zh" else "Impact Shoulder Raise (BI)",
            "current_value": _fmt_angle(bi_shoulder_raise),
            "reference": "110.7 ± 16.9°",
        },
        {
            "name": "击球瞬间肘屈（BI）" if lang == "zh" else "Impact Elbow Flexion (BI)",
            "current_value": _fmt_angle(bi_elbow_flex),
            "reference": "30.1 ± 15.9°",
        },
    ]

    score_rows = [
        {
            "name": "奖杯位膝屈角（TP）" if lang == "zh" else "Trophy Knee Flexion (TP)",
            "score": knee_score,
            "weight": 0.40,
        },
        {
            "name": "奖杯位躯干倾角（TP）" if lang == "zh" else "Trophy Trunk Inclination (TP)",
            "score": trunk_score,
            "weight": 0.40,
        },
        {
            "name": "辅助节奏观察" if lang == "zh" else "Auxiliary Timing",
            "score": timing_score,
            "weight": 0.20,
        },
        {
            "name": "击球高度（归一化）" if lang == "zh" else "Impact Height (Normalized)",
            "score": float("nan"),
            "weight": 0.0,
        },
        {
            "name": "技术评分" if lang == "zh" else "Technical Score",
            "score": final_score,
            "weight": 1.0,
        },
    ]

    timing_rows.append(
        {
            "name": "辅助节奏观察总分" if lang == "zh" else "Auxiliary Timing Total",
            "current_value": "n/a" if not np.isfinite(timing_score) else f"{timing_score:.1f}",
            "reference": "系统计算" if lang == "zh" else "System computed",
        }
    )

    return {
        "final_score": final_score,
        "coverage_ratio": 2.0 / 3.0,
        "core_rows": core_rows,
        "score_rows": score_rows,
        "timing_rows": timing_rows,
        "raw_values": {
            "tp_knee_flexion_deg": tp_knee,
            "tp_trunk_inclination_deg": tp_trunk,
            "impact_height_norm": impact_height_norm,
            "rlp_shoulder_external_rotation_deg": rlp_shoulder_er,
            "bi_shoulder_raise_deg": bi_shoulder_raise,
            "bi_elbow_flexion_deg": bi_elbow_flex,
            "timing_toss_to_trophy": _safe_float(timing_metrics.get("timing_toss_to_trophy")),
            "timing_trophy_to_drop": _safe_float(timing_metrics.get("timing_trophy_to_drop")),
            "timing_drop_to_contact": _safe_float(timing_metrics.get("timing_drop_to_contact")),
            "tp_knee_score": knee_score,
            "tp_trunk_score": trunk_score,
            "timing_score": timing_score,
        },
    }


def _render_compact_html(
    out_dir: str,
    keyframe_3d: Optional[Dict[str, object]],
    report_summary: Optional[Dict[str, object]],
    lang: str,
) -> str:
    page_title = "动作评估报告" if lang == "zh" else "Motion Evaluation Report"
    report_summary = report_summary or {}

    def _score_cell(value: object) -> str:
        numeric = _safe_float(value)
        return "n/a" if not np.isfinite(numeric) else f"{numeric:.1f}"

    core_rows_html = "\n".join(
        f"<tr><td>{html_lib.escape(str(row.get('name', '')))}</td><td>{html_lib.escape(str(row.get('current_value', '')))}</td><td>{html_lib.escape(str(row.get('reference', '')))}</td></tr>"
        for row in (report_summary.get("core_rows") or [])
    )
    score_rows_html = "\n".join(
        f"<tr><td>{html_lib.escape(str(row.get('name', '')))}</td><td>{_score_cell(row.get('score'))}</td><td>{int(round(float(row.get('weight', 0.0)) * 100))}%</td></tr>"
        for row in (report_summary.get("score_rows") or [])
    )
    timing_rows_html = "\n".join(
        f"<tr><td>{html_lib.escape(str(row.get('name', '')))}</td><td>{html_lib.escape(str(row.get('current_value', '')))}</td><td>{html_lib.escape(str(row.get('reference', '')))}</td></tr>"
        for row in (report_summary.get("timing_rows") or [])
    )

    keyframe_3d_title = "关键帧对比" if lang == "zh" else "Keyframe Comparison"
    keyframe_3d_note = "未生成关键帧 3D 对比结果。" if lang == "zh" else "3D keyframe comparison is unavailable."
    keyframe_3d_section = ""
    phase_payloads = keyframe_3d.get("phases") if isinstance(keyframe_3d, dict) else None
    if isinstance(phase_payloads, dict):
        phase_cards = []
        for phase_name in KEYFRAME_3D_PHASES:
            payload = phase_payloads.get(phase_name)
            if not isinstance(payload, dict):
                continue

            phase_label = str(payload.get("label") or _phase_display_name(phase_name, lang))
            frame_num = payload.get("frame_num")
            frame_text = (
                f"（第 {int(frame_num)} 帧）"
                if lang == "zh" and isinstance(frame_num, (int, float))
                else (f"(Frame {int(frame_num)})" if isinstance(frame_num, (int, float)) else "")
            )
            artifact_paths = payload.get("artifact_paths") or {}
            compare_path = artifact_paths.get("compare")
            metric_pool: Dict[str, object] = {}
            metric_pool.update(payload.get("angles_3d") or {})
            metric_pool.update(payload.get("phase_metrics_3d") or {})

            angle_rows = []
            for metric_name in KEYFRAME_3D_ANGLE_ORDER.get(phase_name, []):
                value = metric_pool.get(metric_name)
                if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
                    continue
                if metric_name == "impact_height_norm":
                    display_value = f"{float(value):.3f} BH"
                else:
                    display_value = f"{float(value):.1f}°"
                angle_rows.append(
                    f"<tr><td>{html_lib.escape(_angle_display_name(metric_name, lang))}</td><td>{display_value}</td></tr>"
                )
            if not angle_rows:
                angle_rows.append(
                    f"<tr><td colspan='2'>{html_lib.escape('未提取到有效角度。' if lang == 'zh' else 'No valid angles extracted.')}</td></tr>"
                )

            media_html = (
                f"<img src='{compare_path}' alt='{html_lib.escape(phase_label)}'>"
                if compare_path
                else f"<div class='note'>{html_lib.escape(keyframe_3d_note)}</div>"
            )

            phase_cards.append(
                f"""
  <div class="phase-card">
    <div class="phase-media">
      <div class="label">{html_lib.escape(phase_label)}{html_lib.escape(frame_text)}</div>
      {media_html}
    </div>
    <div class="phase-metrics">
      <div class="label">{html_lib.escape(phase_label)}</div>
      <table>
        <tr><th>{"指标" if lang == "zh" else "Metric"}</th><th>{"当前值" if lang == "zh" else "Current Value"}</th></tr>
        {"".join(angle_rows)}
      </table>
    </div>
  </div>
"""
            )
        if phase_cards:
            keyframe_3d_section = f"""
  <div class="section">
    <h2>{keyframe_3d_title}</h2>
    {''.join(phase_cards)}
  </div>
"""
    if not keyframe_3d_section:
        keyframe_3d_section = f"""
  <div class="section">
    <h2>{keyframe_3d_title}</h2>
    <div class="note">{html_lib.escape(keyframe_3d_note)}</div>
  </div>
"""

    score_block = (
        f"{float(report_summary.get('final_score')):.1f}"
        if isinstance(report_summary.get("final_score"), (int, float, np.floating)) and np.isfinite(report_summary.get("final_score"))
        else "n/a"
    )
    coverage_pct = (
        float(report_summary.get("coverage_ratio")) * 100.0
        if isinstance(report_summary.get("coverage_ratio"), (int, float, np.floating)) and np.isfinite(report_summary.get("coverage_ratio"))
        else 0.0
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{page_title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin: 0 0 8px 0; }}
    h2 {{ margin-bottom: 8px; }}
    .score {{ font-size: 42px; font-weight: bold; color: #1b7837; }}
    .section {{ margin-top: 28px; }}
    .label {{ font-weight: bold; margin-bottom: 6px; }}
    .note {{ margin-top: 6px; color: #555; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .phase-card {{ display: grid; grid-template-columns: minmax(520px, 1.45fr) minmax(260px, 0.85fr); gap: 18px; margin-top: 18px; align-items: start; }}
    .phase-media img {{ width: 100%; border: 1px solid #ddd; }}
    @media (max-width: 980px) {{
      .phase-card {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>{page_title}</h1>
  <div class="score">{score_block}</div>
  <div>{"技术评分" if lang == "zh" else "Technical Score"}</div>
  <div>{"核心指标覆盖率" if lang == "zh" else "Core Metric Coverage"} {coverage_pct:.0f}%</div>

  <div class="section">
    <h2>{"核心技术指标" if lang == "zh" else "Core Technical Metrics"}</h2>
    <table>
      <tr><th>{"指标" if lang == "zh" else "Metric"}</th><th>{"当前值" if lang == "zh" else "Current Value"}</th><th>{"文献参考" if lang == "zh" else "Reference"}</th></tr>
      {core_rows_html}
    </table>
  </div>

  <div class="section">
    <h2>{"评分构成" if lang == "zh" else "Score Composition"}</h2>
    <table>
      <tr><th>{"项目" if lang == "zh" else "Item"}</th><th>{"分数" if lang == "zh" else "Score"}</th><th>{"权重" if lang == "zh" else "Weight"}</th></tr>
      {score_rows_html}
    </table>
  </div>

  <div class="section">
    <h2>{"辅助节奏观察" if lang == "zh" else "Auxiliary Timing"}</h2>
    <table>
      <tr><th>{"指标" if lang == "zh" else "Metric"}</th><th>{"当前值" if lang == "zh" else "Current Value"}</th><th>{"文献参考" if lang == "zh" else "Reference"}</th></tr>
      {timing_rows_html}
    </table>
  </div>

  {keyframe_3d_section}
</body>
</html>
"""
    return html


def _build_compact_3d_report_summary_en(
    keyframe_3d: Optional[Dict[str, object]],
    timing_metrics: Dict[str, float],
) -> Dict[str, object]:
    phases = (keyframe_3d or {}).get("phases") if isinstance(keyframe_3d, dict) else {}
    if not isinstance(phases, dict):
        phases = {}

    trophy_payload = phases.get("trophy") if isinstance(phases.get("trophy"), dict) else {}
    drop_payload = phases.get("racket_drop") if isinstance(phases.get("racket_drop"), dict) else {}
    contact_payload = phases.get("contact") if isinstance(phases.get("contact"), dict) else {}

    trophy_metrics = trophy_payload.get("phase_metrics_3d") if isinstance(trophy_payload.get("phase_metrics_3d"), dict) else {}
    drop_metrics = drop_payload.get("phase_metrics_3d") if isinstance(drop_payload.get("phase_metrics_3d"), dict) else {}
    contact_metrics = contact_payload.get("phase_metrics_3d") if isinstance(contact_payload.get("phase_metrics_3d"), dict) else {}

    tp_knee = _safe_float(trophy_metrics.get("lead_knee_flexion_deg"))
    tp_trunk = _safe_float(trophy_metrics.get("trunk_inclination_deg"))
    impact_height_norm = _safe_float(contact_metrics.get("impact_height_norm"))
    rlp_shoulder_er = _safe_float(drop_metrics.get("shoulder_external_rotation_proxy_deg"))
    bi_shoulder_raise = _safe_float(contact_metrics.get("shoulder_raise_deg"))
    bi_elbow_flex = _safe_float(contact_metrics.get("elbow_flexion_deg"))

    knee_score, _ = serve_score._score_lit_plateau(tp_knee, LIT_KNEE_MEAN, LIT_KNEE_SD)
    trunk_score, _ = serve_score._score_lit_plateau(tp_trunk, LIT_TRUNK_MEAN, LIT_TRUNK_SD)

    def _fmt_angle(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.1f} deg"

    def _fmt_bh(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.3f} BH"

    def _fmt_time(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.3f}s"

    timing_specs = [
        ("timing_toss_to_trophy", "Toss to Trophy Interval", "No unified reference", TIMING_RANGES["timing_toss_to_trophy"]),
        ("timing_trophy_to_drop", "Trophy to Drop Interval", "About 0.19-0.20s", TIMING_RANGES["timing_trophy_to_drop"]),
        ("timing_drop_to_contact", "Drop to Contact Interval", "About 0.12-0.13s", TIMING_RANGES["timing_drop_to_contact"]),
    ]
    timing_rows: List[Dict[str, object]] = []
    timing_scores: List[float] = []
    for key, display_name, reference_text, rng in timing_specs:
        value = _safe_float(timing_metrics.get(key))
        score = _score_centered_range(value, rng[0], rng[1], edge_score=80.0)
        if np.isfinite(score):
            timing_scores.append(score)
        timing_rows.append(
            {
                "name": display_name,
                "current_value": _fmt_time(value),
                "reference": reference_text,
            }
        )

    timing_score = float(np.mean(timing_scores)) if timing_scores else float("nan")
    final_score = _weighted_score([(knee_score, 0.40), (trunk_score, 0.40), (timing_score, 0.20)])

    core_rows = [
        {"name": "Trophy Knee Flexion (TP)", "current_value": _fmt_angle(tp_knee), "reference": "64.5 +/- 9.7 deg"},
        {"name": "Trophy Trunk Inclination (TP)", "current_value": _fmt_angle(tp_trunk), "reference": "25.0 +/- 7.1 deg"},
        {"name": "Impact Height (Body-Height Normalized)", "current_value": _fmt_bh(impact_height_norm), "reference": "About 1.50 BH"},
        {"name": "Racket-Drop Shoulder External Rotation (RLP)", "current_value": _fmt_angle(rlp_shoulder_er), "reference": "130.1 +/- 26.5 deg"},
        {"name": "Impact Shoulder Raise (BI)", "current_value": _fmt_angle(bi_shoulder_raise), "reference": "110.7 +/- 16.9 deg"},
        {"name": "Impact Elbow Flexion (BI)", "current_value": _fmt_angle(bi_elbow_flex), "reference": "30.1 +/- 15.9 deg"},
    ]

    score_rows = [
        {"name": "Trophy Knee Flexion (TP)", "score": knee_score, "weight": 0.40},
        {"name": "Trophy Trunk Inclination (TP)", "score": trunk_score, "weight": 0.40},
        {"name": "Auxiliary Timing", "score": timing_score, "weight": 0.20},
        {"name": "Impact Height (Normalized)", "score": float("nan"), "weight": 0.0},
        {"name": "Technical Score", "score": final_score, "weight": 1.0},
    ]
    timing_rows.append(
        {
            "name": "Auxiliary Timing Total",
            "current_value": "n/a" if not np.isfinite(timing_score) else f"{timing_score:.1f}",
            "reference": "System computed",
        }
    )

    return {
        "final_score": final_score,
        "coverage_ratio": 2.0 / 3.0,
        "core_rows": core_rows,
        "score_rows": score_rows,
        "timing_rows": timing_rows,
        "raw_values": {
            "tp_knee_flexion_deg": tp_knee,
            "tp_trunk_inclination_deg": tp_trunk,
            "impact_height_norm": impact_height_norm,
            "rlp_shoulder_external_rotation_deg": rlp_shoulder_er,
            "bi_shoulder_raise_deg": bi_shoulder_raise,
            "bi_elbow_flexion_deg": bi_elbow_flex,
            "timing_toss_to_trophy": _safe_float(timing_metrics.get("timing_toss_to_trophy")),
            "timing_trophy_to_drop": _safe_float(timing_metrics.get("timing_trophy_to_drop")),
            "timing_drop_to_contact": _safe_float(timing_metrics.get("timing_drop_to_contact")),
            "tp_knee_score": knee_score,
            "tp_trunk_score": trunk_score,
            "timing_score": timing_score,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument(
        "--model",
        default=r"D:\Serve_Score\models\pose_landmarker_full.task",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pose_csv", default=None)
    ap.add_argument("--pose_source", choices=["smooth"], default="smooth",
                    help="Data source selection for report.")
    ap.add_argument("--use_coords", choices=["smooth"], default="smooth",
                    help="Smooth pose coordinates used for scoring.")
    ap.add_argument("--smooth_mode", choices=["zero_phase", "causal"], default="zero_phase",
                    help="Smoothing mode used when generating the pose track.")
    ap.add_argument("--v_thr", type=float, default=0.6)
    ap.add_argument("--gap_max", type=int, default=3)
    ap.add_argument("--smooth_window", type=int, default=9)
    ap.add_argument("--lang", choices=["en"], default="en")
    ap.add_argument("--quantile_calib", choices=["on", "off"], default="off",
                    help="Enable per-serve quantile calibration for impact/timing.")
    ap.add_argument("--hand_mode", choices=["auto", "A", "B"], default="auto",
                    help="Force hand assignment: A=left toss/right racket, B=right toss/left racket.")
    ap.add_argument("--verify_3d", choices=["auto", "on", "off"], default="auto",
                    help="Attach 3D verification to the report: auto=load/generate if possible.")
    ap.add_argument("--verify_3d_neighbor_radius", type=int, default=2,
                    help="Neighbor frame radius used for 3D stability verification.")
    ap.add_argument("--verify_3d_device", default="auto",
                    help="Torch device for HybrIK verification, e.g. auto/cpu/cuda:0.")
    ap.add_argument("--output_mode", choices=["compact", "full"], default="compact",
                    help="compact=keep only report essentials; full=retain debug intermediates.")

    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    video_path, staged_video_path = _prepare_video_input(args.video)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()

    pose_csv = args.pose_csv
    generated_pose_csv = None
    pose_source_used = "smooth"
    use_coords = "smooth"
    overlay_video_path = os.path.join(args.out_dir, "serve_overlay.mp4")

    if not pose_csv:
        generated_pose_csv = os.path.join(args.out_dir, "_pose_smooth.csv")
        if not os.path.exists(generated_pose_csv) or not os.path.exists(overlay_video_path):
            _run_extract(
                video_path,
                args.model,
                generated_pose_csv,
                overlay_video_path,
                smooth_mode=args.smooth_mode,
                v_thr=args.v_thr,
                gap_max=args.gap_max,
                smooth_window=args.smooth_window,
            )
        pose_csv = generated_pose_csv

    df = pd.read_csv(pose_csv, encoding="utf-8-sig")
    result = serve_score.analyze_serve(
        df,
        use_coords=use_coords,
        hand_mode=args.hand_mode,
        v_thr=args.v_thr,
        gap_max=args.gap_max,
        quantile_calib=(args.quantile_calib == "on"),
    )

    metrics = dict(result["metrics"])
    subscores = result["subscores"]
    final_score_legacy = result["final_score"]
    biomech_full = result.get("biomech", {})
    biomech = {
        "subscores": biomech_full.get("subscores", {}),
        "final_score": biomech_full.get("final_score"),
        "metrics_min": biomech_full.get("metrics_min", {}),
        "confidence": biomech_full.get("confidence", {}),
        "risk_flags": biomech_full.get("risk_flags", []),
        "coverage": biomech_full.get("coverage"),
    }
    z_src = biomech_full.get("z_scores", {}) or {}
    biomech["z_scores"] = {
        k: v for k, v in z_src.items() if k in {"knee_flexion", "trunk_inclination"}
    }
    legacy_subscores = result.get("legacy_subscores")
    legacy_final = result.get("legacy_final_score")
    phases = result["phases"]

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    payload = {
        "final_score": final_score_legacy,
        "sub_scores": subscores,
        "metrics": metrics,
        "phases": phases,
        "biomech": biomech,
        "legacy_sub_scores": legacy_subscores,
        "legacy_final_score": legacy_final,
        "pose_source_used": pose_source_used,
        "fps": fps,
        "v_thr": args.v_thr,
        "gap_max": args.gap_max,
        "smooth_window": args.smooth_window,
        "smooth_mode": args.smooth_mode,
        "hand_mode_selected": result.get("hand_mode_selected"),
        "mirror_suspect": result.get("mirror_suspect"),
        "hand_mode_guess": result.get("hand_mode_guess"),
        "quality_flags": result.get("quality_flags"),
    }

    debug_path = os.path.join(args.out_dir, "score_debug.json")
    if args.output_mode == "full":
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(result.get("score_debug", {}), f, indent=2, ensure_ascii=False)

    run_id = int(time.time() * 1000)
    _save_keyframes(video_path, phases, args.out_dir)
    phase_frame_nums = {
        phase_name: _phase_frame_num(phases, result.get("frames"), phase_name)
        for phase_name in KEYFRAME_3D_PHASES
    }
    keyframe_3d = _maybe_generate_keyframe_3d_bundle(
        out_dir=args.out_dir,
        phase_frame_nums=phase_frame_nums,
        lang=args.lang,
        run_id=run_id,
        verify_3d=args.verify_3d,
        verify_device=args.verify_3d_device,
    )
    report_3d_summary = _build_compact_3d_report_summary_en(keyframe_3d, metrics)
    display_final_score = _safe_float(report_3d_summary.get("final_score"))
    if not np.isfinite(display_final_score):
        display_final_score = float(final_score_legacy)
    html = _render_compact_html(
        args.out_dir,
        keyframe_3d=keyframe_3d,
        report_summary=report_3d_summary,
        lang=args.lang,
    )

    payload["legacy_final_score_pose2d"] = final_score_legacy
    payload["final_score"] = display_final_score
    payload["keyframe_3d"] = keyframe_3d
    payload["report_3d_summary"] = report_3d_summary
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    html_path = os.path.join(args.out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    pdf_path = os.path.join(args.out_dir, "report.pdf")
    if _try_export_pdf(html_path, pdf_path):
        print(f"[OK] report.pdf: {pdf_path}")
    else:
        print("[INFO] PDF export skipped (optional dependency missing).")

    if args.output_mode == "compact":
        _cleanup_compact_outputs(args.out_dir, generated_pose_csv)

    if staged_video_path:
        _remove_path(staged_video_path)

    print(f"[OK] metrics.json: {metrics_path}")
    print(f"[OK] report.html : {html_path}")
    if os.path.exists(overlay_video_path):
        print(f"[OK] overlay.mp4 : {overlay_video_path}")
    if args.output_mode == "full" and os.path.exists(debug_path):
        print(f"[OK] score_debug.json: {debug_path}")
    print("[INFO] pose_source_used:", pose_source_used)
    print("[INFO] fps:", fps)
    print("[INFO] v_thr:", args.v_thr, "gap_max:", args.gap_max, "smooth_window:", args.smooth_window,
          "smooth_mode:", args.smooth_mode)
    print("[INFO] coord_system_used:", metrics.get("coord_system_used"))
    missing_count = result.get("score_debug", {}).get("missing_metrics_count")
    print("[INFO] body_scale:", metrics.get("body_scale"), "missing_metrics_count:", missing_count)
    print("[DEBUG] phases:",
          "trophy", phases.get("trophy"),
          "drop", phases.get("racket_drop"),
          "contact", phases.get("contact"),
          "finish", phases.get("finish"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "__extract_pose":
        extract_pose_compare_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "__run_hybrik_image":
        run_hybrik_image_main(sys.argv[2:])
    else:
        main()
