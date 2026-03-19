import os
import math
import argparse

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# One Euro Filter
# -----------------------------
def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class LowPass:
    def __init__(self):
        self.inited = False
        self.y = 0.0

    def apply(self, x: float, a: float) -> float:
        if not self.inited:
            self.inited = True
            self.y = x
            return x
        self.y = a * x + (1.0 - a) * self.y
        return self.y


class OneEuro:
    """
    min_cutoff: 越大 -> 越“跟手”(更少滞后)，但更容易抖
    beta: 越大 -> 快速运动时更少滞后（挥臂/加速段更有用），但也可能更抖
    d_cutoff: 导数低通截止频率，常用 1~2
    """
    def __init__(self, min_cutoff=1.2, beta=0.6, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = None
        self.dx_lp = LowPass()
        self.x_lp = LowPass()

    def apply(self, x: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) / max(dt, 1e-6)
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = self.dx_lp.apply(dx, a_d)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = self.x_lp.apply(x, a)

        self.x_prev = x_hat
        return x_hat


# -----------------------------
# Smoothing helpers
# -----------------------------
def _interp_short(arr: np.ndarray, gap_max: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    s = pd.Series(arr)
    return s.interpolate(limit=int(gap_max), limit_direction="both").to_numpy()


def _centered_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
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
    out = np.where(cnt > 0, sm / cnt, np.nan)
    return out


def _smooth_sequence(
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
    mask = (~np.isfinite(arr)) | (vis < v_thr)
    arr[mask] = np.nan

    mode = (mode or "zero_phase").lower()
    if mode == "raw":
        return arr

    if mode == "zero_phase":
        interp = _interp_short(arr, gap_max=gap_max)
        sm = _centered_moving_average(interp, window=window)
        sm[np.isnan(interp)] = np.nan
        return sm

    # causal mode (OneEuro) with visibility gate + gap reset
    out = np.full_like(arr, np.nan, dtype=np.float64)
    filt = OneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    for i in range(len(arr)):
        if not np.isfinite(arr[i]):
            out[i] = np.nan
            filt.x_prev = None
            continue
        out[i] = filt.apply(arr[i], dt)
    return out


# -----------------------------
# Drawing utilities
# -----------------------------
POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22)
]


def draw_pose(frame, lm_xy, color=(0, 255, 0), thickness=2):
    h, w = frame.shape[:2]
    for a, b in POSE_CONNECTIONS:
        if a not in lm_xy or b not in lm_xy:
            continue
        ax, ay = lm_xy[a]
        bx, by = lm_xy[b]
        cv2.line(frame, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)),
                 color, thickness, cv2.LINE_AA)

    # joints
    for _, (x, y) in lm_xy.items():
        cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 0, 255), -1, cv2.LINE_AA)


def put_label(frame, text, org=(20, 60), color=(255, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2, cv2.LINE_AA)


def build_lm_xy(x_row, y_row, v_row, v_thr: float):
    lm_xy = {}
    for i in range(len(x_row)):
        x = x_row[i]
        y = y_row[i]
        v = v_row[i]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if v < v_thr:
            continue
        lm_xy[i] = (x, y)
    return lm_xy


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_video", required=True)

    ap.add_argument("--num_poses", type=int, default=2)
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--min_presence", type=float, default=0.5)
    ap.add_argument("--min_track", type=float, default=0.6)

    # OneEuro args (你要调“滞后/抖动”就调这三个)
    ap.add_argument("--min_cutoff", type=float, default=1.2)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--d_cutoff", type=float, default=1.0)

    # visibility gating（挥臂时偶发低置信度会跳点）
    ap.add_argument("--vis_gate", type=float, default=0.6)
    ap.add_argument("--v_thr", type=float, default=None, help="Visibility threshold (overrides vis_gate).")
    ap.add_argument("--smooth_mode", choices=["raw", "causal", "zero_phase"], default="zero_phase")
    ap.add_argument("--gap_max", type=int, default=3, help="Max gap (frames) for short interpolation.")
    ap.add_argument("--smooth_window", type=int, default=9, help="Centered smoothing window (odd preferred).")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dt = 1.0 / float(fps)

    # MediaPipe PoseLandmarker (VIDEO mode needs timestamp_ms)
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_det,
        min_pose_presence_confidence=args.min_presence,
        min_tracking_confidence=args.min_track
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    v_thr = args.v_thr if args.v_thr is not None else args.vis_gate

    # identity lock: keep the same person across frames
    prev_center = None

    raw_x_list = []
    raw_y_list = []
    vis_list = []
    ts_list = []

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
            # choose best pose (closest to previous center)
            if len(result.pose_landmarks) == 1 or prev_center is None:
                chosen = 0
            else:
                centers = []
                for lms in result.pose_landmarks:
                    if len(lms) > 24:
                        cx = (lms[23].x + lms[24].x) * 0.5
                        cy = (lms[23].y + lms[24].y) * 0.5
                    else:
                        cx = float(np.mean([lm.x for lm in lms]))
                        cy = float(np.mean([lm.y for lm in lms]))
                    centers.append((cx, cy))

                dists = [
                    (centers[i][0] - prev_center[0]) ** 2 + (centers[i][1] - prev_center[1]) ** 2
                    for i in range(len(centers))
                ]
                chosen = int(np.argmin(dists))

            lms = result.pose_landmarks[chosen]

            # update prev center
            if len(lms) > 24:
                prev_center = ((lms[23].x + lms[24].x) * 0.5, (lms[23].y + lms[24].y) * 0.5)

            for i, lm in enumerate(lms):
                arr_x[i] = float(lm.x)
                arr_y[i] = float(lm.y)
                arr_v[i] = float(getattr(lm, "visibility", 1.0))

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
        smooth_x[:, lm_id] = _smooth_sequence(
            raw_x[:, lm_id],
            vis[:, lm_id],
            mode=args.smooth_mode,
            v_thr=v_thr,
            gap_max=args.gap_max,
            window=args.smooth_window,
            dt=dt,
            min_cutoff=args.min_cutoff,
            beta=args.beta,
            d_cutoff=args.d_cutoff,
        )
        smooth_y[:, lm_id] = _smooth_sequence(
            raw_y[:, lm_id],
            vis[:, lm_id],
            mode=args.smooth_mode,
            v_thr=v_thr,
            gap_max=args.gap_max,
            window=args.smooth_window,
            dt=dt,
            min_cutoff=args.min_cutoff,
            beta=args.beta,
            d_cutoff=args.d_cutoff,
        )

    # write compare video (second pass)
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out_video, fourcc, fps, (W * 2, H))
    if not vw.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {args.out_video}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(ts_ms):
            break
        left = frame.copy()
        right = frame.copy()

        raw_xy = build_lm_xy(raw_x[frame_idx], raw_y[frame_idx], vis[frame_idx], v_thr=v_thr)
        smooth_xy = build_lm_xy(smooth_x[frame_idx], smooth_y[frame_idx], vis[frame_idx], v_thr=v_thr)

        draw_pose(left, raw_xy, color=(255, 0, 0), thickness=2)
        draw_pose(right, smooth_xy, color=(0, 255, 0), thickness=2)
        put_label(left, "RAW (BLUE)", color=(255, 200, 80))
        put_label(right, f"SMOOTH {args.smooth_mode} (GREEN)", color=(120, 255, 120))

        compare = np.hstack([left, right])
        vw.write(compare)
        frame_idx += 1

    cap.release()
    vw.release()

    rows = []
    for i in range(len(ts_ms)):
        for lm_id in range(33):
            rows.append({
                "frame": i,
                "ts_ms": int(ts_ms[i]),
                "lm": lm_id,
                "raw_x": raw_x[i, lm_id],
                "raw_y": raw_y[i, lm_id],
                "smooth_x": smooth_x[i, lm_id],
                "smooth_y": smooth_y[i, lm_id],
                "vis": vis[i, lm_id],
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] out_video: {args.out_video}")
    print(f"[OK] out_csv  : {args.out_csv}")



if __name__ == "__main__":
    main()
