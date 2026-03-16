import os, math, argparse
import cv2
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


# -----------------------------
# 1€ (One Euro) Filter
# -----------------------------
def _alpha(cutoff_hz: float, dt: float) -> float:
    # cutoff -> smoothing factor
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)

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
    min_cutoff: 越大越稳(但可能更“肉”)
    beta: 越大越能在“快速运动”时减少滞后（适合挥臂）
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
# Drawing utilities
# -----------------------------
POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),
    (11,12),(11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(28,30),(29,31),(30,32),
    (15,17),(15,19),(15,21),
    (16,18),(16,20),(16,22)
]

def draw_pose(frame, lm_xy, color=(0,255,0), thickness=2):
    h, w = frame.shape[:2]
    for a,b in POSE_CONNECTIONS:
        if a not in lm_xy or b not in lm_xy:
            continue
        ax, ay = lm_xy[a]
        bx, by = lm_xy[b]
        cv2.line(frame, (int(ax*w), int(ay*h)), (int(bx*w), int(by*h)), color, thickness, cv2.LINE_AA)
    for i,(x,y) in lm_xy.items():
        cv2.circle(frame, (int(x*w), int(y*h)), 3, (0,0,255), -1, cv2.LINE_AA)

def put_label(frame, text, org=(20,60), color=(255,255,255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 2, cv2.LINE_AA)


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
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # OneEuro filters per landmark per coord
    # 给“挥臂”更友好：beta 稍大；min_cutoff 中等
    filters = {}  # (lm_id, coord) -> OneEuro
    def get_filter(lm_id, coord):
        key = (lm_id, coord)
        if key not in filters:
            # 手腕/肘更容易抖：可以更“稳”一点
            if lm_id in (13,14,15,16):
                filters[key] = OneEuro(min_cutoff=1.4, beta=0.8, d_cutoff=1.0)
            else:
                filters[key] = OneEuro(min_cutoff=1.2, beta=0.6, d_cutoff=1.0)
        return filters[key]

    # identity lock：防止教练抢走“主目标”
    prev_center = None

    rows = []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out_video, fourcc, fps, (W*2, H))

    frame_idx = 0
    dt = 1.0 / fps

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = int(frame_idx * 1000.0 / fps)

        mp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_frame  # numpy array ok for Image?
        image = mp_image

        # Use MediaPipe Image wrapper
        mp_img = vision.Image(image_format=vision.ImageFormat.SRGB, data=image)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        raw_xy = {}
        smooth_xy = {}

        # choose best pose (closest to previous center)
        chosen = None
        if result.pose_landmarks:
            if len(result.pose_landmarks) == 1 or prev_center is None:
                chosen = 0
            else:
                # compute center by hips (23,24) if available
                centers = []
                for pi, lms in enumerate(result.pose_landmarks):
                    if len(lms) > 24:
                        cx = (lms[23].x + lms[24].x) * 0.5
                        cy = (lms[23].y + lms[24].y) * 0.5
                    else:
                        cx = np.mean([lm.x for lm in lms])
                        cy = np.mean([lm.y for lm in lms])
                    centers.append((cx, cy))
                dists = [ (centers[i][0]-prev_center[0])**2 + (centers[i][1]-prev_center[1])**2 for i in range(len(centers)) ]
                chosen = int(np.argmin(dists))
            lms = result.pose_landmarks[chosen]

            # update prev center
            if len(lms) > 24:
                prev_center = ((lms[23].x + lms[24].x) * 0.5, (lms[23].y + lms[24].y) * 0.5)

            for i, lm in enumerate(lms):
                raw_xy[i] = (lm.x, lm.y)

                # visibility gating：置信度太差就“别更新”，减少挥臂瞬间跳点
                vis = getattr(lm, "visibility", 1.0)
                x, y = lm.x, lm.y
                if vis < 0.35 and i in (13,14,15,16):  # 手臂关键点
                    # 不更新：用上一帧滤波状态输出（如果有）
                    fx = get_filter(i, "x")
                    fy = get_filter(i, "y")
                    x_s = fx.x_lp.y if fx.x_lp.inited else x
                    y_s = fy.x_lp.y if fy.x_lp.inited else y
                else:
                    x_s = get_filter(i, "x").apply(x, dt)
                    y_s = get_filter(i, "y").apply(y, dt)

                smooth_xy[i] = (x_s, y_s)

                rows.append({
                    "frame": frame_idx,
                    "ts_ms": ts_ms,
                    "lm": i,
                    "raw_x": x, "raw_y": y,
                    "smooth_x": x_s, "smooth_y": y_s,
                    "vis": vis,
                })

        # build compare frame
        left = frame.copy()
        right = frame.copy()
        draw_pose(left, raw_xy, color=(255, 0, 0), thickness=2)    # BLUE = RAW
        draw_pose(right, smooth_xy, color=(0, 255, 0), thickness=2) # GREEN = SMOOTH

        put_label(left, "RAW", color=(255, 200, 80))
        put_label(right, "SMOOTH", color=(120, 255, 120))

        compare = np.hstack([left, right])
        vw.write(compare)

        frame_idx += 1

    cap.release()
    vw.release()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] out_video: {args.out_video}")
    print(f"[OK] out_csv  : {args.out_csv}")

if __name__ == "__main__":
    main()
