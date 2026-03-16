import argparse
import os

import cv2
import numpy as np
import pandas as pd

import serve_score


L_WRIST = 15
R_WRIST = 16

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
]


def _max_gap(mask: np.ndarray) -> int:
    max_gap = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            max_gap = max(max_gap, cur)
        else:
            cur = 0
    return int(max_gap)


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if "lm" not in df.columns and "lm_id" in df.columns:
        df = df.rename(columns={"lm_id": "lm"})
    return df.pivot_table(index="frame", columns="lm", values=col, aggfunc="mean").sort_index()


def _draw_pose(frame, lm_xy, color=(0, 255, 0), thickness=2):
    h, w = frame.shape[:2]
    for a, b in POSE_CONNECTIONS:
        if a not in lm_xy or b not in lm_xy:
            continue
        ax, ay = lm_xy[a]
        bx, by = lm_xy[b]
        cv2.line(frame, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)),
                 color, thickness, cv2.LINE_AA)
    for _, (x, y) in lm_xy.items():
        cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 0, 255), -1, cv2.LINE_AA)


def _build_lm_xy(x_row, y_row, v_row, v_thr: float):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_csv", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out_dir", default=r"D:\Serve_Score\out\verify_smoothing")
    ap.add_argument("--v_thr", type=float, default=0.6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.pose_csv, encoding="utf-8-sig")

    y_raw = _pivot(df, "raw_y")
    y_smooth = _pivot(df, "smooth_y")
    vis = _pivot(df, "vis") if "vis" in df.columns else None

    frames = y_raw.index.to_numpy()
    ts_ms = None
    if "ts_ms" in df.columns:
        ts_map = df.drop_duplicates(subset=["frame"])[["frame", "ts_ms"]].set_index("frame")
        ts_ms = ts_map.reindex(frames)["ts_ms"].to_numpy(dtype=float)
    if ts_ms is None or np.any(np.isnan(ts_ms)):
        ts_ms = frames.astype(float) * (1000.0 / 60.0)
    t = ts_ms / 1000.0

    rw_raw = y_raw.get(R_WRIST).to_numpy()
    rw_smooth = y_smooth.get(R_WRIST).to_numpy()
    rw_vis = vis.get(R_WRIST).to_numpy() if vis is not None and R_WRIST in vis.columns else np.ones_like(rw_raw)

    low_vis = ~np.isfinite(rw_vis) | (rw_vis < args.v_thr)
    occlusion_gap = _max_gap(low_vis)
    nan_frames = int(np.sum(~np.isfinite(rw_smooth)))
    nan_ratio = float(np.mean(low_vis))

    print("[verify_smoothing]")
    print(f"right_wrist_nan_ratio={nan_ratio:.3f}")
    print(f"occlusion_gap_frames={occlusion_gap}")
    print(f"num_nan_frames={nan_frames}")

    # timing check via scoring
    result = serve_score.analyze_serve(df, use_coords="smooth")
    timing = {
        "timing_toss_to_trophy": result["metrics"].get("timing_toss_to_trophy"),
        "timing_trophy_to_drop": result["metrics"].get("timing_trophy_to_drop"),
        "timing_drop_to_contact": result["metrics"].get("timing_drop_to_contact"),
    }
    print("timing_metrics:", timing)

    # plot raw vs smooth
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 3))
    plt.plot(t, rw_raw, label="raw", color="#2c7fb8", linewidth=1.5, alpha=0.8)
    plt.plot(t, rw_smooth, label="smooth", color="#e34a33", linewidth=2.0)
    plt.legend()
    plt.title("Right Wrist Y: raw vs smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Y (norm)")
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "right_wrist_y.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[OK] plot: {plot_path}")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")

        target = 0
        if occlusion_gap > 0:
            # pick middle frame of the longest low-vis gap
            max_gap = 0
            cur = 0
            end_idx = 0
            for i, v in enumerate(low_vis):
                if v:
                    cur += 1
                    if cur > max_gap:
                        max_gap = cur
                        end_idx = i
                else:
                    cur = 0
            target = max(0, end_idx - max_gap // 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, frame = cap.read()
        if ok:
            # build overlay using smooth points, gated by visibility
            x_s = _pivot(df, "smooth_x")
            y_s = y_smooth
            v_s = vis if vis is not None else pd.DataFrame(np.ones_like(x_s.values), index=x_s.index, columns=x_s.columns)
            x_row = x_s.loc[frames[target]].to_numpy()
            y_row = y_s.loc[frames[target]].to_numpy()
            v_row = v_s.loc[frames[target]].to_numpy()
            lm_xy = _build_lm_xy(x_row, y_row, v_row, v_thr=args.v_thr)
            _draw_pose(frame, lm_xy, color=(0, 255, 0), thickness=2)

            out_img = os.path.join(args.out_dir, f"overlay_frame_{target:04d}.png")
            cv2.imwrite(out_img, frame)
            print(f"[OK] overlay keyframe: {out_img}")
        cap.release()


if __name__ == "__main__":
    main()
