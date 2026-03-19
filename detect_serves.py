"""detect_serves_v3.py

Purpose
  Detect tennis serves from a full video using a MediaPipe pose CSV (long format)
  and optionally cut clips + render a visualization video.

Why v3
  Your dataset/camera angle makes the original gates too strict, causing low recall
  (e.g., only ~4-6 serves). v3 exposes the two previously hard-coded gates as CLI
  arguments and relaxes defaults:
    - elbow_high_thresh (shoulder-elbow vertical gap at Trophy)
    - tp_above_shoulder_thresh (shoulder-wrist vertical gap at Trophy)
    - min_elbow_high_frames (how many frames we require elbow_high in the TP window)

Inputs
  pose CSV columns (long): frame, ts_ms, lm, raw_x/raw_y, smooth_x/smooth_y, vis
  (also accepts lm_id and some legacy x/y names).

Outputs
  serve_segments.csv, serve_vis.mp4, and optional serve clips.
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import cv2


# MediaPipe Pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24


@dataclass
class ServeEvent:
    serve_id: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    BR_frame: int
    TP_frame: int
    RLP_frame: int
    BI_frame: int
    BR_ms: int
    TP_ms: int
    RLP_ms: int
    BI_ms: int


def _derivative(x: np.ndarray, ts_ms: np.ndarray) -> np.ndarray:
    t = ts_ms.astype(np.float64) / 1000.0
    dt = np.gradient(t)
    dt[dt == 0] = 1e-6
    return np.gradient(x.astype(np.float64)) / dt


def _pick_cols(df: pd.DataFrame, detect_on: str):
    detect_on = (detect_on or "raw").lower()
    if detect_on == "raw":
        cand_x = ["raw_x", "x_raw", "x"]
        cand_y = ["raw_y", "y_raw", "y"]
        cand_z = ["raw_z", "z_raw", "z"]
    else:
        cand_x = ["smooth_x", "x_s", "x_smooth", "x"]
        cand_y = ["smooth_y", "y_s", "y_smooth", "y"]
        cand_z = ["smooth_z", "z_s", "z_smooth", "z"]

    xcol = next((c for c in cand_x if c in df.columns), None)
    ycol = next((c for c in cand_y if c in df.columns), None)
    zcol = next((c for c in cand_z if c in df.columns), None)
    if xcol is None or ycol is None:
        raise ValueError(
            "Cannot find usable coordinate columns in pose_csv. "
            f"detect_on={detect_on}, columns={df.columns.tolist()}"
        )
    return xcol, ycol, zcol


def load_pose_wide(pose_csv: str, detect_on: str, fps: Optional[float] = None):
    """Load long-format pose CSV and pivot into wide arrays/DataFrames.

    Returns:
      frames (N,), ts_ms (N,), X (N,33), Y (N,33), V (N,33) as pandas DataFrames.
    """
    df = pd.read_csv(pose_csv, encoding="utf-8-sig")
    if "lm_id" not in df.columns and "lm" in df.columns:
        df = df.rename(columns={"lm": "lm_id"})

    if "pose_id" not in df.columns:
        df["pose_id"] = 0

    xcol, ycol, _ = _pick_cols(df, detect_on)
    vcol = "vis" if "vis" in df.columns else ("visibility" if "visibility" in df.columns else None)

    # pick best pose_id if multiple (highest mean visibility on key joints)
    key_ids = [L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP]
    df_key = df[df["lm_id"].isin(key_ids)].copy()
    if vcol:
        pid = int(df_key.groupby("pose_id")[vcol].mean().sort_values(ascending=False).index[0])
    else:
        pid = int(df["pose_id"].mode().iloc[0])
    df = df[df["pose_id"] == pid].copy()

    # pivot
    px = df.pivot(index="frame", columns="lm_id", values=xcol)
    py = df.pivot(index="frame", columns="lm_id", values=ycol)
    pv = df.pivot(index="frame", columns="lm_id", values=vcol) if vcol else None

    # ensure sorted columns
    cols = sorted([int(c) for c in px.columns])
    px = px.reindex(columns=cols)
    py = py.reindex(columns=cols)
    if pv is None:
        pv = pd.DataFrame(np.ones_like(px.values, dtype=np.float32), index=px.index, columns=cols)
    else:
        pv = pv.reindex(columns=cols)

    # interpolate missing
    px = px.astype("float32").interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    py = py.astype("float32").interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    pv = pv.astype("float32").fillna(0.0)

    frames = px.index.to_numpy(dtype=np.int32)
    fallback_fps = float(fps) if fps and fps > 0 else 60.0
    if "ts_ms" in df.columns:
        ts_map = df.drop_duplicates(subset=["frame"])[["frame", "ts_ms"]].set_index("frame")
        ts_ms = ts_map.reindex(frames)["ts_ms"].to_numpy(dtype=np.int64)
        if np.any(pd.isna(ts_ms)):
            ts_ms = frames.astype(np.float64) * (1000.0 / fallback_fps)
    else:
        ts_ms = frames.astype(np.float64) * (1000.0 / fallback_fps)

    # return as DataFrames to keep .columns = lm ids
    return frames, ts_ms, px, py, pv


def detect_serves(
    frames: np.ndarray,
    ts_ms: np.ndarray,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    V: pd.DataFrame,
    hand: str,
    toss_up_thresh: float,
    toss_vy_thresh: float,
    min_racket_drop: float,
    min_contact_above_shoulder: float,
    min_toss_peak: float,
    min_gap_ms: int,
    min_len_ms: int,
    max_len_ms: int,
    elbow_high_thresh: float,
    tp_above_shoulder_thresh: float,
    min_elbow_high_frames: int,
    contact_window: int = 10,
    shoulder_ref: str = 'side',
    bi_mode: str = 'speed',
    debug: int = 0,
):
    # Number of frames (used for window clipping)
    n = int(len(ts_ms))
    if hand.lower() == "right":
        racket_wrist = R_WRIST
        toss_wrist = L_WRIST
        racket_elbow = R_ELBOW
        racket_shoulder = R_SHOULDER
        toss_shoulder = L_SHOULDER
    else:
        racket_wrist = L_WRIST
        toss_wrist = R_WRIST
        racket_elbow = L_ELBOW
        racket_shoulder = L_SHOULDER
        toss_shoulder = R_SHOULDER

    needed = [toss_wrist, toss_shoulder, racket_wrist, racket_elbow, racket_shoulder, L_HIP, R_HIP]
    for k in needed:
        if k not in Y.columns:
            raise ValueError(f"Missing landmark id {k} in pose csv.")

    toss_y = Y[toss_wrist].to_numpy()
    toss_sh_y = Y[toss_shoulder].to_numpy()
    racket_y = Y[racket_wrist].to_numpy()
    racket_sh_y = Y[racket_shoulder].to_numpy()
    racket_el_y = Y[racket_elbow].to_numpy()

    # Optional: use the average of both shoulders as a more stable reference
    # (helps when the side shoulder jitters during fast arm swing).
    if shoulder_ref == 'avg':
        center_sh_y = 0.5 * (Y[L_SHOULDER].to_numpy() + Y[R_SHOULDER].to_numpy())
        toss_sh_y = center_sh_y
        racket_sh_y = center_sh_y

    toss_vy = _derivative(toss_y, ts_ms)
    racket_speed = np.sqrt(
        _derivative(X[racket_wrist].to_numpy(), ts_ms) ** 2 + _derivative(racket_y, ts_ms) ** 2
    )

    toss_up = (toss_sh_y - toss_y) > float(toss_up_thresh)
    rising = toss_vy < float(toss_vy_thresh)
    toss_start = np.where(toss_up & rising)[0]

    serve_candidates = []
    last_ms = -10**18
    for i in toss_start:
        if ts_ms[i] - last_ms >= int(min_gap_ms):
            serve_candidates.append(int(i))
            last_ms = int(ts_ms[i])

    if debug:
        print(f"[DBG] toss_start={len(toss_start)}  candidates(after gap)={len(serve_candidates)}")

    def _clip_idx(i0, i1):
        i0 = max(0, int(i0))
        i1 = min(len(frames) - 1, int(i1))
        return i0, i1

    events: list[ServeEvent] = []
    sid = 0

    # debug counters for gates
    g_tp_elbow = g_len = g_tp_above = g_drop = g_contact = g_tosspeak = 0

    for i0 in serve_candidates:
        # BR
        w0, w1 = _clip_idx(i0, i0 + 180)
        br = None
        for i in range(w0 + 1, w1):
            if toss_up[i] and toss_vy[i - 1] < 0 and toss_vy[i] > 0:
                br = i
                break
        if br is None:
            br = int(w0 + np.argmin(toss_y[w0:w1 + 1]))

        # TP
        tp_w0, tp_w1 = _clip_idx(br, br + 150)
        elbow_high = (racket_sh_y - racket_el_y) > float(elbow_high_thresh)
        cand = np.where(elbow_high[tp_w0:tp_w1 + 1])[0]
        if len(cand) < int(min_elbow_high_frames):
            g_tp_elbow += 1
            continue

        sub = racket_y[tp_w0:tp_w1 + 1].copy()
        mask = np.ones_like(sub, dtype=bool)
        mask[cand] = False
        sub[mask] = 1.0
        tp = int(tp_w0 + np.argmin(sub))

        # RLP
        rlp_w0, rlp_w1 = _clip_idx(tp + 1, tp + 220)
        rlp = int(rlp_w0 + np.argmax(racket_y[rlp_w0:rlp_w1 + 1]))

        # BI
        bi_w0, bi_w1 = _clip_idx(rlp + 1, rlp + 120)
        if bi_mode == 'ymin':
            # choose the highest racket wrist position (smallest y) after racket low point
            bi = int(bi_w0 + np.argmin(racket_y[bi_w0:bi_w1 + 1]))
        else:
            # choose the peak speed frame after racket low point
            bi = int(bi_w0 + np.argmax(racket_speed[bi_w0:bi_w1 + 1]))

        # segment bounds
        start_ms = int(ts_ms[br] - 700)
        end_ms = int(ts_ms[bi] + 700)
        start_idx = int(np.searchsorted(ts_ms, start_ms, side="left"))
        end_idx = int(np.searchsorted(ts_ms, end_ms, side="right") - 1)
        start_idx, end_idx = _clip_idx(start_idx, end_idx)

        seg_len = int(ts_ms[end_idx] - ts_ms[start_idx])
        if seg_len < int(min_len_ms) or seg_len > int(max_len_ms):
            g_len += 1
            continue

        # TP should be above shoulder (relaxed & parameterized)
        if float(racket_sh_y[tp] - racket_y[tp]) < float(tp_above_shoulder_thresh):
            g_tp_above += 1
            continue

        # gates for serve-vs-forehand/backhand
        drop_amp = float(racket_y[rlp] - racket_y[tp])
        if drop_amp < float(min_racket_drop):
            g_drop += 1
            continue

        cw = int(contact_window)
        j0 = max(0, bi - cw)
        j1 = min(n - 1, bi + cw)
        contact_above = float(np.nanmax(racket_sh_y[j0:j1 + 1] - racket_y[j0:j1 + 1]))
        if contact_above < float(min_contact_above_shoulder):
            g_contact += 1
            continue

        if min_toss_peak >= 0:
            toss_peak = float(toss_sh_y[br] - toss_y[br])
            if toss_peak < float(min_toss_peak):
                g_tosspeak += 1
                continue

        sid += 1
        events.append(
            ServeEvent(
                serve_id=sid,
                start_frame=int(frames[start_idx]),
                end_frame=int(frames[end_idx]),
                start_ms=int(ts_ms[start_idx]),
                end_ms=int(ts_ms[end_idx]),
                BR_frame=int(frames[br]),
                TP_frame=int(frames[tp]),
                RLP_frame=int(frames[rlp]),
                BI_frame=int(frames[bi]),
                BR_ms=int(ts_ms[br]),
                TP_ms=int(ts_ms[tp]),
                RLP_ms=int(ts_ms[rlp]),
                BI_ms=int(ts_ms[bi]),
            )
        )

    if debug:
        print(
            "[DBG] dropped_by_gate: "
            f"tp_elbow={g_tp_elbow}, len={g_len}, tp_above={g_tp_above}, "
            f"drop={g_drop}, contact={g_contact}, tosspeak={g_tosspeak}"
        )

    return events


def write_segments_csv(events: list[ServeEvent], out_csv: str):
    rows = []
    for e in events:
        rows.append({
            "serve_id": e.serve_id,
            "start_frame": e.start_frame, "end_frame": e.end_frame,
            "start_ms": e.start_ms, "end_ms": e.end_ms,
            "BR_frame": e.BR_frame, "TP_frame": e.TP_frame, "RLP_frame": e.RLP_frame, "BI_frame": e.BI_frame,
            "BR_ms": e.BR_ms, "TP_ms": e.TP_ms, "RLP_ms": e.RLP_ms, "BI_ms": e.BI_ms,
        })
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    try:
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return str(out_csv)
    except PermissionError:
        base, ext = os.path.splitext(str(out_csv))
        for k in range(1, 51):
            alt = f"{base}_{k}{ext}"
            try:
                df.to_csv(alt, index=False, encoding="utf-8-sig")
                print(f"[WARN] segments csv locked. Wrote to {alt}")
                return alt
            except PermissionError:
                continue
        raise


def render_vis(video_path: str, pose_csv: str, events: list[ServeEvent], out_mp4: str, hand: str, draw_on: str):
    df = pd.read_csv(pose_csv, encoding="utf-8-sig")
    if "lm_id" not in df.columns and "lm" in df.columns:
        df = df.rename(columns={"lm": "lm_id"})
    if "pose_id" not in df.columns:
        df["pose_id"] = 0

    xcol, ycol, _ = _pick_cols(df, draw_on)
    vcol = "vis" if "vis" in df.columns else None

    # choose best pose_id
    key_ids = [L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP]
    df_key = df[df["lm_id"].isin(key_ids)]
    if vcol:
        pid = int(df_key.groupby("pose_id")[vcol].mean().sort_values(ascending=False).index[0])
    else:
        pid = int(df["pose_id"].mode().iloc[0])
    df = df[df["pose_id"] == pid].copy()

    df = df[["frame", "lm_id", xcol, ycol]].copy()
    df.rename(columns={xcol: "x", ycol: "y"}, inplace=True)

    frame_map = {}
    for f, g in df.groupby("frame"):
        frame_map[int(f)] = dict(zip(g["lm_id"].astype(int).tolist(), zip(g["x"].tolist(), g["y"].tolist())))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))

    # event lookup
    ev_by_frame = {}
    for e in events:
        for fr in range(e.start_frame, e.end_frame + 1):
            ev_by_frame[fr] = e

    # simple skeleton edges (subset)
    edges = [
        (L_SHOULDER, R_SHOULDER),
        (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
        (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
        (L_HIP, R_HIP),
    ]

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr = frame_idx
        pts = frame_map.get(fr)
        if pts:
            for a, b in edges:
                if a in pts and b in pts:
                    ax, ay = pts[a]
                    bx, by = pts[b]
                    cv2.line(frame, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)), (0, 255, 0), 2)
        if fr in ev_by_frame:
            e = ev_by_frame[fr]
            cv2.putText(frame, f"Serve {e.serve_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def cut_clips(video_path: str, events: list[ServeEvent], out_dir: str):
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # build quick index: frame -> event id
    by_id = {e.serve_id: e for e in events}

    for sid, e in by_id.items():
        out_mp4 = os.path.join(clips_dir, f"serve_{sid:03d}.mp4")
        writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, e.start_frame)
        for fr in range(e.start_frame, e.end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()

    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--detect_on", default="raw", choices=["raw", "smooth"])
    ap.add_argument("--draw_on", default="smooth", choices=["raw", "smooth"])

    ap.add_argument("--toss_up_thresh", type=float, default=0.03)
    ap.add_argument("--toss_vy_thresh", type=float, default=-0.06)
    ap.add_argument("--min_racket_drop", type=float, default=0.03)
    ap.add_argument("--min_contact_above_shoulder", type=float, default=0.015)
    ap.add_argument("--min_toss_peak", type=float, default=-1.0)
    ap.add_argument("--contact_window", type=int, default=10, help="Max window (+/- frames) used to evaluate contact height.")
    ap.add_argument("--shoulder_ref", choices=["side", "avg"], default="side", help="Reference shoulder for height gates: side shoulder or average of both.")
    ap.add_argument("--bi_mode", choices=["speed", "ymin"], default="speed", help="How to pick the BI frame after RLP: peak speed or highest wrist (min y).")

    ap.add_argument("--min_gap_ms", type=int, default=800)
    ap.add_argument("--min_len_ms", type=int, default=650)
    ap.add_argument("--max_len_ms", type=int, default=7000)

    # newly exposed gates (key for low-recall issue)
    ap.add_argument("--elbow_high_thresh", type=float, default=0.01)
    ap.add_argument("--tp_above_shoulder_thresh", type=float, default=0.015)
    ap.add_argument("--min_elbow_high_frames", type=int, default=2)

    ap.add_argument("--make_clips", type=int, default=1)
    ap.add_argument("--debug", type=int, default=1)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fps = args.fps
    if fps is None:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        cap.release()

    frames, ts_ms, X, Y, V = load_pose_wide(args.pose_csv, args.detect_on, fps=fps)

    events = detect_serves(
        frames, ts_ms, X, Y, V,
        hand=args.hand,
        toss_up_thresh=args.toss_up_thresh,
        toss_vy_thresh=args.toss_vy_thresh,
        min_racket_drop=args.min_racket_drop,
        min_contact_above_shoulder=args.min_contact_above_shoulder,
        min_toss_peak=args.min_toss_peak,
        min_gap_ms=args.min_gap_ms,
        min_len_ms=args.min_len_ms,
        max_len_ms=args.max_len_ms,
        elbow_high_thresh=args.elbow_high_thresh,
        tp_above_shoulder_thresh=args.tp_above_shoulder_thresh,
        min_elbow_high_frames=args.min_elbow_high_frames,
        contact_window=args.contact_window,
            shoulder_ref=args.shoulder_ref,
            bi_mode=args.bi_mode,
            debug=args.debug,
    )

    print(f"[OK] Detected serves: {len(events)}")

    segments_csv = os.path.join(args.out_dir, "serve_segments.csv")
    write_segments_csv(events, segments_csv)
    print("[OK] saved:", segments_csv)

    out_vis = os.path.join(args.out_dir, "serve_vis.mp4")
    render_vis(args.video, args.pose_csv, events, out_vis, args.hand, args.draw_on)
    print("[OK] saved:", out_vis)

    if int(args.make_clips) == 1:
        cut_clips(args.video, events, args.out_dir)
        print("[OK] clips ->", os.path.join(args.out_dir, "clips"))


if __name__ == "__main__":
    main()
    ap.add_argument("--fps", type=float, default=None, help="Override fps used for ts_ms fallback (default: use video fps or 60).")
