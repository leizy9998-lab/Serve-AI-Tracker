import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# MediaPipe Pose landmark indices
LM = {
    "nose": 0,
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

DEFAULT_WEIGHTS = {
    "toss": 0.2,
    "trophy": 0.15,
    "knee_bend": 0.15,
    "shoulder_tilt": 0.1,
    "racket_drop": 0.15,
    "contact": 0.15,
    "timing": 0.1,
}


def long_to_wide(df: pd.DataFrame, use_coords: str = "smooth") -> Dict[str, np.ndarray]:
    if use_coords not in {"raw", "smooth"}:
        raise ValueError("use_coords must be 'raw' or 'smooth'")

    x_col = f"{use_coords}_x"
    y_col = f"{use_coords}_y"
    required = {"frame", "ts_ms", "lm", x_col, y_col, "vis"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"pose csv missing columns: {sorted(missing)}")

    df = df.sort_values(["frame", "lm"]).reset_index(drop=True)
    frames = df["frame"].dropna().unique()
    frames.sort()
    frame_to_idx = {int(f): i for i, f in enumerate(frames)}
    t_len = len(frames)

    x = np.full((t_len, 33), np.nan, dtype=float)
    y = np.full((t_len, 33), np.nan, dtype=float)
    vis = np.full((t_len, 33), np.nan, dtype=float)

    for _, row in df.iterrows():
        f = int(row["frame"])
        lm = int(row["lm"])
        idx = frame_to_idx.get(f)
        if idx is None or lm < 0 or lm >= 33:
            continue
        x[idx, lm] = float(row[x_col])
        y[idx, lm] = float(row[y_col])
        vis[idx, lm] = float(row["vis"])

    ts_ms = df.groupby("frame")["ts_ms"].median().reindex(frames).values.astype(float)

    return {"frames": frames, "ts_ms": ts_ms, "x": x, "y": y, "vis": vis}


def _fill_nan(arr: np.ndarray, gap_max: int = 3) -> np.ndarray:
    if arr.size == 0:
        return arr
    s = pd.Series(arr)
    if s.isna().all():
        return arr
    return s.interpolate(limit=int(gap_max), limit_direction="both").to_numpy()


def _median_filter_1d(arr: np.ndarray, k: int = 5) -> np.ndarray:
    if arr.size == 0 or k <= 1:
        return arr
    k = int(k)
    if k % 2 == 0:
        k += 1
    half = k // 2
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        window = arr[lo:hi]
        if np.all(np.isnan(window)):
            out[i] = np.nan
        else:
            out[i] = np.nanmedian(window)
    return out


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return float("nan")
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _series_angle(x: np.ndarray, y: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        pa = np.array([x[i, a], y[i, a]])
        pb = np.array([x[i, b], y[i, b]])
        pc = np.array([x[i, c], y[i, c]])
        out[i] = _angle_deg(pa, pb, pc)
    return out


def _side_indices(side: str) -> Dict[str, int]:
    side = (side or "left").lower()
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    return {
        "wrist": LM[f"{side}_wrist"],
        "elbow": LM[f"{side}_elbow"],
        "shoulder": LM[f"{side}_shoulder"],
        "hip": LM[f"{side}_hip"],
        "knee": LM[f"{side}_knee"],
        "ankle": LM[f"{side}_ankle"],
    }


def _infer_toss_side(left_wrist_y: np.ndarray, right_wrist_y: np.ndarray) -> Tuple[str, Dict[str, float]]:
    def _score(wy: np.ndarray) -> Dict[str, float]:
        n = len(wy)
        if n == 0:
            return {"score": -1.0, "peak_idx": 0, "rise_idx": 0, "height": 0.0}
        w = _median_filter_1d(_fill_nan(wy), k=5)
        peak_search_end = max(1, int(n * 0.7))
        peak_idx = int(np.nanargmax(w[:peak_search_end]))
        base = np.nanmedian(w[: max(3, int(n * 0.1))])
        delta = 0.03
        rise_idx = peak_idx
        for i in range(0, peak_idx):
            if np.isfinite(w[i]) and w[i] >= base + delta:
                rise_idx = i
                break
        height = float(w[peak_idx] - base)
        early = 1.0 - (rise_idx / max(peak_search_end, 1))
        score = height * 2.0 + early
        return {"score": score, "peak_idx": peak_idx, "rise_idx": rise_idx, "height": height}

    l = _score(left_wrist_y)
    r = _score(right_wrist_y)
    if r["score"] > l["score"]:
        return "right", {"left": l["score"], "right": r["score"]}
    return "left", {"left": l["score"], "right": r["score"]}


def compute_signals(
    wide: Dict[str, np.ndarray],
    median_k: int = 5,
    toss_side: str = "left",
    racket_side: str = "right",
    gap_max: int = 3,
    v_thr: float = 0.6,
) -> Dict[str, np.ndarray]:
    x = wide["x"]
    y = wide["y"]
    vis = wide["vis"]

    x_m = x.copy()
    y_m = y.copy()
    if vis is not None:
        mask = (~np.isfinite(vis)) | (vis < v_thr)
        x_m[mask] = np.nan
        y_m[mask] = np.nan
    y_up = 1.0 - y_m

    left_wrist_y = y_up[:, LM["left_wrist"]]
    right_wrist_y = y_up[:, LM["right_wrist"]]
    left_shoulder_y = y_up[:, LM["left_shoulder"]]
    right_shoulder_y = y_up[:, LM["right_shoulder"]]
    nose_y = y_up[:, LM["nose"]]
    left_hip_y = y_up[:, LM["left_hip"]]
    right_hip_y = y_up[:, LM["right_hip"]]
    left_ankle_y = y_up[:, LM["left_ankle"]]
    right_ankle_y = y_up[:, LM["right_ankle"]]
    left_ankle_vis = vis[:, LM["left_ankle"]] if vis is not None else np.full(y.shape[0], np.nan)
    right_ankle_vis = vis[:, LM["right_ankle"]] if vis is not None else np.full(y.shape[0], np.nan)
    nose_vis = vis[:, LM["nose"]] if vis is not None else np.full(y.shape[0], np.nan)
    left_wrist_vis = vis[:, LM["left_wrist"]] if vis is not None else np.full(y.shape[0], np.nan)
    right_wrist_vis = vis[:, LM["right_wrist"]] if vis is not None else np.full(y.shape[0], np.nan)
    left_shoulder_vis = vis[:, LM["left_shoulder"]] if vis is not None else np.full(y.shape[0], np.nan)
    right_shoulder_vis = vis[:, LM["right_shoulder"]] if vis is not None else np.full(y.shape[0], np.nan)
    left_hip_vis = vis[:, LM["left_hip"]] if vis is not None else np.full(y.shape[0], np.nan)
    right_hip_vis = vis[:, LM["right_hip"]] if vis is not None else np.full(y.shape[0], np.nan)

    toss_idx = _side_indices(toss_side)
    racket_idx = _side_indices(racket_side)

    toss_wrist_y = y_up[:, toss_idx["wrist"]]
    racket_wrist_y = y_up[:, racket_idx["wrist"]]
    racket_elbow_y = y_up[:, racket_idx["elbow"]]
    toss_shoulder_y = y_up[:, toss_idx["shoulder"]]
    racket_shoulder_y = y_up[:, racket_idx["shoulder"]]
    racket_wrist_x = x_m[:, racket_idx["wrist"]]

    shoulder_tilt_signed = right_shoulder_y - left_shoulder_y
    shoulder_tilt_abs = np.abs(shoulder_tilt_signed)
    elbow_angle = _series_angle(x_m, y_m, racket_idx["shoulder"], racket_idx["elbow"], racket_idx["wrist"])

    left_knee_angle = _series_angle(x_m, y_m, LM["left_hip"], LM["left_knee"], LM["left_ankle"])
    right_knee_angle = _series_angle(x_m, y_m, LM["right_hip"], LM["right_knee"], LM["right_ankle"])
    knee_angle = np.nanmin(np.vstack([left_knee_angle, right_knee_angle]), axis=0)

    # trunk inclination (angle from vertical, degrees)
    trunk_dx = 0.5 * (x_m[:, LM["left_shoulder"]] + x_m[:, LM["right_shoulder"]]) - 0.5 * (
        x_m[:, LM["left_hip"]] + x_m[:, LM["right_hip"]]
    )
    trunk_dy = 0.5 * (y_up[:, LM["left_shoulder"]] + y_up[:, LM["right_shoulder"]]) - 0.5 * (
        y_up[:, LM["left_hip"]] + y_up[:, LM["right_hip"]]
    )
    trunk_angle = np.degrees(np.arctan2(np.abs(trunk_dx), np.maximum(trunk_dy, 1e-6)))

    signals = {
        "left_wrist_y": left_wrist_y,
        "right_wrist_y": right_wrist_y,
        "left_shoulder_y": left_shoulder_y,
        "right_shoulder_y": right_shoulder_y,
        "nose_y": nose_y,
        "left_hip_y": left_hip_y,
        "right_hip_y": right_hip_y,
        "left_ankle_y": left_ankle_y,
        "right_ankle_y": right_ankle_y,
        "left_ankle_vis": left_ankle_vis,
        "right_ankle_vis": right_ankle_vis,
        "nose_vis": nose_vis,
        "left_wrist_vis": left_wrist_vis,
        "right_wrist_vis": right_wrist_vis,
        "racket_wrist_vis": vis[:, racket_idx["wrist"]] if vis is not None else np.full(y.shape[0], np.nan),
        "toss_wrist_vis": vis[:, toss_idx["wrist"]] if vis is not None else np.full(y.shape[0], np.nan),
        "left_shoulder_vis": left_shoulder_vis,
        "right_shoulder_vis": right_shoulder_vis,
        "left_hip_vis": left_hip_vis,
        "right_hip_vis": right_hip_vis,
        "v_thr": float(v_thr),
        "shoulder_tilt": shoulder_tilt_signed,
        "shoulder_tilt_abs": shoulder_tilt_abs,
        "elbow_angle": elbow_angle,
        "knee_angle": knee_angle,
        "toss_wrist_y": toss_wrist_y,
        "racket_wrist_y": racket_wrist_y,
        "racket_elbow_y": racket_elbow_y,
        "toss_shoulder_y": toss_shoulder_y,
        "racket_shoulder_y": racket_shoulder_y,
        "racket_wrist_x": racket_wrist_x,
        "racket_elbow_height": racket_elbow_y - racket_shoulder_y,
        "trunk_angle": trunk_angle,
    }

    for key, series in signals.items():
        if not isinstance(series, np.ndarray):
            continue
        series = _fill_nan(series, gap_max=gap_max)
        series = _median_filter_1d(series, k=median_k)
        signals[key] = series

    return signals


def detect_phases(signals: Dict[str, np.ndarray], ts_ms: np.ndarray) -> Dict[str, Optional[int]]:
    toss_y = signals["toss_wrist_y"]
    racket_y = signals["racket_wrist_y"]
    elbow_y = signals["racket_elbow_y"]
    elbow_angle = signals["elbow_angle"]
    elbow_height = signals["racket_elbow_height"]
    nose_y = signals["nose_y"]
    racket_x = signals["racket_wrist_x"]

    n = len(toss_y)
    if n == 0:
        return {
            "toss_start": None,
            "toss_peak": None,
            "trophy": None,
            "racket_drop": None,
            "contact": None,
            "finish": None,
        }

    valid_toss = np.isfinite(toss_y)
    valid_racket = np.isfinite(racket_y)
    if not valid_toss.any() or not valid_racket.any():
        return {
            "toss_start": 0,
            "toss_peak": int(np.nanargmax(toss_y)) if valid_toss.any() else 0,
            "trophy": None,
            "racket_drop": None,
            "contact": int(np.nanargmax(racket_y)) if valid_racket.any() else n - 1,
            "finish": n - 1,
        }

    peak_search_end = max(1, int(n * 0.7))
    toss_peak = int(np.nanargmax(toss_y[:peak_search_end]))

    baseline = np.nanmedian(toss_y[: max(3, int(n * 0.1))])
    delta = 0.03
    toss_start = 0
    for i in range(0, toss_peak):
        if np.isfinite(toss_y[i]) and toss_y[i] >= baseline + delta:
            toss_start = i
            break

    dt_ms = float(np.nanmedian(np.diff(ts_ms))) if len(ts_ms) > 1 else 16.7
    fps = 1000.0 / max(dt_ms, 1e-6)
    contact_window = max(10, int(800.0 / max(dt_ms, 1e-6)))

    # speed (finite difference)
    t = ts_ms.astype(np.float64) / 1000.0
    dt = np.gradient(t)
    dt[dt == 0] = 1e-6
    vx = np.gradient(racket_x.astype(np.float64)) / dt
    vy = np.gradient(racket_y.astype(np.float64)) / dt
    speed = np.sqrt(vx * vx + vy * vy)

    # provisional contact from toss_peak window
    contact = toss_peak
    contact_status = "ok"
    cand = np.arange(toss_peak + 1, min(n, toss_peak + contact_window + 1))
    cand = cand[np.isfinite(racket_y[cand])]
    if len(cand) > 0:
        head_ref = nose_y.copy()
        head_ref[~np.isfinite(head_ref)] = np.nanmax(
            np.vstack([signals["left_shoulder_y"], signals["right_shoulder_y"]]),
            axis=0,
        )[~np.isfinite(head_ref)]
        head_gap = racket_y[cand] - head_ref[cand]
        cand_pos = cand[head_gap > 0]
        if len(cand_pos) > 0:
            cand = cand_pos
        y_val = racket_y[cand]
        s_val = speed[cand]
        head_gap = racket_y[cand] - head_ref[cand]
        y_std = np.nanstd(y_val)
        s_std = np.nanstd(s_val)
        y_z = (y_val - np.nanmean(y_val)) / (y_std if y_std > 1e-6 else 1.0)
        s_z = (s_val - np.nanmean(s_val)) / (s_std if s_std > 1e-6 else 1.0)
        bonus = np.where(head_gap > 0, 0.6, 0.0)
        score = y_z + 0.6 * s_z + bonus
        contact = int(cand[int(np.nanargmax(score))])
    else:
        contact = int(np.nanargmax(racket_y)) if valid_racket.any() else n - 1

    def _norm(val: float, vmin: float, vmax: float) -> float:
        if not np.isfinite(val):
            return 0.0
        return (val - vmin) / max(vmax - vmin, 1e-6)

    # Trophy (HRP) detection: multi-cue
    trophy_status = "ok"
    knee_flexion = 180.0 - signals["knee_angle"]
    trunk_inc = signals["trunk_angle"]
    win_start = max(0, toss_start)
    win_end = max(win_start + 1, min(n - 1, contact - 6))
    win_idx = np.arange(win_start, win_end + 1)
    knee_win = knee_flexion[win_idx]
    knee_thr = np.nanpercentile(knee_win[np.isfinite(knee_win)], 90) if np.isfinite(knee_win).any() else np.nan
    local_max = []
    for i in range(win_start + 1, win_end):
        if np.isfinite(knee_flexion[i]) and knee_flexion[i] >= knee_flexion[i - 1] and knee_flexion[i] >= knee_flexion[i + 1]:
            local_max.append(i)
    cand = set(local_max)
    if np.isfinite(knee_thr):
        cand.update([i for i in win_idx if np.isfinite(knee_flexion[i]) and knee_flexion[i] >= knee_thr])
    if not cand:
        cand = {int(win_idx[int(np.nanargmax(knee_win))])} if np.isfinite(knee_win).any() else {win_start}
    cand = np.array(sorted(cand), dtype=int)

    knee_min = np.nanmin(knee_win) if np.isfinite(knee_win).any() else 0.0
    knee_max = np.nanmax(knee_win) if np.isfinite(knee_win).any() else 1.0
    elbow_min = np.nanmin(elbow_y[win_idx]) if np.isfinite(elbow_y[win_idx]).any() else 0.0
    elbow_max = np.nanmax(elbow_y[win_idx]) if np.isfinite(elbow_y[win_idx]).any() else 1.0
    speed_min = np.nanmin(speed[win_idx]) if np.isfinite(speed[win_idx]).any() else 0.0
    speed_max = np.nanmax(speed[win_idx]) if np.isfinite(speed[win_idx]).any() else 1.0
    wrist_min = np.nanmin(racket_y[win_idx]) if np.isfinite(racket_y[win_idx]).any() else 0.0
    wrist_max = np.nanmax(racket_y[win_idx]) if np.isfinite(racket_y[win_idx]).any() else 1.0
    min_wrist_after = np.nanmin(racket_y[win_start:contact + 1]) if contact > win_start else np.nan

    scores = []
    for idx in cand:
        knee_norm = _norm(knee_flexion[idx], knee_min, knee_max) if np.isfinite(knee_flexion[idx]) else 0.0
        trunk_z = abs((trunk_inc[idx] - 25.0) / 7.1) if np.isfinite(trunk_inc[idx]) else 2.0
        trunk_norm = max(0.0, 1.0 - trunk_z / 2.0)
        elbow_norm = _norm(elbow_y[idx], elbow_min, elbow_max) if np.isfinite(elbow_y[idx]) else 0.0
        speed_norm = _norm(speed[idx], speed_min, speed_max) if np.isfinite(speed[idx]) else 1.0
        wrist_high = _norm(racket_y[idx], wrist_min, wrist_max) if np.isfinite(racket_y[idx]) else 0.0
        wrist_not_low = 1.0 if np.isfinite(min_wrist_after) and np.isfinite(racket_y[idx]) and (racket_y[idx] > min_wrist_after + 0.02) else 0.0
        score = (
            0.30 * knee_norm
            + 0.20 * trunk_norm
            + 0.25 * elbow_norm
            + 0.15 * (1.0 - speed_norm)
            + 0.10 * wrist_high
            + 0.05 * wrist_not_low
        )
        scores.append(score)
    trophy = int(cand[int(np.nanargmax(scores))])
    if trophy >= contact:
        trophy = max(win_start, contact - 6)
        trophy_status = "fallback"

    # Drop (LRP) detection
    drop_status = "ok"
    drop_start = min(n - 1, trophy + 3)
    drop_end = max(drop_start + 1, contact - 2)
    drop_end = min(drop_end, n - 2)
    drop = drop_start
    sub = racket_y[drop_start:drop_end + 1]
    if np.isfinite(sub).any():
        drop = drop_start + int(np.nanargmin(sub))
    else:
        sub = elbow_y[drop_start:drop_end + 1]
        if np.isfinite(sub).any():
            drop = drop_start + int(np.nanargmin(sub))
        else:
            drop = max(trophy + 1, contact - 2)
            drop_status = "fallback"

    # HRP/LRP swap check (auto correction)
    def _score_hrp(idx: int) -> float:
        return (
            _norm(elbow_y[idx], elbow_min, elbow_max)
            + _norm(knee_flexion[idx], knee_min, knee_max)
            - (1.0 - _norm(racket_y[idx], wrist_min, wrist_max))
            - _norm(speed[idx], speed_min, speed_max)
        )

    def _score_lrp(idx: int) -> float:
        wrist_low = 1.0 - _norm(racket_y[idx], wrist_min, wrist_max)
        speed_low = 1.0 - _norm(speed[idx], speed_min, speed_max)
        return wrist_low + speed_low

    hrp_t = _score_hrp(trophy)
    hrp_d = _score_hrp(drop)
    lrp_t = _score_lrp(trophy)
    lrp_d = _score_lrp(drop)
    swapped = False
    if hrp_t < hrp_d and lrp_d < lrp_t:
        trophy, drop = drop, trophy
        swapped = True
        trophy_status = "swap"
        drop_status = "swap"
        drop_start = min(n - 1, trophy + 3)
        drop_end = max(drop_start + 1, contact - 2)
        drop_end = min(drop_end, n - 2)
        sub = racket_y[drop_start:drop_end + 1]
        if np.isfinite(sub).any():
            drop = drop_start + int(np.nanargmin(sub))
        else:
            sub = elbow_y[drop_start:drop_end + 1]
            if np.isfinite(sub).any():
                drop = drop_start + int(np.nanargmin(sub))
            else:
                drop = max(trophy + 1, contact - 2)
                drop_status = "fallback"

    # Enforce ordering and minimum frame gaps
    if drop - trophy < 3 or drop <= trophy:
        cand2 = cand[cand <= max(win_start, drop - 3)] if isinstance(cand, np.ndarray) else np.array([], dtype=int)
        if cand2.size > 0:
            scores2 = []
            for idx in cand2:
                knee_norm = _norm(knee_flexion[idx], knee_min, knee_max) if np.isfinite(knee_flexion[idx]) else 0.0
                trunk_z = abs((trunk_inc[idx] - 25.0) / 7.1) if np.isfinite(trunk_inc[idx]) else 2.0
                trunk_norm = max(0.0, 1.0 - trunk_z / 2.0)
                elbow_norm = _norm(elbow_y[idx], elbow_min, elbow_max) if np.isfinite(elbow_y[idx]) else 0.0
                speed_norm = _norm(speed[idx], speed_min, speed_max) if np.isfinite(speed[idx]) else 1.0
                wrist_high = _norm(racket_y[idx], wrist_min, wrist_max) if np.isfinite(racket_y[idx]) else 0.0
                score = (
                    0.30 * knee_norm
                    + 0.20 * trunk_norm
                    + 0.25 * elbow_norm
                    + 0.15 * (1.0 - speed_norm)
                    + 0.10 * wrist_high
                )
                scores2.append(score)
            trophy = int(cand2[int(np.nanargmax(scores2))])
            trophy_status = "adjusted"
        else:
            trophy = max(win_start, drop - 3)
            trophy_status = "fallback"
        drop_start = min(n - 1, trophy + 3)
        drop_end = max(drop_start + 1, contact - 2)
        drop_end = min(drop_end, n - 2)
        sub = racket_y[drop_start:drop_end + 1]
        if np.isfinite(sub).any():
            drop = drop_start + int(np.nanargmin(sub))
        else:
                drop = max(trophy + 1, contact - 2)
                drop_status = "fallback"

    # Contact sanity: must be after drop + 2
    if contact <= drop + 1:
        search_start = min(n - 2, drop + 2)
        search_end = min(n - 1, drop + 30)
        speed_seg = speed[search_start:search_end]
        if np.isfinite(speed_seg).any():
            contact = search_start + int(np.nanargmax(speed_seg)) + 1
        else:
            sub = racket_y[search_start:search_end + 1]
            if np.isfinite(sub).any():
                contact = search_start + int(np.nanargmax(sub))
            else:
                contact = min(n - 1, drop + 2)
                contact_status = "fallback"

    # Finish detection: speed settles
    finish_status = "ok"
    finish = contact
    if contact < n - 1:
        post = speed[contact + 1:]
        if np.isfinite(post).any():
            v_end = np.nanpercentile(post, 20)
            N = 8
            consec = 0
            found = None
            for i in range(contact + 1, n):
                if np.isfinite(speed[i]) and speed[i] < v_end:
                    consec += 1
                    if consec >= N:
                        found = i
                        break
                else:
                    consec = 0
            if found is not None:
                finish = found
            else:
                finish = min(n - 1, contact + int(round(0.35 * fps)))
                finish_status = "fallback"
        else:
            finish = min(n - 1, contact + int(round(0.35 * fps)))
            finish_status = "fallback"

    print(
        "[DEBUG] HRP/LRP scores: trophy=%d drop=%d hrp_t=%.3f hrp_d=%.3f lrp_t=%.3f lrp_d=%.3f swapped=%s"
        % (trophy, drop, hrp_t, hrp_d, lrp_t, lrp_d, str(swapped))
    )

    return {
        "toss_start": int(toss_start),
        "toss_peak": int(toss_peak),
        "trophy": int(trophy),
        "racket_drop": int(drop),
        "contact": int(contact),
        "finish": int(finish),
        "trophy_status": trophy_status,
        "drop_status": drop_status,
        "contact_status": contact_status,
        "finish_status": finish_status,
    }


def _score_trapezoid(value: float, ideal_low: float, ideal_high: float, min_low: float, max_high: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    if value < min_low or value > max_high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return 100.0 * (value - min_low) / max(ideal_low - min_low, 1e-6)
    return 100.0 * (max_high - value) / max(max_high - ideal_high, 1e-6)


def _score_inverse(value: float, ideal_max: float, worst_max: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    if value <= ideal_max:
        return 100.0
    if value >= worst_max:
        return 0.0
    return 100.0 * (worst_max - value) / max(worst_max - ideal_max, 1e-6)


def _compute_body_scale(signals: Dict[str, np.ndarray]) -> float:
    ls = signals.get("left_shoulder_y")
    rs = signals.get("right_shoulder_y")
    lh = signals.get("left_hip_y")
    rh = signals.get("right_hip_y")
    la = signals.get("left_ankle_y")
    ra = signals.get("right_ankle_y")

    if ls is None or rs is None or lh is None or rh is None:
        return float("nan")

    shoulder_mid = 0.5 * (ls + rs)
    hip_mid = 0.5 * (lh + rh)
    scale_series = np.abs(shoulder_mid - hip_mid)
    scale = float(np.nanmedian(scale_series))

    if not np.isfinite(scale) or scale < 1e-6:
        if la is not None and ra is not None:
            ankle_mid = 0.5 * (la + ra)
            scale_series = np.abs(shoulder_mid - ankle_mid)
            scale = float(np.nanmedian(scale_series))
    return scale


def fix_hrp_lrp(
    phases: Dict[str, Optional[int]],
    signals: Dict[str, np.ndarray],
    ts_ms: np.ndarray,
    fps: float,
) -> Dict[str, Optional[int]]:
    """
    Propulsion = HRP->LRP, Forward swing = LRP->Impact. (Tubez 2019)

    强制修正策略（按你的要求）：
    1) 奖杯位(HRP) = 原“拍头下落”帧（旧 drop）
    2) 在 HRP 之后、击球(contact) 之前，重新搜索真正的拍头下落(LRP)
    3) 自动判断 y 的方向（y-down or y-up），避免把最低点选反
    """
    trophy_idx = phases.get("trophy")
    drop_idx = phases.get("racket_drop") or phases.get("racket_drop_idx") or phases.get("drop")
    contact_idx = phases.get("contact") or phases.get("impact")
    if trophy_idx is None or drop_idx is None or contact_idx is None:
        return phases

    old_trophy = int(trophy_idx)
    old_drop = int(drop_idx)
    contact_idx = int(contact_idx)

    t = lambda k: float(ts_ms[int(k)]) / 1000.0

    phases_out = dict(phases)
    phases_out["old_trophy"] = int(old_trophy)
    phases_out["old_drop"] = int(old_drop)

    # ---------- helper: infer y axis direction ----------
    def _infer_y_is_down() -> Optional[bool]:
        """
        返回 True 表示 y 越大越靠下（常见图像坐标系）；
        返回 False 表示 y 越大越靠上（你项目若提前做过 1-y 翻转可能会这样）；
        返回 None 表示无法判断（将默认按 y-down 处理）。
        """
        nose = signals.get("nose_y")
        la = signals.get("left_ankle_y")
        ra = signals.get("right_ankle_y")

        if nose is None:
            return None

        diffs = []
        for ankle in (la, ra):
            if ankle is None:
                continue
            m = np.isfinite(nose) & np.isfinite(ankle)
            if m.any():
                diffs.append(float(np.nanmedian(ankle[m] - nose[m])))

        if diffs:
            # y-down: ankle_y 通常 > nose_y ； y-up: ankle_y 通常 < nose_y
            return float(np.nanmedian(diffs)) > 0.0

        # 退化：用 hip vs shoulder
        lhip = signals.get("left_hip_y")
        rhip = signals.get("right_hip_y")
        lsho = signals.get("left_shoulder_y")
        rsho = signals.get("right_shoulder_y")
        if lhip is not None and lsho is not None:
            m = np.isfinite(lhip) & np.isfinite(lsho)
            if m.any():
                return float(np.nanmedian(lhip[m] - lsho[m])) > 0.0
        if rhip is not None and rsho is not None:
            m = np.isfinite(rhip) & np.isfinite(rsho)
            if m.any():
                return float(np.nanmedian(rhip[m] - rsho[m])) > 0.0

        return None

    y_is_down = _infer_y_is_down()
    if y_is_down is None:
        y_is_down = True  # 默认按常见图像坐标：y 向下

    # ---------- FORCE: HRP = old_drop ----------
    new_trophy = int(old_drop)

    min_gap_trophy_drop = max(3, int(round(0.05 * fps)))
    min_gap_drop_contact = max(2, int(round(0.03 * fps)))
    start = new_trophy + min_gap_trophy_drop
    end = contact_idx - min_gap_drop_contact

    drop_status = "refind_after_forced_trophy"
    new_drop: Optional[int] = None

    wrist_y = signals.get("racket_wrist_y")
    elbow_y = signals.get("racket_elbow_y")

    def _find_local_extreme(series: Optional[np.ndarray], s: int, e: int) -> Optional[int]:
        """
        LRP = “最低点”：
        - y-down(越大越靠下)：取 argmax
        - y-up(越大越靠上)：取 argmin
        """
        if series is None:
            return None
        seg = series[s : e + 1]
        if not np.isfinite(seg).any():
            return None

        if y_is_down:
            cand = s + int(np.nanargmax(seg))
        else:
            cand = s + int(np.nanargmin(seg))

        lo = max(s, cand - 2)
        hi = min(e, cand + 2)
        local = series[lo : hi + 1]
        if np.isfinite(local).any():
            if y_is_down:
                cand = lo + int(np.nanargmax(local))
            else:
                cand = lo + int(np.nanargmin(local))
        return int(cand)

    if start < end:
        new_drop = _find_local_extreme(wrist_y, start, end)
        if new_drop is None:
            new_drop = _find_local_extreme(elbow_y, start, end)
            if new_drop is not None:
                drop_status = "elbow_fallback"
    else:
        drop_status = "unscorable_window"
        new_drop = None

    # ---------- enforce order & minimal gaps (retry once) ----------
    if new_drop is not None:
        if (
            (new_drop - new_trophy) < min_gap_trophy_drop
            or (contact_idx - new_drop) < min_gap_drop_contact
            or not (new_trophy < new_drop < contact_idx)
        ):
            s2 = new_trophy + min_gap_trophy_drop + 1
            e2 = contact_idx - (min_gap_drop_contact + 2)
            retry = None
            if s2 < e2:
                retry = _find_local_extreme(wrist_y, s2, e2)
                if retry is None:
                    retry = _find_local_extreme(elbow_y, s2, e2)
                    if retry is not None:
                        drop_status = "elbow_fallback"
            if retry is not None:
                new_drop = int(retry)

            if (
                new_drop is None
                or (new_drop - new_trophy) < min_gap_trophy_drop
                or (contact_idx - new_drop) < min_gap_drop_contact
                or not (new_trophy < new_drop < contact_idx)
            ):
                drop_status = "invalid_order"
                new_drop = None

    phases_out["trophy"] = int(new_trophy)
    phases_out["racket_drop"] = None if new_drop is None else int(new_drop)
    phases_out["drop_status"] = drop_status
    phases_out["trophy_status"] = "forced_from_drop"
    phases_out["y_axis"] = "down" if y_is_down else "up"

    # ---------- final hard ordering gate ----------
    t_idx = phases_out.get("trophy")
    d_idx = phases_out.get("racket_drop")
    if t_idx is not None and d_idx is not None:
        if (
            not (t_idx < d_idx < contact_idx)
            or (d_idx - t_idx) < min_gap_trophy_drop
            or (contact_idx - d_idx) < min_gap_drop_contact
        ):
            phases_out["racket_drop"] = None
            phases_out["drop_status"] = "invalid_order"

    # ---------- debug dt ----------
    dt_tt = float("nan")
    if phases_out.get("toss_peak") is not None and phases_out.get("trophy") is not None:
        dt_tt = t(phases_out.get("trophy")) - t(phases_out.get("toss_peak"))

    dt_td_new = float("nan")
    dt_dc_new = float("nan")
    if phases_out.get("trophy") is not None and phases_out.get("racket_drop") is not None:
        dt_td_new = t(phases_out.get("racket_drop")) - t(phases_out.get("trophy"))
    if phases_out.get("racket_drop") is not None:
        dt_dc_new = t(contact_idx) - t(phases_out.get("racket_drop"))

    print(
        "[DEBUG] fix_hrp_lrp: old_trophy=%s old_drop=%s new_trophy=%s new_drop=%s contact=%s y=%s"
        % (
            str(old_trophy),
            str(old_drop),
            str(phases_out.get("trophy")),
            str(phases_out.get("racket_drop")),
            str(contact_idx),
            phases_out.get("y_axis"),
        )
    )
    print(
        "[DEBUG] dt: toss->trophy=%.3f trophy->drop=%.3f drop->contact=%.3f"
        % (dt_tt, dt_td_new, dt_dc_new)
    )
    return phases_out


def _zscore_score(value: float, mean: float, sd: float, one_sided_high: bool = False) -> Tuple[float, float]:
    if not np.isfinite(value) or sd <= 1e-6:
        return float("nan"), float("nan")
    if one_sided_high and value >= mean:
        z = 0.0
    else:
        z = (value - mean) / sd
    score = float(np.exp(-0.5 * z * z) * 100.0)
    return score, float(z)


def _score_lit_plateau(
    value: float,
    mean: float,
    sd: float,
    w_plateau: float = 98.0,
    w_1sd: float = 92.0,
    w_2sd: float = 75.0,
    w_3sd: float = 45.0,
    floor: float = 10.0,
) -> Tuple[float, float]:
    if not np.isfinite(value) or sd <= 1e-6:
        return float("nan"), float("nan")
    z = abs((value - mean) / sd)
    if z <= 1.0:
        score = w_plateau - (w_plateau - w_1sd) * (z / 1.0)
    elif z <= 2.0:
        score = w_1sd - (w_1sd - w_2sd) * ((z - 1.0) / 1.0)
    elif z <= 3.0:
        score = w_2sd - (w_2sd - w_3sd) * ((z - 2.0) / 1.0)
    else:
        score = floor
    return float(score), float(z)


def _window_slice(n: int, start: Optional[int], end: Optional[int]) -> slice:
    if n <= 0:
        return slice(0, 0)
    s = 0 if start is None else max(0, int(start))
    e = n - 1 if end is None else max(0, int(end))
    if e < s:
        s, e = e, s
    return slice(s, min(e + 1, n))


_CALIB_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def _load_calibration_stats() -> Dict[str, Dict[str, float]]:
    global _CALIB_CACHE
    if _CALIB_CACHE is not None:
        return _CALIB_CACHE

    candidates = [
        os.path.join(os.path.dirname(__file__), "calibration"),
        os.path.join(os.path.dirname(__file__), "out", "calibration"),
    ]
    calib_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if not calib_dir:
        _CALIB_CACHE = {}
        return _CALIB_CACHE

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
            biomech_min = biomech.get("metrics_min") or {}

            for key in targets:
                val = biomech_min.get(key, metrics.get(key))
                if isinstance(val, (int, float)) and np.isfinite(val):
                    values[key].append(float(val))

    stats: Dict[str, Dict[str, float]] = {}
    for key, series in values.items():
        if len(series) < 8:
            continue
        arr = np.asarray(series, dtype=float)
        p5, p15, p85, p95 = np.nanpercentile(arr, [5, 15, 85, 95])
        stats[key] = {"p5": float(p5), "p15": float(p15), "p85": float(p85), "p95": float(p95)}

    _CALIB_CACHE = stats
    return stats


def _score_percentile_plateau(value: float, stats: Dict[str, float], floor: float = 10.0) -> float:
    if not np.isfinite(value) or not stats:
        return float("nan")
    p5, p15, p85, p95 = stats.get("p5"), stats.get("p15"), stats.get("p85"), stats.get("p95")
    if not all(np.isfinite(x) for x in [p5, p15, p85, p95]):
        return float("nan")
    if value <= p5:
        return floor
    if value <= p15:
        return float(60.0 + (value - p5) / (p15 - p5) * 35.0)
    if value <= p85:
        return 95.0
    if value <= p95:
        return float(95.0 - (value - p85) / (p95 - p85) * 35.0)
    return floor


def compute_biomech_scores(
    signals: Dict[str, np.ndarray],
    metrics: Dict[str, float],
    phases: Dict[str, Optional[int]],
    quantile_calib: bool = False,
    fps: float = 60.0,
) -> Dict[str, object]:
    knee_angle = signals.get("knee_angle")
    trunk_angle = signals.get("trunk_angle")
    elbow_angle = signals.get("elbow_angle")
    shoulder_tilt = signals.get("shoulder_tilt")

    n = len(signals.get("toss_wrist_y", []))
    win = _window_slice(n, phases.get("toss_peak"), phases.get("contact"))

    knee_series = knee_angle[win] if knee_angle is not None else np.array([])
    trunk_series = trunk_angle[win] if trunk_angle is not None else np.array([])
    elbow_series = elbow_angle[win] if elbow_angle is not None else np.array([])
    shoulder_series = shoulder_tilt[win] if shoulder_tilt is not None else np.array([])

    knee_min = float(np.nanmin(knee_series)) if knee_series.size else float("nan")
    knee_flexion_raw = 180.0 - knee_min if np.isfinite(knee_min) else float("nan")
    # 2D projection bias correction (cap to avoid extreme values)
    knee_flexion_est = float(np.clip(knee_flexion_raw / 0.7, 0.0, 120.0)) if np.isfinite(knee_flexion_raw) else float("nan")

    trunk_mean = float(np.nanmean(trunk_series)) if trunk_series.size else float("nan")
    trunk_incl_est = float(np.clip(trunk_mean / 0.6, 0.0, 80.0)) if np.isfinite(trunk_mean) else float("nan")
    body_scale = metrics.get("body_scale", 1.0)
    if not np.isfinite(body_scale) or body_scale <= 1e-6:
        body_scale = 1.0

    def _finite_ratio(series: np.ndarray) -> Tuple[int, int, float]:
        total = int(series.size)
        if total == 0:
            return 0, 0, 0.0
        valid = int(np.isfinite(series).sum())
        return valid, total, valid / total

    knee_valid, knee_total, knee_conf = _finite_ratio(knee_series)
    trunk_valid, trunk_total, trunk_conf = _finite_ratio(trunk_series)

    v_thr = float(signals.get("v_thr", 0.6))

    impact_bh_ratio = float("nan")
    impact_source = "none"
    impact_valid = 0
    impact_total = 0
    impact_conf = 0.0
    impact_vis = float("nan")
    std_local = float("nan")
    impact_status = "可评估"
    impact_reason = ""

    contact_idx = phases.get("contact")
    standing_h = float("nan")
    if contact_idx is not None and n > 0:
        win_radius = 2
        start = max(0, int(contact_idx) - win_radius)
        end = min(n, int(contact_idx) + win_radius + 1)
        if end > start:
            head_y = signals.get("nose_y")
            ls_y = signals.get("left_shoulder_y")
            rs_y = signals.get("right_shoulder_y")
            wrist_y = signals.get("racket_wrist_y")
            elbow_y = signals.get("racket_elbow_y")
            la_y = signals.get("left_ankle_y")
            ra_y = signals.get("right_ankle_y")

            nose_seg = head_y[start:end] if head_y is not None else None
            shoulder_seg = (
                np.nanmax(np.vstack([ls_y[start:end], rs_y[start:end]]), axis=0)
                if ls_y is not None and rs_y is not None
                else None
            )
            if nose_seg is None:
                head_seg = shoulder_seg
            elif shoulder_seg is None:
                head_seg = nose_seg
            else:
                head_seg = np.where(np.isfinite(nose_seg), nose_seg, shoulder_seg)

            if la_y is not None and ra_y is not None:
                ankle_seg = np.nanmean(np.vstack([la_y[start:end], ra_y[start:end]]), axis=0)
            elif la_y is not None:
                ankle_seg = la_y[start:end]
            elif ra_y is not None:
                ankle_seg = ra_y[start:end]
            else:
                ankle_seg = None

            # estimate standing height from early stable window
            early_end = int(phases.get("toss_start") or phases.get("toss_peak") or max(4, int(n * 0.2)))
            early_end = max(4, min(n, early_end))
            head_full = head_y[:early_end] if head_y is not None else None
            ankle_full = None
            if la_y is not None and ra_y is not None:
                ankle_full = np.nanmean(np.vstack([la_y[:early_end], ra_y[:early_end]]), axis=0)
            elif la_y is not None:
                ankle_full = la_y[:early_end]
            elif ra_y is not None:
                ankle_full = ra_y[:early_end]
            if head_full is not None and ankle_full is not None:
                denom = ankle_full - head_full
                vis_head = signals.get("nose_vis")
                vis_la = signals.get("left_ankle_vis")
                vis_ra = signals.get("right_ankle_vis")
                ankle_vis = np.nanmean(
                    np.vstack([vis_la[:early_end], vis_ra[:early_end]]), axis=0
                ) if vis_la is not None and vis_ra is not None else np.nan
                valid_mask = (
                    np.isfinite(denom)
                    & (np.abs(denom) > 1e-4)
                    & (vis_head[:early_end] >= v_thr)
                    & (ankle_vis >= v_thr)
                ) if vis_head is not None else (np.isfinite(denom) & (np.abs(denom) > 1e-4))
                if valid_mask.any():
                    standing_h = float(np.nanmedian(denom[valid_mask]))

            def _ratio_from(series: np.ndarray) -> Tuple[float, int, int, float, float]:
                if series is None or head_seg is None or ankle_seg is None:
                    return float("nan"), 0, 0, 0.0, float("nan")
                seg = series[start:end]
                denom = ankle_seg - head_seg
                valid_mask = (
                    np.isfinite(seg)
                    & np.isfinite(head_seg)
                    & np.isfinite(ankle_seg)
                    & (np.abs(denom) > 1e-4)
                )
                total = int(seg.size)
                valid = int(valid_mask.sum())
                ratio = valid / total if total > 0 else 0.0
                if valid == 0:
                    return float("nan"), valid, total, ratio, float("nan")
                height_ref = standing_h if np.isfinite(standing_h) and standing_h > 1e-4 else np.nanmedian(denom[valid_mask])
                if not np.isfinite(height_ref) or height_ref <= 1e-4:
                    return float("nan"), valid, total, ratio, float("nan")
                bh_values = (ankle_seg[valid_mask] - seg[valid_mask]) / height_ref
                return float(np.nanmedian(bh_values)), valid, total, ratio, float(np.nanstd(bh_values))

            wrist_val, wrist_valid, wrist_total, wrist_ratio, wrist_std = _ratio_from(wrist_y)
            elbow_val, elbow_valid, elbow_total, elbow_ratio, elbow_std = _ratio_from(elbow_y)
            thr = 0.4
            if wrist_ratio >= thr:
                impact_bh_ratio = wrist_val
                impact_source = "wrist"
                impact_valid = wrist_valid
                impact_total = wrist_total
                impact_conf = wrist_ratio
                std_local = wrist_std
            elif elbow_ratio >= thr:
                impact_bh_ratio = elbow_val
                impact_source = "elbow_fallback"
                impact_valid = elbow_valid
                impact_total = elbow_total
                impact_conf = elbow_ratio
                std_local = elbow_std

            vis_wrist = signals.get("racket_wrist_vis")
            vis_nose = signals.get("nose_vis")
            vis_la = signals.get("left_ankle_vis")
            vis_ra = signals.get("right_ankle_vis")
            if vis_la is not None and vis_ra is not None:
                ankle_vis = float(np.nanmean([vis_la[contact_idx], vis_ra[contact_idx]]))
            else:
                ankle_vis = float("nan")
            if vis_wrist is not None and vis_nose is not None:
                impact_vis = float(
                    np.nanmean(
                        [
                            float(vis_wrist[contact_idx]),
                            float(vis_nose[contact_idx]),
                            ankle_vis,
                        ]
                    )
                )

    if np.isfinite(impact_vis):
        impact_conf = min(impact_conf, impact_vis) if np.isfinite(impact_conf) and impact_conf > 0 else impact_vis

    # optional quantile calibration (per-serve)
    if quantile_calib and np.isfinite(impact_bh_ratio) and n > 0:
        series_vals = []
        start = max(0, int(phases.get("toss_peak", 0)))
        end = min(n, int(phases.get("contact", n - 1)) + 1)
        if end > start:
            head_y = signals.get("nose_y")
            ls_y = signals.get("left_shoulder_y")
            rs_y = signals.get("right_shoulder_y")
            wrist_y = signals.get("racket_wrist_y")
            la_y = signals.get("left_ankle_y")
            ra_y = signals.get("right_ankle_y")
            if head_y is not None and wrist_y is not None and la_y is not None and ra_y is not None:
                head_seg = np.where(
                    np.isfinite(head_y[start:end]),
                    head_y[start:end],
                    np.nanmax(np.vstack([ls_y[start:end], rs_y[start:end]]), axis=0),
                )
                ankle_seg = np.nanmean(np.vstack([la_y[start:end], ra_y[start:end]]), axis=0)
                denom = ankle_seg - head_seg
                valid_mask = (
                    np.isfinite(wrist_y[start:end])
                    & np.isfinite(head_seg)
                    & np.isfinite(ankle_seg)
                    & (np.abs(denom) > 1e-4)
                )
                if valid_mask.any():
                    series_vals = ((ankle_seg - wrist_y[start:end]) / (ankle_seg - head_seg))[valid_mask]
        if isinstance(series_vals, np.ndarray) and series_vals.size >= 5:
            p10, p90 = np.nanpercentile(series_vals, [10, 90])
            impact_bh_ratio = float(np.clip(impact_bh_ratio, p10, p90))

    # timing (seconds)
    t1 = metrics.get("timing_toss_to_trophy", float("nan"))
    t2 = metrics.get("timing_trophy_to_drop", float("nan"))
    t3 = metrics.get("timing_drop_to_contact", float("nan"))
    timing_invalid = False
    timing_values = {"timing_toss_to_trophy": t1, "timing_trophy_to_drop": t2, "timing_drop_to_contact": t3}
    for k, v in timing_values.items():
        if np.isfinite(v) and v < 0:
            timing_values[k] = float("nan")
            timing_invalid = True
    t1, t2, t3 = timing_values["timing_toss_to_trophy"], timing_values["timing_trophy_to_drop"], timing_values["timing_drop_to_contact"]
    timing_redetected = False
    trophy_idx = phases.get("trophy")
    drop_idx = phases.get("racket_drop")
    contact_idx = phases.get("contact")
    wrist_y = signals.get("racket_wrist_y")
    if wrist_y is not None and trophy_idx is not None and contact_idx is not None:
        min_dt_prop = max(3.0 / fps, 0.05) if np.isfinite(fps) and fps > 0 else 0.05
        min_dt_fwd = max(2.0 / fps, 0.03) if np.isfinite(fps) and fps > 0 else 0.03
        needs_redetect = False
        if np.isfinite(t2) and (t2 < min_dt_prop or t2 > 0.50):
            needs_redetect = True
        if np.isfinite(t3) and (t3 < min_dt_fwd or t3 > 0.35):
            needs_redetect = True
        if needs_redetect:
            k = 4
            k2 = 2
            start = int(min(len(wrist_y) - 1, trophy_idx + k))
            end = int(max(start + 1, contact_idx - k2))
            if end > start:
                window = wrist_y[start:end]
                if np.isfinite(window).any():
                    drop_rel = int(np.nanargmin(window))
                    drop_idx = start + drop_rel
                    # re-pick contact if it does not follow drop
                    if contact_idx <= drop_idx + 1:
                        speed = np.abs(np.diff(wrist_y))
                        search_start = min(len(speed) - 1, drop_idx + 2)
                        if search_start < len(speed):
                            speed_seg = speed[search_start:]
                            if np.isfinite(speed_seg).any():
                                contact_idx = search_start + int(np.nanargmax(speed_seg)) + 1
                    if contact_idx > drop_idx:
                        t2 = (drop_idx - trophy_idx) / fps if np.isfinite(fps) and fps > 0 else float("nan")
                        t3 = (contact_idx - drop_idx) / fps if np.isfinite(fps) and fps > 0 else float("nan")
                        timing_redetected = True
    if quantile_calib:
        timing_vals = np.array([v for v in [t1, t2, t3] if np.isfinite(v)], dtype=float)
        if timing_vals.size >= 2:
            p10, p90 = np.nanpercentile(timing_vals, [10, 90])
            t1 = float(np.clip(t1, p10, p90)) if np.isfinite(t1) else t1
            t2 = float(np.clip(t2, p10, p90)) if np.isfinite(t2) else t2
            t3 = float(np.clip(t3, p10, p90)) if np.isfinite(t3) else t3

    timing_valid = sum(1 for v in [t1, t2, t3] if np.isfinite(v))
    timing_conf = timing_valid / 3.0

    # Literature-grounded plateau scoring (Jacquier-Bret & Gorce 2024)
    # Note: single-camera 2D cannot capture full kinetic chain (Kovacs 2011).
    # Knee flexion magnitude relates to performance proxies (Hornestam 2021).
    knee_score, knee_z = _score_lit_plateau(knee_flexion_est, 64.5, 9.7)
    trunk_score, trunk_z = _score_lit_plateau(trunk_incl_est, 25.0, 7.1)

    # impact height ratio (Mendes 2013; player height : impact height ~1:1.5)
    impact_raw_score = _score_trapezoid(impact_bh_ratio, 1.45, 1.55, 1.25, 1.75)
    impact_cap_reason = ""
    impact_valid_flag = True
    if not np.isfinite(impact_bh_ratio):
        impact_valid_flag = False
    if np.isfinite(impact_vis) and impact_vis < v_thr:
        impact_valid_flag = False
    if np.isfinite(impact_bh_ratio) and (impact_bh_ratio < 1.20 or impact_bh_ratio > 1.85):
        impact_valid_flag = False
    if np.isfinite(std_local) and std_local > 0.035:
        impact_valid_flag = False

    if not impact_valid_flag:
        impact_score = float("nan")
        impact_status = "不可评估"
        impact_reason = "关键帧/遮挡/尺度估计不可靠"
    else:
        impact_score = min(impact_raw_score, 92.0)

    # timing scores (Tubez 2019; approximate)
    t1_score = _score_trapezoid(t1, 0.30, 1.40, 0.05, 1.80)
    t2_score = _score_trapezoid(t2, 0.12, 0.26, 0.06, 0.40)
    t3_score = _score_trapezoid(t3, 0.08, 0.16, 0.05, 0.22)
    timing_raw = float(np.nanmean([t1_score, t2_score, t3_score]))
    phase_collapse = False
    if np.isfinite(fps) and fps > 1e-6:
        min_dt = 2.0 / fps
        for v, lim_lo, lim_hi in [
            (t2, max(3.0 / fps, 0.05), 0.50),
            (t3, max(2.0 / fps, 0.03), 0.35),
        ]:
            if np.isfinite(v) and (v < lim_lo or v > lim_hi):
                phase_collapse = True
                break
    timing_cap_reason = ""
    timing_status = "可评估"
    if phase_collapse:
        timing_score = float("nan")
        timing_status = "不可评估"
        timing_cap_reason = "关键事件塌缩/错位（例如 trophy->drop 仅 1 帧）"
    else:
        timing_score = min(timing_raw, 90.0)

    # timing confidence: keypoint visibility + phase collapse
    timing_vis = []
    toss_idx = phases.get("toss_peak")
    trophy_idx = phases.get("trophy")
    drop_idx = phases.get("racket_drop")
    contact_idx = phases.get("contact")
    toss_vis = signals.get("toss_wrist_vis")
    racket_vis = signals.get("racket_wrist_vis")
    if toss_vis is not None and toss_idx is not None and 0 <= toss_idx < len(toss_vis):
        timing_vis.append(toss_vis[toss_idx])
    for idx in [trophy_idx, drop_idx, contact_idx]:
        if racket_vis is not None and idx is not None and 0 <= idx < len(racket_vis):
            timing_vis.append(racket_vis[idx])
    vis_mean = float(np.nanmean(timing_vis)) if timing_vis else float("nan")
    if np.isfinite(vis_mean):
        timing_conf = min(timing_conf, vis_mean)
    if phase_collapse:
        timing_conf = min(timing_conf, 0.5)

    # stability penalties
    knee_std = float(np.nanstd(knee_flexion_raw)) if knee_series.size else float("nan")
    trunk_std = float(np.nanstd(trunk_series)) if trunk_series.size else float("nan")
    knee_pen = max(0.0, (knee_std - 8.0) / 3.0) * 2.0 if np.isfinite(knee_std) else 0.0
    trunk_pen = max(0.0, (trunk_std - 6.0) / 3.0) * 2.0 if np.isfinite(trunk_std) else 0.0
    stability_penalty = float(min(10.0, knee_pen + trunk_pen))

    subscores = {
        "knee_flexion": knee_score,
        "trunk_inclination": trunk_score,
        "impact_height": impact_score,
        "timing": timing_score,
    }

    weights = {
        "knee_flexion": 0.35,
        "trunk_inclination": 0.35,
        "impact_height": 0.15,
        "timing": 0.15,
    }
    valid_keys = [k for k, v in subscores.items() if np.isfinite(v)]
    total = float(sum(weights.get(k, 0.0) for k in valid_keys))
    base_score = 0.0
    if total > 0:
        for k in valid_keys:
            base_score += (weights.get(k, 0.0) / total) * subscores[k]

    bottleneck = float(np.nanmin([subscores[k] for k in valid_keys])) if valid_keys else float("nan")
    alpha = 0.65
    combined = base_score if not np.isfinite(bottleneck) else (alpha * base_score + (1.0 - alpha) * bottleneck)
    all_weight = float(sum(weights.values())) if weights else 1.0
    coverage = total / all_weight if all_weight > 0 else 0.0
    final_score = max(0.0, combined * (0.85 + 0.15 * coverage) - stability_penalty)

    # risk flags (warnings only)
    risk_flags = []
    if np.isfinite(knee_flexion_est):
        if knee_flexion_est < 64.5 - 1.5 * 9.7 or knee_flexion_est > 64.5 + 1.5 * 9.7:
            risk_flags.append("knee_flexion_out_of_range")
    if np.isfinite(trunk_incl_est):
        if trunk_incl_est < 25.0 - 7.1 or trunk_incl_est > 25.0 + 7.1:
            risk_flags.append("trunk_inclination_risk")

    return {
        "subscores": subscores,
        "final_score": float(final_score),
        "coverage": coverage,
        "z_scores": {
            "knee_flexion": knee_z,
            "trunk_inclination": trunk_z,
        },
        "metrics_min": {
            "knee_flexion_est_deg": knee_flexion_est,
            "trunk_inclination_est_deg": trunk_incl_est,
            "impact_bh_ratio": impact_bh_ratio,
            "timing_toss_to_trophy": t1,
            "timing_trophy_to_drop": t2,
            "timing_drop_to_contact": t3,
        },
        "confidence": {
            "knee_confidence": knee_conf,
            "knee_valid_frames": knee_valid,
            "knee_total_frames": knee_total,
            "trunk_confidence": trunk_conf,
            "trunk_valid_frames": trunk_valid,
            "trunk_total_frames": trunk_total,
            "impact_confidence": impact_conf,
            "impact_valid_frames": impact_valid,
            "impact_total_frames": impact_total,
            "impact_source": impact_source,
            "impact_cap_reason": impact_cap_reason,
            "impact_capped": False,
            "impact_status": impact_status,
            "impact_reason": impact_reason,
            "timing_confidence": timing_conf,
            "timing_valid_count": timing_valid,
            "timing_total_count": 3,
            "timing_invalid": timing_invalid,
            "timing_cap_reason": timing_cap_reason,
            "timing_capped": bool(timing_cap_reason),
            "timing_status": timing_status,
            "timing_redetected": timing_redetected,
        },
        "stability": {
            "knee_flexion_std": knee_std,
            "trunk_inclination_std": trunk_std,
        },
        "penalties": {
            "stability_penalty": stability_penalty,
        },
        "risk_flags": risk_flags,
        "raw": {
            "knee_flexion_raw": knee_flexion_raw,
            "knee_flexion_est": knee_flexion_est,
            "trunk_inclination_mean": trunk_mean,
            "trunk_inclination_est": trunk_incl_est,
            "impact_bh_ratio": impact_bh_ratio,
            "impact_source": impact_source,
            "impact_cap_reason": impact_cap_reason,
            "timing_cap_reason": timing_cap_reason,
        },
    }

def compute_metrics(
    signals: Dict[str, np.ndarray],
    phases: Dict[str, Optional[int]],
    ts_ms: np.ndarray,
) -> Dict[str, float]:
    lw_y = signals["toss_wrist_y"]
    rw_y = signals["racket_wrist_y"]
    re_y = signals["racket_elbow_y"]
    ls_y = signals["toss_shoulder_y"]
    rs_y = signals["racket_shoulder_y"]
    nose_y = signals["nose_y"]
    shoulder_tilt_signed = signals["shoulder_tilt"]
    shoulder_tilt_abs = signals["shoulder_tilt_abs"]
    elbow_angle = signals["elbow_angle"]
    knee_angle = signals["knee_angle"]
    l_sh = signals["left_shoulder_y"]
    r_sh = signals["right_shoulder_y"]

    body_scale = _compute_body_scale(signals)
    if not np.isfinite(body_scale) or body_scale <= 1e-6:
        body_scale = 1.0

    def idx(name: str) -> Optional[int]:
        return phases.get(name)

    def t(idx_val: Optional[int]) -> Optional[float]:
        if idx_val is None or idx_val < 0 or idx_val >= len(ts_ms):
            return None
        return float(ts_ms[idx_val]) / 1000.0

    metrics = {
        "toss_peak_height": float("nan"),
        "toss_smoothness": float("nan"),
        "toss_time_to_peak": float("nan"),
        "trophy_elbow_angle": float("nan"),
        "trophy_elbow_height": float("nan"),
        "knee_angle_min": float("nan"),
        "shoulder_tilt_trophy": float("nan"),
        "shoulder_tilt_trophy_signed": float("nan"),
        "racket_drop_depth": float("nan"),
        "racket_drop_depth_raw": float("nan"),
        "contact_height_raw": float("nan"),
        "contact_height": float("nan"),
        "contact_above_head_raw": float("nan"),
        "contact_above_head": float("nan"),
        "body_scale": float(body_scale),
        "coord_system_used": "y_up",
        "timing_toss_to_trophy": float("nan"),
        "timing_trophy_to_drop": float("nan"),
        "timing_drop_to_contact": float("nan"),
    }

    toss_peak = idx("toss_peak")
    toss_start = idx("toss_start")
    trophy = idx("trophy")
    drop = idx("racket_drop")
    contact = idx("contact")

    if toss_peak is not None and toss_peak < len(lw_y):
        metrics["toss_peak_height"] = float((lw_y[toss_peak] - ls_y[toss_peak]) / body_scale)

    if toss_start is not None and toss_peak is not None and toss_peak > toss_start:
        t0 = t(toss_start)
        t1 = t(toss_peak)
        if t0 is not None and t1 is not None:
            metrics["toss_time_to_peak"] = max(0.0, t1 - t0)
        seg = lw_y[toss_start:toss_peak + 1]
        seg_t = ts_ms[toss_start:toss_peak + 1] / 1000.0
        if len(seg) >= 3:
            dt = np.diff(seg_t)
            dy = np.diff(seg)
            vel = np.divide(dy, dt, out=np.full_like(dy, np.nan), where=dt > 1e-6)
            metrics["toss_smoothness"] = float(np.nanstd(vel))

    if trophy is not None and trophy < len(elbow_angle):
        metrics["trophy_elbow_angle"] = float(elbow_angle[trophy])
        metrics["trophy_elbow_height"] = float((re_y[trophy] - rs_y[trophy]) / body_scale)
        metrics["shoulder_tilt_trophy"] = float(shoulder_tilt_abs[trophy] / body_scale)
        metrics["shoulder_tilt_trophy_signed"] = float(shoulder_tilt_signed[trophy] / body_scale)

    if knee_angle.size > 0:
        metrics["knee_angle_min"] = float(np.nanmin(knee_angle))

    if drop is not None and drop < len(rw_y):
        shoulder_ref = float(np.nanmax([l_sh[drop], r_sh[drop]]))
        depth_raw = float(shoulder_ref - rw_y[drop])
        metrics["racket_drop_depth_raw"] = depth_raw
        metrics["racket_drop_depth"] = float(depth_raw / body_scale)

    if contact is not None and contact < len(rw_y):
        shoulder_ref = float(np.nanmax([l_sh[contact], r_sh[contact]]))
        metrics["contact_height_raw"] = float(rw_y[contact] - shoulder_ref)
        metrics["contact_height"] = float(metrics["contact_height_raw"] / body_scale)
        head_ref = nose_y[contact]
        if not np.isfinite(head_ref):
            head_ref = float(np.nanmax([l_sh[contact], r_sh[contact]]))
        metrics["contact_above_head_raw"] = float(rw_y[contact] - head_ref)
        metrics["contact_above_head"] = float(metrics["contact_above_head_raw"] / body_scale)

        # sanity check: ensure sign matches image y-down relationship
        wrist_above = bool(rw_y[contact] > head_ref)
        sign_mismatch = (wrist_above and metrics["contact_above_head"] < 0) or (
            (not wrist_above) and metrics["contact_above_head"] > 0
        )
        if sign_mismatch:
            for k in [
                "toss_peak_height",
                "trophy_elbow_height",
                "racket_drop_depth",
                "racket_drop_depth_raw",
                "contact_height_raw",
                "contact_height",
                "contact_above_head_raw",
                "contact_above_head",
            ]:
                if np.isfinite(metrics.get(k, float("nan"))):
                    metrics[k] = -metrics[k]
            if np.isfinite(metrics.get("shoulder_tilt_trophy_signed", float("nan"))):
                metrics["shoulder_tilt_trophy_signed"] = -metrics["shoulder_tilt_trophy_signed"]
                metrics["shoulder_tilt_trophy"] = abs(metrics["shoulder_tilt_trophy_signed"])
            metrics["coord_system_used"] = "image_y_down_flipped"

    if toss_peak is not None and trophy is not None:
        t_peak = t(toss_peak)
        t_trophy = t(trophy)
        if t_peak is not None and t_trophy is not None:
            metrics["timing_toss_to_trophy"] = t_trophy - t_peak

    if trophy is not None and drop is not None:
        t_trophy = t(trophy)
        t_drop = t(drop)
        if t_trophy is not None and t_drop is not None:
            metrics["timing_trophy_to_drop"] = t_drop - t_trophy

    if drop is not None and contact is not None:
        t_drop = t(drop)
        t_contact = t(contact)
        if t_drop is not None and t_contact is not None:
            metrics["timing_drop_to_contact"] = t_contact - t_drop

    return metrics


def score_metrics(metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], float]:
    def _avg(vals: List[float]) -> float:
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    # if toss smoothness is extreme, treat as missing (often noise)
    if np.isfinite(metrics.get("toss_smoothness", float("nan"))) and metrics["toss_smoothness"] > 0.1:
        metrics["toss_smoothness"] = float("nan")

    # normalized (body-scale) thresholds
    toss_height = _score_trapezoid(metrics["toss_peak_height"], 0.60, 1.40, 0.30, 2.00)
    toss_smooth = _score_inverse(metrics["toss_smoothness"], 0.003, 0.02)
    toss_time = _score_trapezoid(metrics["toss_time_to_peak"], 0.3, 1.4, 0.05, 2.0)
    toss_score = _avg([toss_height, toss_smooth, toss_time])

    elbow_angle = _score_trapezoid(metrics["trophy_elbow_angle"], 70, 170, 50, 180)
    elbow_height = _score_trapezoid(metrics["trophy_elbow_height"], 0.05, 0.45, 0.0, 0.80)
    trophy_score = _avg([elbow_angle, elbow_height])

    knee_score = _score_trapezoid(metrics["knee_angle_min"], 100, 150, 70, 170)

    shoulder_score = _score_trapezoid(metrics["shoulder_tilt_trophy"], 0.03, 0.12, 0.01, 0.30)

    drop_score = _score_trapezoid(metrics["racket_drop_depth"], 0.15, 0.60, 0.05, 1.00)

    contact_height = _score_trapezoid(metrics["contact_height"], 0.20, 0.70, 0.05, 1.20)
    contact_head = _score_trapezoid(metrics["contact_above_head"], 0.10, 0.60, 0.00, 1.00)
    contact_score = _avg([contact_height, contact_head])

    timing_scores = [
        _score_trapezoid(metrics["timing_toss_to_trophy"], 0.3, 1.0, 0.05, 1.8),
        _score_trapezoid(metrics["timing_trophy_to_drop"], 0.03, 0.25, 0.0, 0.6),
        _score_trapezoid(metrics["timing_drop_to_contact"], 0.02, 0.25, 0.0, 0.6),
    ]
    timing_score = _avg(timing_scores)

    subscores = {
        "toss": toss_score,
        "trophy": trophy_score,
        "knee_bend": knee_score,
        "shoulder_tilt": shoulder_score,
        "racket_drop": drop_score,
        "contact": contact_score,
        "timing": timing_score,
    }

    use_weights = weights or DEFAULT_WEIGHTS
    valid_keys = [k for k, v in subscores.items() if np.isfinite(v)]
    total = float(sum(use_weights.get(k, 0.0) for k in valid_keys))
    if total <= 0:
        return subscores, 0.0
    final_score = 0.0
    for key in valid_keys:
        w = float(use_weights.get(key, 0.0)) / total
        final_score += w * subscores[key]

    return subscores, float(final_score)


def _plausibility_score(metrics: Dict[str, float]) -> float:
    score = 0.0
    drop_raw = metrics.get("racket_drop_depth_raw", float("nan"))
    contact_above = metrics.get("contact_above_head", float("nan"))
    if np.isfinite(drop_raw):
        score += 1.0 if drop_raw > 0 else -0.5
    if np.isfinite(contact_above):
        score += 4.0 if contact_above > 0 else -3.0
    timing = [
        metrics.get("timing_toss_to_trophy", float("nan")),
        metrics.get("timing_trophy_to_drop", float("nan")),
        metrics.get("timing_drop_to_contact", float("nan")),
    ]
    valid_timing = [t for t in timing if np.isfinite(t)]
    if len(valid_timing) >= 2 and not all(t <= 1e-6 for t in valid_timing):
        score += 2.0
    elif len(valid_timing) >= 2:
        score -= 2.0
    if np.isfinite(metrics.get("trophy_elbow_angle", float("nan"))):
        score += 1.0
    if np.isfinite(metrics.get("knee_angle_min", float("nan"))):
        score += 1.0
    if np.isfinite(metrics.get("toss_peak_height", float("nan"))) and metrics["toss_peak_height"] > 0:
        score += 1.0
    if np.isfinite(metrics.get("contact_height", float("nan"))) and metrics["contact_height"] > 0:
        score += 0.5
    return score


def _finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    if np.isfinite(metrics.get("racket_drop_depth", float("nan"))) and metrics["racket_drop_depth"] < 0:
        metrics["racket_drop_depth"] = abs(metrics["racket_drop_depth"])
    if np.isfinite(metrics.get("shoulder_tilt_trophy_signed", float("nan"))):
        metrics["shoulder_tilt_trophy"] = abs(metrics["shoulder_tilt_trophy_signed"])
    return metrics


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


def _quality_flags(
    df: pd.DataFrame,
    hand_mode_selected: str,
    v_thr: float = 0.6,
) -> Dict[str, float]:
    if "vis" not in df.columns:
        return {"vis_available": False}
    if "lm" not in df.columns and "lm_id" in df.columns:
        df = df.rename(columns={"lm_id": "lm"})

    if "frame" not in df.columns or "lm" not in df.columns:
        return {"vis_available": False}

    vis = df[["frame", "lm", "vis"]].copy()
    vis = vis.pivot_table(index="frame", columns="lm", values="vis", aggfunc="mean")

    left_w = vis.get(LM["left_wrist"])
    right_w = vis.get(LM["right_wrist"])
    if left_w is None or right_w is None:
        return {"vis_available": False}

    if hand_mode_selected == "A":
        toss_vis = left_w.to_numpy()
        racket_vis = right_w.to_numpy()
    else:
        toss_vis = right_w.to_numpy()
        racket_vis = left_w.to_numpy()

    toss_low = ~np.isfinite(toss_vis) | (toss_vis < v_thr)
    racket_low = ~np.isfinite(racket_vis) | (racket_vis < v_thr)

    total = max(len(toss_vis), 1)
    flags = {
        "vis_available": True,
        "v_thr": float(v_thr),
        "num_frames": int(total),
        "toss_wrist_nan_ratio": float(np.mean(toss_low)),
        "hand_wrist_nan_ratio": float(np.mean(racket_low)),
        "num_nan_frames": int(np.sum(toss_low | racket_low)),
        "occlusion_gap_frames": _max_gap(racket_low),
    }
    return flags


def _evaluate_mode(
    wide: Dict[str, np.ndarray],
    toss_side: str,
    racket_side: str,
    weights: Optional[Dict[str, float]],
    median_k: int,
    gap_max: int,
    v_thr: float,
    quantile_calib: bool,
    fps: float,
) -> Dict[str, object]:
    signals = compute_signals(
        wide,
        median_k=median_k,
        toss_side=toss_side,
        racket_side=racket_side,
        gap_max=gap_max,
        v_thr=v_thr,
    )
    phases = detect_phases(signals, wide["ts_ms"])
    phases = fix_hrp_lrp(phases, signals, wide["ts_ms"], fps)
    metrics = compute_metrics(signals, phases, wide["ts_ms"])
    legacy_subscores, legacy_final = score_metrics(metrics, weights=weights)
    biomech = compute_biomech_scores(signals, metrics, phases, quantile_calib=quantile_calib, fps=fps)
    return {
        "metrics": metrics,
        "subscores": biomech["subscores"],
        "final_score": biomech["final_score"],
        "legacy_subscores": legacy_subscores,
        "legacy_final_score": legacy_final,
        "biomech": biomech,
        "phases": phases,
        "signals": signals,
    }


def analyze_serve_dual(
    df: pd.DataFrame,
    use_coords: str = "smooth",
    weights: Optional[Dict[str, float]] = None,
    median_k: int = 5,
    hand_mode: str = "auto",
    gap_max: int = 3,
    v_thr: float = 0.6,
    quantile_calib: bool = False,
) -> Dict[str, object]:
    wide = long_to_wide(df, use_coords=use_coords)
    ts_ms = wide.get("ts_ms")
    fps = 60.0
    if ts_ms is not None and len(ts_ms) > 1:
        dt = np.diff(ts_ms)
        dt = dt[np.isfinite(dt) & (dt > 1e-3)]
        if dt.size:
            fps = float(1000.0 / np.median(dt))
    y_up = 1.0 - wide["y"]
    left_wrist_y = _median_filter_1d(_fill_nan(y_up[:, LM["left_wrist"]], gap_max=gap_max), k=5)
    right_wrist_y = _median_filter_1d(_fill_nan(y_up[:, LM["right_wrist"]], gap_max=gap_max), k=5)
    toss_guess, toss_scores = _infer_toss_side(left_wrist_y, right_wrist_y)

    res_a = _evaluate_mode(
        wide,
        toss_side="left",
        racket_side="right",
        weights=weights,
        median_k=median_k,
        gap_max=gap_max,
        v_thr=v_thr,
        quantile_calib=quantile_calib,
        fps=fps,
    )
    res_b = _evaluate_mode(
        wide,
        toss_side="right",
        racket_side="left",
        weights=weights,
        median_k=median_k,
        gap_max=gap_max,
        v_thr=v_thr,
        quantile_calib=quantile_calib,
        fps=fps,
    )

    plaus_a = _plausibility_score(res_a["metrics"])
    plaus_b = _plausibility_score(res_b["metrics"])
    if toss_guess == "left":
        plaus_a += 0.5
    else:
        plaus_b += 0.5

    hand_mode = (hand_mode or "auto").upper()
    if hand_mode == "A":
        selected_mode = "A"
        selected = res_a
    elif hand_mode == "B":
        selected_mode = "B"
        selected = res_b
    else:
        if plaus_b > plaus_a:
            selected_mode = "B"
            selected = res_b
        elif plaus_a > plaus_b:
            selected_mode = "A"
            selected = res_a
        else:
            selected_mode = "A" if res_a["final_score"] >= res_b["final_score"] else "B"
            selected = res_a if selected_mode == "A" else res_b

    selected["metrics"] = _finalize_metrics(selected["metrics"])
    selected["metrics_full"] = dict(selected["metrics"])
    selected["legacy_subscores"], selected["legacy_final_score"] = score_metrics(selected["metrics"], weights=weights)
    selected["biomech"] = compute_biomech_scores(
        selected["signals"],
        selected["metrics"],
        selected["phases"],
        quantile_calib=quantile_calib,
        fps=fps,
    )
    selected["metrics_min"] = selected["biomech"].get("metrics_min", {})
    selected["subscores"] = selected["biomech"]["subscores"]
    selected["final_score"] = selected["biomech"]["final_score"]

    mirror_suspect = selected_mode == "B"

    return {
        "A": res_a,
        "B": res_b,
        "selected": selected,
        "hand_mode_selected": selected_mode,
        "mirror_suspect": mirror_suspect,
        "hand_mode_guess": toss_guess,
        "hand_mode_guess_scores": toss_scores,
        "ts_ms": wide["ts_ms"],
        "frames": wide["frames"],
    }


def analyze_serve(
    df: pd.DataFrame,
    use_coords: str = "smooth",
    weights: Optional[Dict[str, float]] = None,
    median_k: int = 5,
    hand_mode: str = "auto",
    v_thr: float = 0.6,
    gap_max: int = 3,
    quantile_calib: bool = False,
) -> Dict[str, object]:
    dual = analyze_serve_dual(
        df,
        use_coords=use_coords,
        weights=weights,
        median_k=median_k,
        hand_mode=hand_mode,
        gap_max=gap_max,
        v_thr=v_thr,
        quantile_calib=quantile_calib,
    )
    selected = dual["selected"]
    metrics_full = selected.get("metrics_full", selected.get("metrics", {}))
    quality_flags = _quality_flags(df, dual["hand_mode_selected"], v_thr=v_thr)

    timing_bad = False
    for k in ["timing_trophy_to_drop", "timing_drop_to_contact"]:
        v = metrics_full.get(k, float("nan"))
        if not np.isfinite(v) or v <= 1e-6:
            timing_bad = True
            break

    if timing_bad and use_coords != "raw":
        fallback = analyze_serve_dual(
            df,
            use_coords="raw",
            weights=weights,
            median_k=median_k,
            hand_mode=dual["hand_mode_selected"],
            gap_max=gap_max,
            quantile_calib=quantile_calib,
        )
        fallback_sel = fallback["selected"]
        ok = True
        for k in ["timing_trophy_to_drop", "timing_drop_to_contact"]:
            v = fallback_sel["metrics"].get(k, float("nan"))
            if not np.isfinite(v) or v <= 1e-6:
                ok = False
                break
        if ok:
            metrics_full.update({
                "timing_toss_to_trophy": fallback_sel["metrics"].get("timing_toss_to_trophy"),
                "timing_trophy_to_drop": fallback_sel["metrics"].get("timing_trophy_to_drop"),
                "timing_drop_to_contact": fallback_sel["metrics"].get("timing_drop_to_contact"),
            })
            selected["phases"] = fallback_sel["phases"]
            selected["subscores"], selected["final_score"] = score_metrics(metrics_full, weights=weights)
            quality_flags["timing_fallback"] = "raw"
        else:
            quality_flags["timing_fallback"] = "failed"

    metrics_out = selected.get("metrics_min") or {}
    if not metrics_out:
        metrics_out = metrics_full

    missing_keys = []
    for k, v in metrics_out.items():
        if k in {"coord_system_used"}:
            continue
        if not np.isfinite(v):
            missing_keys.append(k)
    score_debug = {
        "weights": weights or DEFAULT_WEIGHTS,
        "subscores": selected["subscores"],
        "metrics": metrics_out,
        "debug_metrics": metrics_full,
        "missing_metrics_count": int(len(missing_keys)),
        "missing_metrics": missing_keys,
        "penalties": {
            "missing_metrics_count": int(len(missing_keys)),
            "stability_penalty": selected.get("biomech", {}).get("penalties", {}).get("stability_penalty"),
        },
        "biomech": selected.get("biomech"),
        "legacy_subscores": selected.get("legacy_subscores"),
        "legacy_final_score": selected.get("legacy_final_score"),
    }
    return {
        "metrics": metrics_out,
        "subscores": selected["subscores"],
        "final_score": selected["final_score"],
        "biomech": selected.get("biomech"),
        "legacy_subscores": selected.get("legacy_subscores"),
        "legacy_final_score": selected.get("legacy_final_score"),
        "phases": selected["phases"],
        "signals": selected["signals"],
        "ts_ms": dual["ts_ms"],
        "frames": dual["frames"],
        "hand_mode_selected": dual["hand_mode_selected"],
        "mirror_suspect": dual["mirror_suspect"],
        "hand_mode_guess": dual["hand_mode_guess"],
        "hand_mode_guess_scores": dual["hand_mode_guess_scores"],
        "quality_flags": quality_flags,
        "score_debug": score_debug,
    }
