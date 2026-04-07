import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
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
    knee_angle = signals.get("knee_angle")
    trunk_angle = signals.get("trunk_angle")
    elbow_angle = signals.get("elbow_angle")

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

    def _norm_segment(values: np.ndarray, inverse: bool = False) -> np.ndarray:
        out = np.zeros_like(values, dtype=float)
        mask = np.isfinite(values)
        if not mask.any():
            return out
        lo = float(np.nanmin(values[mask]))
        hi = float(np.nanmax(values[mask]))
        if hi - lo > 1e-8:
            out[mask] = (values[mask] - lo) / (hi - lo)
        if inverse:
            out[mask] = 1.0 - out[mask]
        return out

    def _positive_grad_score(series: Optional[np.ndarray], idxs: np.ndarray) -> np.ndarray:
        if series is None or len(series) == 0:
            return np.zeros_like(idxs, dtype=float)
        grad = np.gradient(series.astype(np.float64))
        grad = np.where(np.isfinite(grad), np.maximum(grad, 0.0), np.nan)
        return _norm_segment(grad[idxs])

    def _direct_progress_score(series: Optional[np.ndarray], idxs: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        if series is None or len(series) == 0:
            return np.zeros_like(idxs, dtype=float)
        start_val = float(series[start_idx]) if np.isfinite(series[start_idx]) else float("nan")
        end_val = float(series[end_idx]) if np.isfinite(series[end_idx]) else float("nan")
        if not np.isfinite(start_val) or not np.isfinite(end_val):
            return np.zeros_like(idxs, dtype=float)
        sign = 1.0 if end_val >= start_val else -1.0
        progress = sign * (series[idxs] - start_val)
        return _norm_segment(progress)

    def _find_lrp_composite(s: int, e: int) -> Optional[int]:
        idxs = np.arange(s, e + 1, dtype=int)
        if idxs.size == 0:
            return None

        wrist_low = (
            _norm_segment(wrist_y[idxs], inverse=not y_is_down)
            if wrist_y is not None
            else np.zeros_like(idxs, dtype=float)
        )
        elbow_low = (
            _norm_segment(elbow_y[idxs], inverse=not y_is_down)
            if elbow_y is not None
            else np.zeros_like(idxs, dtype=float)
        )
        knee_extend = _positive_grad_score(knee_angle, idxs)
        elbow_open = _positive_grad_score(elbow_angle, idxs)
        trunk_drive = _direct_progress_score(trunk_angle, idxs, new_trophy, contact_idx)

        score = (
            0.30 * wrist_low
            + 0.10 * elbow_low
            + 0.20 * knee_extend
            + 0.15 * trunk_drive
            + 0.25 * elbow_open
        )
        valid = np.isfinite(score)
        if not valid.any():
            return None

        max_score = float(np.nanmax(score[valid]))
        keep = valid & (score >= max_score * 0.98)
        if not keep.any():
            return int(idxs[int(np.nanargmax(score))])
        return int(idxs[np.flatnonzero(keep)[0]])

    if start < end:
        new_drop = _find_lrp_composite(start, end)
        if new_drop is not None:
            drop_status = "composite_refind_after_forced_trophy"
        else:
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
                retry = _find_lrp_composite(s2, e2)
                if retry is not None:
                    drop_status = "composite_retry"
                else:
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


# -----------------------------
# HybrIK / 3D verification utils
# -----------------------------
HYBRIK_SKELETON_EDGES = [
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

HYBRIK_MP_JOINT_MAP = {
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

HYBRIK_SYMMETRY_SEGMENTS = {
    "upper_arm": ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow"),
    "forearm": ("left_elbow", "left_wrist", "right_elbow", "right_wrist"),
    "thigh": ("left_hip", "left_knee", "right_hip", "right_knee"),
    "shank": ("left_knee", "left_ankle", "right_knee", "right_ankle"),
}

HYBRIK_3D_ANGLE_KEYS = [
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "trunk_inclination_deg",
    "shoulder_tilt_deg",
    "right_elbow_angle_deg",
]


def compute_3d_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-12:
        return float("nan")
    cosine_angle = np.dot(ba, bc) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def compute_vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom <= 1e-12:
        return float("nan")
    cosine_angle = np.dot(v1, v2) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def compute_line_tilt_to_horizontal(v: np.ndarray, vertical_axis: int = 1) -> float:
    if np.linalg.norm(v) <= 1e-12:
        return float("nan")
    horizontal = np.delete(v, vertical_axis)
    horizontal_norm = np.linalg.norm(horizontal)
    vertical_mag = abs(v[vertical_axis])
    return float(np.degrees(np.arctan2(vertical_mag, horizontal_norm)))


def _load_json_object(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_hybrik_joint_items(node: object) -> List[Dict[str, object]]:
    if isinstance(node, list):
        if node and isinstance(node[0], dict) and "xyz" in node[0]:
            return [item for item in node if isinstance(item, dict) and "xyz" in item]
        for item in node:
            extracted = _extract_hybrik_joint_items(item)
            if extracted:
                return extracted
        return []

    if isinstance(node, dict):
        for key in ["joints_3d_29", "joints", "keypoints"]:
            extracted = _extract_hybrik_joint_items(node.get(key))
            if extracted:
                return extracted
        for value in node.values():
            extracted = _extract_hybrik_joint_items(value)
            if extracted:
                return extracted

    return []


def _normalize_joint_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _build_hybrik_joint_map(items: Iterable[Dict[str, object]]) -> Dict[str, np.ndarray]:
    joints: Dict[str, np.ndarray] = {}
    for item in items:
        name = item.get("name")
        xyz = item.get("xyz")
        if not isinstance(name, str) or not isinstance(xyz, list) or len(xyz) != 3:
            continue
        try:
            joints[_normalize_joint_name(name)] = np.array(xyz, dtype=float)
        except (TypeError, ValueError):
            continue
    return joints


def _find_hybrik_joint(joints: Dict[str, np.ndarray], aliases: Iterable[str]) -> Optional[np.ndarray]:
    normalized_aliases = [_normalize_joint_name(alias) for alias in aliases]
    for alias in normalized_aliases:
        if alias in joints:
            return joints[alias]
    for alias in normalized_aliases:
        for name, value in joints.items():
            if alias in name:
                return value
    return None


def _infer_hybrik_y_axis_down(joints: Dict[str, np.ndarray]) -> bool:
    shoulders = [
        _find_hybrik_joint(joints, ["left_shoulder"]),
        _find_hybrik_joint(joints, ["right_shoulder"]),
    ]
    hips = [
        _find_hybrik_joint(joints, ["left_hip"]),
        _find_hybrik_joint(joints, ["right_hip"]),
    ]
    shoulders = [v for v in shoulders if v is not None]
    hips = [v for v in hips if v is not None]
    if not shoulders or not hips:
        return True
    mean_shoulder_y = float(np.mean([v[1] for v in shoulders]))
    mean_hip_y = float(np.mean([v[1] for v in hips]))
    return mean_hip_y > mean_shoulder_y


def load_hybrik_json(data_or_path: Union[str, Dict[str, object]]) -> Dict[str, object]:
    if isinstance(data_or_path, str):
        data = _load_json_object(data_or_path)
    else:
        data = data_or_path
    if not isinstance(data, dict):
        raise RuntimeError("Expected HybrIK JSON to be a dictionary object.")
    return data


def extract_hybrik_joint_items(data_or_path: Union[str, Dict[str, object]]) -> List[Dict[str, object]]:
    data = load_hybrik_json(data_or_path)
    return _extract_hybrik_joint_items(data)


def load_hybrik_joint_xyz_map(data_or_path: Union[str, Dict[str, object]]) -> Dict[str, np.ndarray]:
    return _build_hybrik_joint_map(extract_hybrik_joint_items(data_or_path))


def load_hybrik_joint_score_map(data_or_path: Union[str, Dict[str, object]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for item in extract_hybrik_joint_items(data_or_path):
        name = item.get("name")
        score = item.get("score")
        if isinstance(name, str) and isinstance(score, (int, float)):
            scores[_normalize_joint_name(name)] = float(score)
    return scores


def analyze_trophy_pose_3d(data_or_path: Union[str, Dict[str, object]]) -> Tuple[Dict[str, float], Dict[str, object]]:
    joints = load_hybrik_joint_xyz_map(data_or_path)
    if not joints:
        raise RuntimeError("No valid 3D joint entries with `name` and `xyz` were found in the HybrIK JSON.")

    l_hip = _find_hybrik_joint(joints, ["left_hip", "l_hip"])
    r_hip = _find_hybrik_joint(joints, ["right_hip", "r_hip"])
    l_knee = _find_hybrik_joint(joints, ["left_knee", "l_knee"])
    r_knee = _find_hybrik_joint(joints, ["right_knee", "r_knee"])
    l_ankle = _find_hybrik_joint(joints, ["left_ankle", "l_ankle"])
    r_ankle = _find_hybrik_joint(joints, ["right_ankle", "r_ankle"])
    l_shoulder = _find_hybrik_joint(joints, ["left_shoulder", "l_shoulder"])
    r_shoulder = _find_hybrik_joint(joints, ["right_shoulder", "r_shoulder"])
    l_elbow = _find_hybrik_joint(joints, ["left_elbow", "l_elbow"])
    r_elbow = _find_hybrik_joint(joints, ["right_elbow", "r_elbow"])
    l_wrist = _find_hybrik_joint(joints, ["left_wrist", "l_wrist"])
    r_wrist = _find_hybrik_joint(joints, ["right_wrist", "r_wrist"])
    neck = _find_hybrik_joint(joints, ["neck", "spine3", "spine_3"])

    if neck is None and l_shoulder is not None and r_shoulder is not None:
        neck = (l_shoulder + r_shoulder) / 2.0

    y_axis_down = _infer_hybrik_y_axis_down(joints)
    vertical_up = np.array([0.0, -1.0 if y_axis_down else 1.0, 0.0], dtype=float)

    metrics: Dict[str, float] = {}
    metrics["left_knee_angle_deg"] = (
        compute_3d_angle(l_hip, l_knee, l_ankle)
        if l_hip is not None and l_knee is not None and l_ankle is not None
        else float("nan")
    )
    metrics["right_knee_angle_deg"] = (
        compute_3d_angle(r_hip, r_knee, r_ankle)
        if r_hip is not None and r_knee is not None and r_ankle is not None
        else float("nan")
    )

    pelvis_mid = None
    spine_vector = None
    if l_hip is not None and r_hip is not None and neck is not None:
        pelvis_mid = (l_hip + r_hip) / 2.0
        spine_vector = neck - pelvis_mid
        metrics["trunk_inclination_deg"] = compute_vector_angle(spine_vector, vertical_up)
    else:
        metrics["trunk_inclination_deg"] = float("nan")

    shoulder_vector = None
    if l_shoulder is not None and r_shoulder is not None:
        shoulder_vector = r_shoulder - l_shoulder
        metrics["shoulder_tilt_deg"] = compute_line_tilt_to_horizontal(shoulder_vector, vertical_axis=1)
    else:
        metrics["shoulder_tilt_deg"] = float("nan")

    metrics["left_elbow_angle_deg"] = (
        compute_3d_angle(l_shoulder, l_elbow, l_wrist)
        if l_shoulder is not None and l_elbow is not None and l_wrist is not None
        else float("nan")
    )
    metrics["right_elbow_angle_deg"] = (
        compute_3d_angle(r_shoulder, r_elbow, r_wrist)
        if r_shoulder is not None and r_elbow is not None and r_wrist is not None
        else float("nan")
    )

    source_json = os.path.abspath(data_or_path) if isinstance(data_or_path, str) else None
    debug = {
        "num_joints_loaded": len(joints),
        "y_axis_down": y_axis_down,
        "vertical_up_vector": vertical_up.tolist(),
        "pelvis_mid": None if pelvis_mid is None else pelvis_mid.tolist(),
        "spine_vector": None if spine_vector is None else spine_vector.tolist(),
        "shoulder_vector": None if shoulder_vector is None else shoulder_vector.tolist(),
        "source_json": source_json,
    }
    return metrics, debug


def _require_joint(joints: Dict[str, np.ndarray], aliases: Iterable[str]) -> Optional[np.ndarray]:
    return _find_hybrik_joint(joints, aliases)


def compute_hybrik_body_height_proxy(data_or_path: Union[str, Dict[str, object]]) -> float:
    joints = load_hybrik_joint_xyz_map(data_or_path)
    l_hip = _require_joint(joints, ["left_hip"])
    r_hip = _require_joint(joints, ["right_hip"])
    l_knee = _require_joint(joints, ["left_knee"])
    r_knee = _require_joint(joints, ["right_knee"])
    l_ankle = _require_joint(joints, ["left_ankle"])
    r_ankle = _require_joint(joints, ["right_ankle"])
    neck = _require_joint(joints, ["neck", "spine3", "spine_3"])
    head = _require_joint(joints, ["head", "jaw"])

    if any(v is None for v in [l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle, neck]):
        return float("nan")

    left_leg = np.linalg.norm(l_hip - l_knee) + np.linalg.norm(l_knee - l_ankle)
    right_leg = np.linalg.norm(r_hip - r_knee) + np.linalg.norm(r_knee - r_ankle)
    pelvis_mid = (l_hip + r_hip) / 2.0
    trunk = np.linalg.norm(neck - pelvis_mid)
    head_len = np.linalg.norm(head - neck) if head is not None else 0.0
    return float((left_leg + right_leg) / 2.0 + trunk + head_len)


def _hybrik_vertical_up_vector(joints: Dict[str, np.ndarray]) -> np.ndarray:
    y_axis_down = _infer_hybrik_y_axis_down(joints)
    return np.array([0.0, -1.0 if y_axis_down else 1.0, 0.0], dtype=float)


def compute_hybrik_contact_height_norm(
    data_or_path: Union[str, Dict[str, object]],
    side: str = "right",
) -> float:
    joints = load_hybrik_joint_xyz_map(data_or_path)
    wrist = _require_joint(joints, [f"{side}_wrist"])
    l_ankle = _require_joint(joints, ["left_ankle"])
    r_ankle = _require_joint(joints, ["right_ankle"])
    if wrist is None or l_ankle is None or r_ankle is None:
        return float("nan")

    body_height = compute_hybrik_body_height_proxy(data_or_path)
    if not np.isfinite(body_height) or body_height <= 1e-8:
        return float("nan")

    ankle_mid = (l_ankle + r_ankle) / 2.0
    vertical_up = _hybrik_vertical_up_vector(joints)
    return float(np.dot(wrist - ankle_mid, vertical_up) / body_height)


def compute_hybrik_shoulder_raise_deg(
    data_or_path: Union[str, Dict[str, object]],
    side: str = "right",
) -> float:
    joints = load_hybrik_joint_xyz_map(data_or_path)
    shoulder = _require_joint(joints, [f"{side}_shoulder"])
    elbow = _require_joint(joints, [f"{side}_elbow"])
    l_hip = _require_joint(joints, ["left_hip"])
    r_hip = _require_joint(joints, ["right_hip"])
    if shoulder is None or elbow is None or l_hip is None or r_hip is None:
        return float("nan")
    pelvis_mid = (l_hip + r_hip) / 2.0
    return compute_vector_angle(elbow - shoulder, pelvis_mid - shoulder)


def compute_hybrik_elbow_flexion_deg(
    data_or_path: Union[str, Dict[str, object]],
    side: str = "right",
) -> float:
    joints = load_hybrik_joint_xyz_map(data_or_path)
    shoulder = _require_joint(joints, [f"{side}_shoulder"])
    elbow = _require_joint(joints, [f"{side}_elbow"])
    wrist = _require_joint(joints, [f"{side}_wrist"])
    if shoulder is None or elbow is None or wrist is None:
        return float("nan")
    elbow_angle = compute_3d_angle(shoulder, elbow, wrist)
    return float(180.0 - elbow_angle) if np.isfinite(elbow_angle) else float("nan")


def compute_hybrik_shoulder_external_rotation_proxy_deg(
    data_or_path: Union[str, Dict[str, object]],
    side: str = "right",
) -> float:
    joints = load_hybrik_joint_xyz_map(data_or_path)
    shoulder = _require_joint(joints, [f"{side}_shoulder"])
    elbow = _require_joint(joints, [f"{side}_elbow"])
    wrist = _require_joint(joints, [f"{side}_wrist"])
    neck = _require_joint(joints, ["neck", "spine3", "spine_3"])
    l_hip = _require_joint(joints, ["left_hip"])
    r_hip = _require_joint(joints, ["right_hip"])
    if any(v is None for v in [shoulder, elbow, wrist, neck, l_hip, r_hip]):
        return float("nan")

    pelvis_mid = (l_hip + r_hip) / 2.0
    torso = neck - pelvis_mid
    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    plane_normal = np.cross(torso, upper_arm)
    plane_norm = np.linalg.norm(plane_normal)
    forearm_norm = np.linalg.norm(forearm)
    if plane_norm <= 1e-8 or forearm_norm <= 1e-8:
        return float("nan")
    sine_value = abs(float(np.dot(forearm / forearm_norm, plane_normal / plane_norm)))
    sine_value = float(np.clip(sine_value, 0.0, 1.0))
    return float(np.degrees(np.arcsin(sine_value)))


def summarize_keyframe_pose_3d(
    data_or_path: Union[str, Dict[str, object]],
    phase: str,
    dominant_hand: str = "right",
) -> Dict[str, float]:
    angles, _debug = analyze_trophy_pose_3d(data_or_path)
    metrics: Dict[str, float] = dict(angles)

    dominant_hand = dominant_hand.lower()
    lead_side = "left" if dominant_hand == "right" else "right"
    metrics["lead_knee_flexion_deg"] = float(
        180.0 - metrics.get(f"{lead_side}_knee_angle_deg", float("nan"))
    ) if np.isfinite(metrics.get(f"{lead_side}_knee_angle_deg", float("nan"))) else float("nan")
    metrics["body_height_proxy"] = compute_hybrik_body_height_proxy(data_or_path)

    if phase == "contact":
        metrics["impact_height_norm"] = compute_hybrik_contact_height_norm(data_or_path, side=dominant_hand)
        metrics["shoulder_raise_deg"] = compute_hybrik_shoulder_raise_deg(data_or_path, side=dominant_hand)
        metrics["elbow_flexion_deg"] = compute_hybrik_elbow_flexion_deg(data_or_path, side=dominant_hand)
    if phase == "racket_drop":
        metrics["shoulder_external_rotation_proxy_deg"] = compute_hybrik_shoulder_external_rotation_proxy_deg(
            data_or_path, side=dominant_hand
        )

    return metrics


def project_hybrik_joints_2d(data_or_path: Union[str, Dict[str, object]]) -> Dict[str, np.ndarray]:
    data = load_hybrik_json(data_or_path)
    bbox = data.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RuntimeError("Expected `bbox_xyxy` in HybrIK JSON.")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1

    points: Dict[str, np.ndarray] = {}
    for item in extract_hybrik_joint_items(data):
        name = item.get("name")
        uvd = item.get("uvd")
        if not isinstance(name, str) or not isinstance(uvd, list) or len(uvd) != 3:
            continue
        x = float(uvd[0]) * bw + cx
        y = float(uvd[1]) * bw + cy
        points[_normalize_joint_name(name)] = np.array([x, y], dtype=float)
    return points


def load_obj_mesh(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                raw = []
                for token in line.strip().split()[1:]:
                    idx_str = token.split("/")[0]
                    if idx_str:
                        raw.append(int(idx_str) - 1)
                if len(raw) == 3:
                    faces.append(raw)
                elif len(raw) > 3:
                    for start in range(1, len(raw) - 1):
                        faces.append([raw[0], raw[start], raw[start + 1]])
    if not vertices or not faces:
        raise RuntimeError(f"Failed to parse a valid mesh from OBJ: {obj_path}")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)


def fit_hybrik_projection(data_or_path: Union[str, Dict[str, object]]) -> Dict[str, float]:
    data = load_hybrik_json(data_or_path)
    translation = data.get("translation")
    if not isinstance(translation, list) or len(translation) != 3:
        raise RuntimeError("Expected `translation` in HybrIK JSON.")

    bbox = data.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RuntimeError("Expected `bbox_xyxy` in HybrIK JSON.")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx0 = (x1 + x2) / 2.0
    cy0 = (y1 + y2) / 2.0
    bw = x2 - x1

    rows_3d: List[np.ndarray] = []
    rows_2d: List[np.ndarray] = []
    transl = np.asarray(translation, dtype=float)

    for item in extract_hybrik_joint_items(data):
        xyz = item.get("xyz")
        uvd = item.get("uvd")
        if not isinstance(xyz, list) or len(xyz) != 3:
            continue
        if not isinstance(uvd, list) or len(uvd) != 3:
            continue
        joint_3d = np.asarray(xyz, dtype=float) + transl
        joint_2d = np.array(
            [float(uvd[0]) * bw + cx0, float(uvd[1]) * bw + cy0],
            dtype=float,
        )
        if not np.isfinite(joint_3d).all() or not np.isfinite(joint_2d).all():
            continue
        if abs(float(joint_3d[2])) <= 1e-8:
            continue
        rows_3d.append(joint_3d)
        rows_2d.append(joint_2d)

    if len(rows_3d) < 4:
        raise RuntimeError("Not enough valid HybrIK joints to fit a camera projection.")

    points_3d = np.stack(rows_3d, axis=0)
    points_2d = np.stack(rows_2d, axis=0)
    x_norm = points_3d[:, 0] / points_3d[:, 2]
    y_norm = points_3d[:, 1] / points_3d[:, 2]

    design = np.zeros((points_3d.shape[0] * 2, 3), dtype=float)
    design[0::2, 0] = x_norm
    design[0::2, 1] = 1.0
    design[1::2, 0] = y_norm
    design[1::2, 2] = 1.0
    targets = np.empty(points_3d.shape[0] * 2, dtype=float)
    targets[0::2] = points_2d[:, 0]
    targets[1::2] = points_2d[:, 1]

    solution, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    focal_px, cx_px, cy_px = [float(v) for v in solution]
    fitted_2d = np.column_stack([focal_px * x_norm + cx_px, focal_px * y_norm + cy_px])
    errors = np.linalg.norm(fitted_2d - points_2d, axis=1)

    return {
        "focal_px": focal_px,
        "cx_px": cx_px,
        "cy_px": cy_px,
        "num_joints": int(points_3d.shape[0]),
        "fit_mean_error_px": float(np.mean(errors)),
        "fit_max_error_px": float(np.max(errors)),
    }


def fit_similarity_2d(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the optimal 2D similarity transform (uniform scale + rotation + translation)
    that maps *src_pts* → *dst_pts* using the Umeyama algorithm.

    Returns
    -------
    scale : float
        Uniform scale factor.
    R : (2, 2) ndarray
        Rotation matrix.
    t : (2,) ndarray
        Translation vector.
    The transform is applied as: dst ≈ scale * R @ src + t
    """
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape:
        raise RuntimeError("fit_similarity_2d requires src and dst of shape (N, 2).")
    if src.shape[0] < 2:
        raise RuntimeError("fit_similarity_2d requires at least 2 point pairs.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = float(np.mean(np.sum(src_c ** 2, axis=1)))
    if var_src < 1e-12:
        # degenerate source — fall back to pure translation
        return 1.0, np.eye(2, dtype=float), mu_dst - mu_src

    M = (dst_c.T @ src_c) / src.shape[0]
    U, S, Vt = np.linalg.svd(M)

    # correct for reflection so det(R) = +1
    det_sign = 1.0 if np.linalg.det(U @ Vt) > 0 else -1.0
    sign_correction = np.diag([1.0, det_sign])
    R = U @ sign_correction @ Vt
    scale = float(np.sum(S * np.diag(sign_correction)) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def apply_similarity_2d(
    pts: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply a 2D similarity transform to an (N, 2) array of points."""
    pts = np.asarray(pts, dtype=float)
    return scale * (pts @ R.T) + t[None, :]


def build_pose_alignment_anchors(
    data: Dict[str, object],
    focal_px: float,
    cx_px: float,
    cy_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build matched (src, dst) 2D point arrays for pose-driven alignment.

    *src_pts* are joint positions projected with the fitted pinhole camera.
    *dst_pts* are the UVD-based 2D joint positions from HybrIK output.

    These pairs are used to fit a 2D similarity transform that corrects any
    residual camera-fitting error and maps the mesh exactly onto the predicted
    joint locations in the image.
    """
    translation = data.get("translation")
    if not isinstance(translation, list) or len(translation) != 3:
        raise RuntimeError("Expected `translation` in HybrIK JSON.")
    transl = np.asarray(translation, dtype=float)

    bbox = data.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RuntimeError("Expected `bbox_xyxy` in HybrIK JSON.")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bbox_cx = (x1 + x2) / 2.0
    bw = x2 - x1

    src_list: List[np.ndarray] = []
    dst_list: List[np.ndarray] = []

    for item in extract_hybrik_joint_items(data):
        xyz = item.get("xyz")
        uvd = item.get("uvd")
        if not isinstance(xyz, list) or len(xyz) != 3:
            continue
        if not isinstance(uvd, list) or len(uvd) != 3:
            continue
        # camera-space 3D position
        xyz_cam = np.asarray(xyz, dtype=float) + transl
        if not np.isfinite(xyz_cam).all() or xyz_cam[2] <= 1e-8:
            continue
        # project using fitted camera → src
        sx = focal_px * (xyz_cam[0] / xyz_cam[2]) + cx_px
        sy = focal_px * (xyz_cam[1] / xyz_cam[2]) + cy_px
        # UVD → image pixel → dst
        dx = float(uvd[0]) * bw + bbox_cx
        dy = float(uvd[1]) * bw + (y1 + y2) / 2.0
        if not all(map(np.isfinite, [sx, sy, dx, dy])):
            continue
        src_list.append(np.array([sx, sy], dtype=float))
        dst_list.append(np.array([dx, dy], dtype=float))

    if len(src_list) < 4:
        raise RuntimeError(
            f"Not enough valid anchor joints for pose alignment (got {len(src_list)}, need ≥4)."
        )
    return np.stack(src_list, axis=0), np.stack(dst_list, axis=0)


def project_points_pinhole(
    points_3d: np.ndarray,
    focal_px: float,
    cx_px: float,
    cy_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_3d, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError("Expected points_3d to have shape (N, 3).")

    z = pts[:, 2]
    projected = np.full((pts.shape[0], 2), np.nan, dtype=float)
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-8)
    if np.any(valid):
        projected[valid, 0] = focal_px * (pts[valid, 0] / z[valid]) + cx_px
        projected[valid, 1] = focal_px * (pts[valid, 1] / z[valid]) + cy_px
    return projected, valid


def render_projected_mesh_overlay(
    image_bgr: np.ndarray,
    vertices_3d: np.ndarray,
    faces: np.ndarray,
    focal_px: float,
    cx_px: float,
    cy_px: float,
    alpha: float = 0.90,
    base_color_bgr: Tuple[int, int, int] = (222, 206, 184),
    outline_color_bgr: Tuple[int, int, int] = (154, 136, 113),
    target_bbox_xyxy: Optional[Iterable[float]] = None,
    target_fill_ratio: float = 0.98,
    alignment_anchors: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Render the SMPL mesh as a semi-transparent overlay on *image_bgr*.

    Alignment strategy (applied in priority order):
    1. **Pose-driven** (preferred): when *alignment_anchors* = (src_pts, dst_pts)
       is provided, fit a 2D similarity transform (scale + rotation + translation)
       from the camera-projected joint positions (src) to the UVD joint positions
       (dst) and apply it to all vertex projections.  This achieves a 1:1
       alignment with the person in the photo.
    2. **Bbox-based fallback**: when *target_bbox_xyxy* is provided but no
       anchors, rescale the mesh bounding box to fit inside the detector bbox.
    3. **Raw projection**: if neither is provided, use the raw pinhole projection
       without any 2D post-correction.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError("Expected a valid BGR image for mesh overlay rendering.")

    image_h, image_w = image_bgr.shape[:2]
    verts_2d, valid_vertices = project_points_pinhole(vertices_3d, focal_px, cx_px, cy_px)
    transform_debug: Dict[str, object] = {
        "alignment_mode": "none",
        "target_bbox_xyxy": None,
        "target_fill_ratio": float(target_fill_ratio),
        "overlay_scale": 1.0,
        "overlay_shift_xy": [0.0, 0.0],
    }

    valid_points = verts_2d[valid_vertices]

    # --- Priority 1: pose-driven similarity transform ---
    if alignment_anchors is not None and valid_points.size:
        src_pts, dst_pts = alignment_anchors
        try:
            sim_scale, sim_R, sim_t = fit_similarity_2d(src_pts, dst_pts)
            verts_2d = apply_similarity_2d(verts_2d, sim_scale, sim_R, sim_t)
            # recompute anchor fit residual for diagnostics
            src_aligned = apply_similarity_2d(src_pts, sim_scale, sim_R, sim_t)
            anchor_errors = np.linalg.norm(src_aligned - dst_pts, axis=1)
            transform_debug = {
                "alignment_mode": "pose_driven_similarity",
                "target_bbox_xyxy": None,
                "target_fill_ratio": float(target_fill_ratio),
                "overlay_scale": float(sim_scale),
                "overlay_shift_xy": sim_t.tolist(),
                "sim_rotation_deg": float(np.degrees(np.arctan2(sim_R[1, 0], sim_R[0, 0]))),
                "anchor_mean_error_px": float(np.mean(anchor_errors)),
                "anchor_max_error_px": float(np.max(anchor_errors)),
                "num_anchor_pairs": int(src_pts.shape[0]),
            }
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as _align_err:
            # fall through to bbox-based alignment if similarity fitting fails
            transform_debug["alignment_fallback_reason"] = str(_align_err)

    # --- Priority 2: bbox-based scale + center (legacy fallback) ---
    if transform_debug.get("alignment_mode") == "none" and target_bbox_xyxy is not None and valid_points.size:
        tx1, ty1, tx2, ty2 = [float(v) for v in target_bbox_xyxy]
        mesh_min = valid_points.min(axis=0)
        mesh_max = valid_points.max(axis=0)
        mesh_size = np.maximum(mesh_max - mesh_min, 1e-6)
        mesh_center = (mesh_min + mesh_max) / 2.0
        target_center = np.array([(tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0], dtype=float)
        target_size = np.array([max(tx2 - tx1, 1e-6), max(ty2 - ty1, 1e-6)], dtype=float)
        scale = float(min(target_size[0] / mesh_size[0], target_size[1] / mesh_size[1]) * target_fill_ratio)
        verts_2d = (verts_2d - mesh_center[None, :]) * scale + target_center[None, :]
        transform_debug = {
            "alignment_mode": "bbox_scale_center",
            "target_bbox_xyxy": [tx1, ty1, tx2, ty2],
            "target_fill_ratio": float(target_fill_ratio),
            "overlay_scale": scale,
            "overlay_shift_xy": (target_center - mesh_center * scale).tolist(),
            "mesh_bbox_before_xyxy": [float(mesh_min[0]), float(mesh_min[1]), float(mesh_max[0]), float(mesh_max[1])],
            "mesh_bbox_after_xyxy": [
                float(np.nanmin(verts_2d[valid_vertices, 0])),
                float(np.nanmin(verts_2d[valid_vertices, 1])),
                float(np.nanmax(verts_2d[valid_vertices, 0])),
                float(np.nanmax(verts_2d[valid_vertices, 1])),
            ],
        }

    faces_arr = np.asarray(faces, dtype=np.int32)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise RuntimeError("Expected triangular faces with shape (M, 3).")

    valid_faces = np.all(valid_vertices[faces_arr], axis=1)
    if not np.any(valid_faces):
        raise RuntimeError("No valid mesh faces remained after 3D projection.")

    tri_3d = vertices_3d[faces_arr]
    tri_2d = verts_2d[faces_arr]
    normals = np.cross(tri_3d[:, 1] - tri_3d[:, 0], tri_3d[:, 2] - tri_3d[:, 0])
    normal_mag = np.linalg.norm(normals, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        facing = np.abs(normals[:, 2]) / normal_mag
    facing = np.where(np.isfinite(facing), facing, 0.0)
    depths = np.mean(tri_3d[:, :, 2], axis=1)
    areas = 0.5 * np.abs(
        tri_2d[:, 0, 0] * (tri_2d[:, 1, 1] - tri_2d[:, 2, 1])
        + tri_2d[:, 1, 0] * (tri_2d[:, 2, 1] - tri_2d[:, 0, 1])
        + tri_2d[:, 2, 0] * (tri_2d[:, 0, 1] - tri_2d[:, 1, 1])
    )
    valid_faces &= np.isfinite(depths) & np.isfinite(areas) & (areas > 0.5)

    canvas = np.zeros_like(image_bgr)
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    draw_order = np.argsort(depths[valid_faces])[::-1]
    face_indices = np.flatnonzero(valid_faces)[draw_order]
    base_color = np.asarray(base_color_bgr, dtype=float)

    for face_idx in face_indices:
        tri = tri_2d[face_idx]
        if not np.isfinite(tri).all():
            continue
        tri_int = np.round(tri).astype(np.int32)
        if np.all(tri_int[:, 0] < 0) or np.all(tri_int[:, 0] >= image_w):
            continue
        if np.all(tri_int[:, 1] < 0) or np.all(tri_int[:, 1] >= image_h):
            continue
        intensity = 0.55 + 0.35 * float(facing[face_idx])
        color = np.clip(base_color * intensity, 0, 255).astype(np.uint8).tolist()
        cv2.fillConvexPoly(canvas, tri_int, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, tri_int, 255, lineType=cv2.LINE_AA)

    if int(mask.sum()) == 0:
        raise RuntimeError("Mesh projection produced an empty overlay mask.")

    alpha_mask = (mask.astype(np.float32) / 255.0)[:, :, None] * float(np.clip(alpha, 0.0, 1.0))
    blended = image_bgr.astype(np.float32) * (1.0 - alpha_mask) + canvas.astype(np.float32) * alpha_mask
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(blended, contours, -1, outline_color_bgr, 1, lineType=cv2.LINE_AA)

    return blended, {
        "mask_area_px": int(np.count_nonzero(mask)),
        "mask_area_ratio": float(np.count_nonzero(mask) / float(image_h * image_w)),
        "num_faces_drawn": int(len(face_indices)),
        "num_vertices_projected": int(np.count_nonzero(valid_vertices)),
        **transform_debug,
    }


def render_hybrik_mesh_overlay(
    image_or_path: Union[str, np.ndarray],
    data_or_path: Union[str, Dict[str, object]],
    mesh_obj_path: Optional[str] = None,
    alpha: float = 0.90,
    base_color_bgr: Tuple[int, int, int] = (222, 206, 184),
    outline_color_bgr: Tuple[int, int, int] = (154, 136, 113),
    fit_to_detector_bbox: bool = True,
    use_pose_alignment: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Render the SMPL mesh over the input image with adaptive 1:1 alignment.

    When *use_pose_alignment* is True (the default), a 2D similarity transform
    is fitted from camera-projected joint positions to the UVD joint positions
    predicted by HybrIK.  This makes the mesh overlay precisely match the pose
    visible in the photograph.  If alignment anchor fitting fails, the method
    falls back to detector-bbox-based scaling when *fit_to_detector_bbox* is
    True, and then to the raw projection.
    """
    if isinstance(image_or_path, str):
        image_bgr = cv2.imread(image_or_path)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_or_path}")
        image_path = os.path.abspath(image_or_path)
    else:
        image_bgr = np.asarray(image_or_path).copy()
        image_path = None

    if mesh_obj_path is None:
        if not isinstance(data_or_path, str):
            raise RuntimeError("mesh_obj_path is required when HybrIK JSON is passed as an in-memory object.")
        mesh_obj_path = os.path.join(os.path.dirname(os.path.abspath(data_or_path)), "mesh.obj")
    if not os.path.exists(mesh_obj_path):
        raise FileNotFoundError(f"Mesh OBJ not found: {mesh_obj_path}")

    data = load_hybrik_json(data_or_path)
    projection = fit_hybrik_projection(data)
    translation = data.get("translation")
    if not isinstance(translation, list) or len(translation) != 3:
        raise RuntimeError("Expected `translation` in HybrIK JSON.")

    vertices, faces = load_obj_mesh(mesh_obj_path)
    vertices_cam = vertices + np.asarray(translation, dtype=float)[None, :]

    # Build pose-driven alignment anchors (src = camera-projected joints,
    # dst = UVD joint positions predicted by HybrIK).
    alignment_anchors: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if use_pose_alignment:
        try:
            alignment_anchors = build_pose_alignment_anchors(
                data,
                focal_px=projection["focal_px"],
                cx_px=projection["cx_px"],
                cy_px=projection["cy_px"],
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            alignment_anchors = None

    # Detector bbox is used as a fallback when pose alignment is unavailable.
    target_bbox = None
    if fit_to_detector_bbox and alignment_anchors is None:
        bbox = data.get("detector_bbox_xyxy")
        if isinstance(bbox, list) and len(bbox) == 4:
            target_bbox = bbox

    overlay_bgr, render_debug = render_projected_mesh_overlay(
        image_bgr=image_bgr,
        vertices_3d=vertices_cam,
        faces=faces,
        focal_px=projection["focal_px"],
        cx_px=projection["cx_px"],
        cy_px=projection["cy_px"],
        alpha=alpha,
        base_color_bgr=base_color_bgr,
        outline_color_bgr=outline_color_bgr,
        target_bbox_xyxy=target_bbox,
        alignment_anchors=alignment_anchors,
    )
    debug = {
        "image_path": image_path,
        "mesh_obj_path": os.path.abspath(mesh_obj_path),
        "projection": projection,
        "fit_to_detector_bbox": bool(fit_to_detector_bbox),
        "use_pose_alignment": bool(use_pose_alignment),
        "vertices_count": int(vertices.shape[0]),
        "faces_count": int(faces.shape[0]),
        **render_debug,
    }
    return overlay_bgr, debug



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
    for name, lm in HYBRIK_MP_JOINT_MAP.items():
        row = frame_df[frame_df["lm"] == lm]
        if row.empty:
            continue
        row0 = row.iloc[0]
        x = float(row0[x_col]) * image_w
        y = float(row0[y_col]) * image_h
        if np.isfinite(x) and np.isfinite(y):
            points[name] = np.array([x, y], dtype=float)
    return points


def compute_hybrik_pose_consistency(
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


def compute_hybrik_confidence_stats(data_or_path: Union[str, Dict[str, object]]) -> Dict[str, float]:
    scores = np.array(list(load_hybrik_joint_score_map(data_or_path).values()), dtype=float)
    if scores.size == 0:
        return {}
    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }


def compute_hybrik_coverage_stats(
    hybrik_2d: Dict[str, np.ndarray],
    data_or_path: Union[str, Dict[str, object]],
    image_w: int,
    image_h: int,
) -> Dict[str, float]:
    if not hybrik_2d:
        return {}
    data = load_hybrik_json(data_or_path)
    bbox = data.get("detector_bbox_xyxy") or data.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {}
    x1, y1, x2, y2 = [float(v) for v in bbox]

    inside_image = 0
    inside_bbox = 0
    for pt in hybrik_2d.values():
        x, y = float(pt[0]), float(pt[1])
        if 0 <= x < image_w and 0 <= y < image_h:
            inside_image += 1
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside_bbox += 1
    total = max(len(hybrik_2d), 1)
    return {
        "inside_image_ratio": inside_image / total,
        "inside_detector_bbox_ratio": inside_bbox / total,
    }


def compute_hybrik_symmetry_stats(joints: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, (la, lb, ra, rb) in HYBRIK_SYMMETRY_SEGMENTS.items():
        if la not in joints or lb not in joints or ra not in joints or rb not in joints:
            continue
        left_len = float(np.linalg.norm(joints[lb] - joints[la]))
        right_len = float(np.linalg.norm(joints[rb] - joints[ra]))
        out[f"{label}_left_len"] = left_len
        out[f"{label}_right_len"] = right_len
        if min(left_len, right_len) > 1e-8:
            out[f"{label}_symmetry_ratio"] = max(left_len, right_len) / min(left_len, right_len)
    return out


def compute_hybrik_neighbor_stability(
    neighbor_jsons: List[Tuple[int, Union[str, Dict[str, object]]]]
) -> Dict[str, object]:
    if len(neighbor_jsons) < 2:
        return {}

    per_frame = []
    all_joint_arrays: List[np.ndarray] = []
    joint_order: Optional[List[str]] = None

    for frame_num, payload in neighbor_jsons:
        metrics_3d, _debug = analyze_trophy_pose_3d(payload)
        joints = load_hybrik_joint_xyz_map(payload)
        if joint_order is None:
            joint_order = sorted(joints.keys())
        arr = np.stack([joints[name] for name in joint_order], axis=0)
        all_joint_arrays.append(arr)
        per_frame.append({"frame": int(frame_num), **metrics_3d})

    disp = []
    for prev, curr in zip(all_joint_arrays[:-1], all_joint_arrays[1:]):
        disp.append(float(np.mean(np.linalg.norm(curr - prev, axis=1))))

    stability = {
        "frames": [row["frame"] for row in per_frame],
        "per_frame_metrics": per_frame,
        "mean_consecutive_joint_disp": float(np.mean(disp)),
        "max_consecutive_joint_disp": float(np.max(disp)),
    }
    for key in HYBRIK_3D_ANGLE_KEYS:
        values = np.array([row.get(key, float("nan")) for row in per_frame], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            stability[f"{key}_std"] = float(np.std(finite))
            stability[f"{key}_range"] = float(np.max(finite) - np.min(finite))
    return stability


def build_hybrik_warnings(
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

    # Single-frame SMPL body shape is easy to over-interpret. Keep that explicit in the summary.
    warnings.append("Single-frame HybrIK body shape is not trustworthy; use skeleton and joint angles, not chest/body contour, for coaching.")
    return warnings


def _parse_frame_num_from_paths(explicit: Optional[int], *paths: Optional[str]) -> Optional[int]:
    if explicit is not None:
        return int(explicit)
    for source in paths:
        if not source:
            continue
        name = os.path.basename(source)
        match = re.search(r"(\d+)(?=\.[^.]+$)", name)
        if match:
            return int(match.group(1))
    return None


def summarize_hybrik_verification(
    json_or_path: Union[str, Dict[str, object]],
    image_size: Optional[Tuple[int, int]] = None,
    pose_points_2d: Optional[Dict[str, np.ndarray]] = None,
    neighbor_jsons: Optional[List[Tuple[int, Union[str, Dict[str, object]]]]] = None,
    frame_num: Optional[int] = None,
    pose_csv_path: Optional[str] = None,
    image_path: Optional[str] = None,
    body_shape_model: str = "smpl_neutral_single_frame",
) -> Dict[str, object]:
    data = load_hybrik_json(json_or_path)
    joints = load_hybrik_joint_xyz_map(data)
    angles_3d, debug_3d = analyze_trophy_pose_3d(data)
    hybrik_2d = project_hybrik_joints_2d(data)
    confidence = compute_hybrik_confidence_stats(data)
    symmetry = compute_hybrik_symmetry_stats(joints)

    consistency: Dict[str, object] = {"num_compared": 0, "per_joint": []}
    if pose_points_2d:
        consistency = compute_hybrik_pose_consistency(hybrik_2d, pose_points_2d)

    coverage: Dict[str, float] = {}
    if image_size is not None:
        image_w, image_h = image_size
        coverage = compute_hybrik_coverage_stats(hybrik_2d, data, image_w, image_h)

    stability = compute_hybrik_neighbor_stability(neighbor_jsons or [])
    warnings = build_hybrik_warnings(confidence, consistency, symmetry, stability)

    verdict = "good"
    if warnings:
        verdict = "caution"
    if any("differs noticeably" in item or "varies a lot" in item for item in warnings):
        verdict = "needs_review"

    if image_path is None and isinstance(json_or_path, str):
        image_path = data.get("image_path") if isinstance(data.get("image_path"), str) else None
    if frame_num is None:
        frame_num = _parse_frame_num_from_paths(
            None,
            image_path,
            json_or_path if isinstance(json_or_path, str) else None,
        )

    return {
        "verdict": verdict,
        "angles_3d": angles_3d,
        "angles_debug": debug_3d,
        "confidence": confidence,
        "coverage": coverage,
        "symmetry": symmetry,
        "pose_consistency": consistency,
        "stability": stability,
        "warnings": warnings,
        "body_shape_model": body_shape_model,
        "body_shape_trust": "low",
        "inputs": {
            "json_path": os.path.abspath(json_or_path) if isinstance(json_or_path, str) else None,
            "image_path": os.path.abspath(image_path) if image_path else None,
            "pose_csv": os.path.abspath(pose_csv_path) if pose_csv_path else None,
            "frame_num": frame_num,
        },
    }
