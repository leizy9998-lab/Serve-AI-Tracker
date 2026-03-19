import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd

import serve_score


def _synthetic_pose(n: int = 30) -> pd.DataFrame:
    frames = np.arange(n, dtype=int)
    ts_ms = (frames * (1000.0 / 60.0)).astype(float)
    rows = []

    # Simple synthetic: left hand toss (up then down), right hand racket drop then contact
    for i in range(n):
        t = i / max(n - 1, 1)
        # y smaller is higher
        left_wrist_y = 0.6 - 0.25 * np.sin(min(np.pi * t * 1.4, np.pi))
        right_wrist_y = 0.7 + 0.2 * np.sin(min(np.pi * t * 1.8, np.pi))
        left_shoulder_y = 0.5
        right_shoulder_y = 0.52
        right_elbow_y = 0.55
        nose_y = 0.35
        left_hip_y = 0.7
        right_hip_y = 0.7
        left_knee_y = 0.85
        right_knee_y = 0.86
        left_ankle_y = 0.95
        right_ankle_y = 0.95

        lm_values = {
            0: (0.5, nose_y),
            11: (0.45, left_shoulder_y),
            12: (0.55, right_shoulder_y),
            13: (0.45, 0.58),
            14: (0.55, right_elbow_y),
            15: (0.45, left_wrist_y),
            16: (0.55, right_wrist_y),
            23: (0.46, left_hip_y),
            24: (0.54, right_hip_y),
            25: (0.46, left_knee_y),
            26: (0.54, right_knee_y),
            27: (0.46, left_ankle_y),
            28: (0.54, right_ankle_y),
        }
        for lm, (x, y) in lm_values.items():
            rows.append(
                {
                    "frame": frames[i],
                    "ts_ms": ts_ms[i],
                    "lm": lm,
                    "raw_x": x,
                    "raw_y": y,
                    "smooth_x": x,
                    "smooth_y": y,
                    "vis": 0.9,
                }
            )
    return pd.DataFrame(rows)


def _print_result(tag: str, res: dict) -> None:
    print(f"\n[{tag}]")
    print(f"final_score: {res['final_score']:.2f}")
    print("subscores:", {k: round(v, 2) for k, v in res["subscores"].items()})
    print("metrics:", {k: round(v, 4) for k, v in res["metrics"].items() if k in {
        "shoulder_tilt_trophy", "shoulder_tilt_trophy_signed",
        "racket_drop_depth", "racket_drop_depth_raw",
        "contact_above_head",
    }})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_csv", default=None, help="Path to pose CSV (long format).")
    ap.add_argument("--use_coords", choices=["raw", "smooth"], default="smooth")
    ap.add_argument("--hand_mode", choices=["auto", "A", "B"], default="auto")
    args = ap.parse_args()

    if args.pose_csv:
        df = pd.read_csv(args.pose_csv, encoding="utf-8-sig")
    else:
        df = _synthetic_pose()

    dual = serve_score.analyze_serve_dual(df, use_coords=args.use_coords, hand_mode=args.hand_mode)
    _print_result("A (left toss / right racket)", dual["A"])
    _print_result("B (right toss / left racket)", dual["B"])

    print("\n[Selected]")
    print("hand_mode_selected:", dual["hand_mode_selected"])
    print("mirror_suspect:", dual["mirror_suspect"])
    print("hand_mode_guess:", dual["hand_mode_guess"])
    sel = dual["selected"]
    print("selected_final_score:", f"{sel['final_score']:.2f}")
    print("selected_subscores:", {k: round(v, 2) for k, v in sel["subscores"].items()})
    print(
        "selected_metrics:",
        {k: round(v, 4) for k, v in sel["metrics"].items() if k in {
            "shoulder_tilt_trophy", "shoulder_tilt_trophy_signed",
            "racket_drop_depth", "racket_drop_depth_raw",
            "contact_above_head",
        }},
    )


if __name__ == "__main__":
    main()
