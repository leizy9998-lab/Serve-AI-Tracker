import argparse
import json
import os

import numpy as np

import serve_score


def analyze_3d_joints(json_path: str):
    return serve_score.analyze_trophy_pose_3d(json_path)


def _fmt(value: float) -> str:
    if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
        return "nan"
    return f"{float(value):.1f}°"


def save_metrics(out_path: str, metrics: dict, debug: dict) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "debug": debug}, f, ensure_ascii=False, indent=2)


def print_report(metrics: dict, debug: dict) -> None:
    print(f"Loaded {debug['num_joints_loaded']} valid 3D joints.")
    print(f"Axis inference: Y 轴{'向下为正' if debug['y_axis_down'] else '向上为正'}")
    print()
    print("--- 3D 发球奖杯姿势 (Trophy Pose) 体测报告 ---")
    print(f"左膝夹角 (3D): {_fmt(metrics['left_knee_angle_deg'])}")
    print(f"右膝夹角 (3D): {_fmt(metrics['right_knee_angle_deg'])}")
    print(f"躯干倾角 (偏离绝对垂直): {_fmt(metrics['trunk_inclination_deg'])}")
    print(f"双肩倾斜角 (相对水平面): {_fmt(metrics['shoulder_tilt_deg'])}")
    print(f"挥拍臂肘部夹角 (右手 3D): {_fmt(metrics['right_elbow_angle_deg'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HybrIK 3D trophy-pose JSON and compute biomechanical angles.")
    parser.add_argument(
        "--json_path",
        default=os.path.join("out", "hybrik_trophy_125", "joints3d.json"),
        help="Path to the HybrIK joints3d.json file.",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="Optional path to save computed metrics JSON. Defaults to a sibling file named trophy_3d_metrics.json.",
    )
    args = parser.parse_args()

    metrics, debug = serve_score.analyze_trophy_pose_3d(args.json_path)
    print_report(metrics, debug)

    out_path = args.out_path or os.path.join(os.path.dirname(os.path.abspath(args.json_path)), "trophy_3d_metrics.json")
    save_metrics(out_path, metrics, debug)
    print()
    print(f"Saved 3D metrics to: {out_path}")


if __name__ == "__main__":
    main()
