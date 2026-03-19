import argparse
import os
from typing import Optional

import cv2
import pandas as pd

from serve_score import analyze_serve


def _resolve_trophy_frame(result: dict) -> Optional[int]:
    phases = result.get("phases") or {}
    frames = result.get("frames")
    trophy_idx = phases.get("trophy")
    if trophy_idx is None:
        return None

    trophy_idx = int(trophy_idx)
    if frames is not None and 0 <= trophy_idx < len(frames):
        return int(frames[trophy_idx])

    return trophy_idx


def extract_trophy_image(video_path: str, pose_csv_path: str, output_dir: str, use_coords: str = "smooth") -> Optional[str]:
    df = pd.read_csv(pose_csv_path, encoding="utf-8-sig")

    print("Analyzing serve phases...")
    result = analyze_serve(df, use_coords=use_coords)
    trophy_frame_num = _resolve_trophy_frame(result)
    if trophy_frame_num is None:
        print("Failed to detect trophy pose.")
        return None

    print(f"Trophy pose located at video frame {trophy_frame_num}.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, trophy_frame_num)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {trophy_frame_num} from {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"trophy_frame_{trophy_frame_num}.jpg")
        if not cv2.imwrite(out_path, frame):
            raise RuntimeError(f"Failed to save image: {out_path}")
    finally:
        cap.release()

    print(f"Saved trophy pose image to: {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract the trophy-pose frame from a serve video.")
    ap.add_argument("--video", required=True, help="Path to the source serve video.")
    ap.add_argument("--pose_csv", required=True, help="Path to the long-format pose CSV.")
    ap.add_argument(
        "--out_dir",
        default=os.path.join("out", "trophy_extract"),
        help="Output directory for the extracted image.",
    )
    ap.add_argument(
        "--use_coords",
        choices=["smooth", "raw"],
        default="smooth",
        help="Coordinate set used by serve_score analysis.",
    )
    args = ap.parse_args()

    extract_trophy_image(
        video_path=args.video,
        pose_csv_path=args.pose_csv,
        output_dir=args.out_dir,
        use_coords=args.use_coords,
    )


if __name__ == "__main__":
    main()
