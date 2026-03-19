import argparse
import os
import json
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 简化版骨架连线（够你验收用）
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 31), (27, 29),
    (24, 26), (26, 28), (28, 32), (28, 30),
]

def draw_pose(frame_bgr, lms, min_vis=0.5):
    h, w = frame_bgr.shape[:2]

    # lines
    for a, b in POSE_CONNECTIONS:
        la, lb = lms[a], lms[b]
        va = getattr(la, "visibility", 1.0)
        vb = getattr(lb, "visibility", 1.0)
        if va < min_vis or vb < min_vis:
            continue
        xa, ya = int(la.x * w), int(la.y * h)
        xb, yb = int(lb.x * w), int(lb.y * h)
        cv2.line(frame_bgr, (xa, ya), (xb, yb), (0, 255, 0), 2)

    # points
    for lm in lms:
        v = getattr(lm, "visibility", 1.0)
        if v < min_vis:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--model", default="models/pose_landmarker_full.task", help="pose landmarker .task model path")
    ap.add_argument("--out_csv", default="out/pose.csv", help="output csv")
    ap.add_argument("--out_video", default="out/pose_vis.mp4", help="output visualization video (optional)")
    ap.add_argument("--step", type=int, default=1, help="process every N frames (speed-up). 1=every frame")
    ap.add_argument("--min_vis", type=float, default=0.5, help="min visibility for drawing")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # video writer (same fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, float(fps), (width, height))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    rows = []
    frame_idx = 0
    processed = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % args.step != 0:
                # 仍然把原帧写出去，保持视频长度一致（方便你对齐）
                writer.write(frame_bgr)
                frame_idx += 1
                continue

            timestamp_ms = int((frame_idx / fps) * 1000)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]              # 33 normalized landmarks
                wlms = result.pose_world_landmarks[0]       # 33 world landmarks (meters, hip-mid origin)

                # draw
                vis_frame = frame_bgr.copy()
                draw_pose(vis_frame, lms, min_vis=args.min_vis)
                writer.write(vis_frame)

                # dump landmarks
                for i in range(33):
                    lm = lms[i]
                    wlm = wlms[i]
                    rows.append({
                        "frame": frame_idx,
                        "ts_ms": timestamp_ms,
                        "lm_id": i,
                        "x": lm.x, "y": lm.y, "z": lm.z,
                        "vis": getattr(lm, "visibility", np.nan),
                        "wx": wlm.x, "wy": wlm.y, "wz": wlm.z,
                        "wvis": getattr(wlm, "visibility", np.nan),
                    })
            else:
                writer.write(frame_bgr)

            processed += 1
            frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "video": args.video,
        "model": args.model,
        "fps": fps,
        "width": width,
        "height": height,
        "step": args.step,
        "frames_with_pose": int(df["frame"].nunique()) if len(df) else 0,
        "rows": int(len(df)),
    }
    with open(os.path.join(os.path.dirname(args.out_csv), "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("DONE.")
    print("CSV :", args.out_csv)
    print("VIS :", args.out_video)
    print("META:", os.path.join(os.path.dirname(args.out_csv), "meta.json"))

if __name__ == "__main__":
    main()
