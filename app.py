import math
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import pandas as pd

import serve_score


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(APP_ROOT, "out", "space_runs")
SPACE_BUILD_TAG = "space-build-2026-03-20-2d-v1"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_run_dir(prefix: str) -> str:
    _ensure_dir(OUT_ROOT)
    return tempfile.mkdtemp(prefix=f"{prefix}_", dir=OUT_ROOT)


def _resolve_uploaded_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        video_info = value.get("video")
        if isinstance(video_info, dict):
            for key in ("path", "name"):
                candidate = video_info.get(key)
                if isinstance(candidate, str):
                    return candidate
    candidate = getattr(value, "name", None)
    if isinstance(candidate, str):
        return candidate
    return None


def _safe_metric_text(value: Any, suffix: str = "") -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.1f}{suffix}"
    return f"nan{suffix}" if suffix else "nan"


def _safe_frame_text(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and math.isfinite(value):
        return str(int(round(value)))
    return "N/A"


def process_video_and_extract(video_path: str, out_dir: str) -> Tuple[pd.DataFrame, str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or not np.isfinite(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError("Video has no readable frames.")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    overlay_path = os.path.join(out_dir, "pose_compare_fast.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, int(max(1, round(fps))), (width, height))

    rows = []
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = frame_idx * (1000.0 / fps)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

                for lm_id, lm in enumerate(results.pose_landmarks.landmark):
                    rows.append(
                        {
                            "frame": frame_idx,
                            "ts_ms": ts_ms,
                            "lm": lm_id,
                            "raw_x": lm.x,
                            "raw_y": lm.y,
                            "smooth_x": lm.x,
                            "smooth_y": lm.y,
                            "vis": lm.visibility,
                        }
                    )

            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    return pd.DataFrame(rows), overlay_path


def _build_report(result: Dict[str, Any], out_dir: str) -> str:
    final_score = result.get("final_score", 0.0)
    subscores = result.get("subscores") if isinstance(result.get("subscores"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    phases = result.get("phases") if isinstance(result.get("phases"), dict) else {}

    impact_ratio = metrics.get("impact_bh_ratio", float("nan"))
    impact_note = ""
    if not (isinstance(impact_ratio, (int, float)) and math.isfinite(float(impact_ratio))):
        impact_note = " (pose likely out of frame or occluded near contact)"

    lines = [
        "## 2D Serve Report",
        f"- Build tag: `{SPACE_BUILD_TAG}`",
        f"- Output folder: `{out_dir}`",
        f"- Final score: {_safe_metric_text(final_score)}/100",
        "",
        "### Core Subscores",
        f"- Knee Bend: {_safe_metric_text(subscores.get('knee_flexion'))}",
        f"- Trunk Inclination: {_safe_metric_text(subscores.get('trunk_inclination'))}",
        f"- Impact Height: {_safe_metric_text(subscores.get('impact_height'))}",
        f"- Timing: {_safe_metric_text(subscores.get('timing'))}",
        "",
        "### Phase Frames",
        f"- Toss peak: frame {_safe_frame_text(phases.get('toss_peak'))}",
        f"- Trophy: frame {_safe_frame_text(phases.get('trophy'))}",
        f"- Racket drop: frame {_safe_frame_text(phases.get('racket_drop'))}",
        f"- Contact: frame {_safe_frame_text(phases.get('contact'))}",
        "",
        "### Biomechanics",
        f"- Estimated knee flexion: {_safe_metric_text(metrics.get('knee_flexion_est_deg'), ' deg')}",
        f"- Estimated trunk inclination: {_safe_metric_text(metrics.get('trunk_inclination_est_deg'), ' deg')}",
        f"- Impact to body-height ratio: {_safe_metric_text(impact_ratio, 'x')}{impact_note}",
    ]
    return "\n".join(lines)


def analyze_serve_cloud(video_value: Any) -> Tuple[Optional[str], str]:
    video_path = _resolve_uploaded_path(video_value)
    if not video_path or not os.path.exists(video_path):
        return None, "Upload a serve video first."

    run_dir = _make_run_dir("space_2d")
    try:
        df, overlay_video = process_video_and_extract(video_path, run_dir)
        if df.empty:
            return None, "No valid pose landmarks were detected in the uploaded video."

        result = serve_score.analyze_serve(df, use_coords="smooth")
        report = _build_report(result, run_dir)
        return overlay_video, report
    except Exception as exc:
        return None, f"Analysis failed: {exc}"


with gr.Blocks(title="Serve AI Tracker") as demo:
    gr.Markdown(
        "\n".join(
            [
                "## Space 2D Serve Analysis",
                f"- Build tag: `{SPACE_BUILD_TAG}`",
                f"- Gradio runtime: `{getattr(gr, '__version__', 'unknown')}`",
                f"- Python runtime: `{os.sys.version.split()[0]}`",
                "- This space now runs the original 2D scoring flow only.",
                "- Local 3D modeling remains available through `make_report.py` and `run_hybrik_image.py`.",
            ]
        )
    )

    video_input = gr.Video(label="Upload serve video", format="mp4")
    submit_btn = gr.Button("Run 2D analysis", variant="primary")

    with gr.Row():
        video_output = gr.Video(label="Pose overlay replay")
        text_output = gr.Markdown(label="Scoring report")

    submit_btn.click(fn=analyze_serve_cloud, inputs=video_input, outputs=[video_output, text_output])


if __name__ == "__main__":
    demo.launch()
