import base64
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr

import serve_score


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(APP_ROOT, "out", "space_runs")
MODEL_PATH = os.path.join(APP_ROOT, "models", "pose_landmarker_full.task")
MAKE_REPORT_SCRIPT = os.path.join(APP_ROOT, "make_report.py")
RUN_HYBRIK_SCRIPT = os.path.join(APP_ROOT, "run_hybrik_image.py")
HYBRIK_ROOT = os.path.join(APP_ROOT, "third_party", "HybrIK-main")
HYBRIK_CHECKPOINT = os.path.join(HYBRIK_ROOT, "pretrained_models", "hybrik_hrnet.pth")
HYBRIK_CONFIG = os.path.join(HYBRIK_ROOT, "configs", "256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml")
PHASE_NAMES = ["trophy", "racket_drop", "contact", "finish"]
THREE_D_MODULES = ["cv2", "numpy", "torch", "torchvision", "easydict", "yaml", "PIL"]
SPACE_BUILD_TAG = "space-build-2026-03-20-gradio6-v3"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _module_missing(name: str) -> bool:
    return importlib.util.find_spec(name) is None


def _runtime_3d_status() -> Tuple[bool, List[str]]:
    issues: List[str] = []
    missing_modules = [name for name in THREE_D_MODULES if _module_missing(name)]
    if missing_modules:
        issues.append(f"missing Python modules: {', '.join(missing_modules)}")
    if not os.path.exists(HYBRIK_ROOT):
        issues.append(f"HybrIK repo not found: {HYBRIK_ROOT}")
    if not os.path.exists(HYBRIK_CONFIG):
        issues.append(f"HybrIK config not found: {HYBRIK_CONFIG}")
    if not os.path.exists(HYBRIK_CHECKPOINT):
        issues.append(f"HybrIK checkpoint not found: {HYBRIK_CHECKPOINT}")
    return (len(issues) == 0, issues)


def _runtime_note() -> str:
    report_ready = os.path.exists(MAKE_REPORT_SCRIPT) and os.path.exists(MODEL_PATH)
    three_d_ready, issues = _runtime_3d_status()
    lines = [
        "# Serve AI Tracker",
        f"- Build tag: `{SPACE_BUILD_TAG}`",
        f"- Gradio runtime: `{getattr(gr, '__version__', 'unknown')}`",
        f"- Python runtime: `{sys.version.split()[0]}`",
        "Upload a serve clip to generate a scoring report. The report tab keeps the existing 2D scoring flow and adds the local 3D keyframe workflow when the runtime supports it.",
        "",
        f"- Report pipeline: {'ready' if report_ready else 'missing required local files'}",
        f"- 3D runtime: {'ready' if three_d_ready else 'degraded'}",
    ]
    if not three_d_ready:
        lines.append(f"- 3D requirements still missing: {'; '.join(issues)}")
    lines.append("- Outputs are written under `out/space_runs/...`.")
    return "\n".join(lines)


def _make_run_dir(prefix: str) -> str:
    _ensure_dir(OUT_ROOT)
    return tempfile.mkdtemp(prefix=f"{prefix}_", dir=OUT_ROOT)


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _existing_files(paths: Sequence[Optional[str]]) -> List[str]:
    return [path for path in paths if path and os.path.exists(path)]


def _pick_first(paths: Sequence[Optional[str]]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _run_command(cmd: List[str]) -> str:
    completed = subprocess.run(
        cmd,
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_parts = []
    if completed.stdout:
        log_parts.append(completed.stdout.strip())
    if completed.stderr:
        log_parts.append(completed.stderr.strip())
    logs = "\n\n".join(part for part in log_parts if part)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}.\n\n{logs}".strip())
    return logs or "Command completed without console output."


def _score_text(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return "n/a"


def _collect_3d_errors(payload: Dict[str, object]) -> List[str]:
    keyframe_3d = payload.get("keyframe_3d") if isinstance(payload.get("keyframe_3d"), dict) else {}
    phase_payloads = keyframe_3d.get("phases") if isinstance(keyframe_3d.get("phases"), dict) else {}
    errors: List[str] = []
    for phase in PHASE_NAMES:
        phase_payload = phase_payloads.get(phase)
        if not isinstance(phase_payload, dict):
            continue
        error = phase_payload.get("error")
        if error:
            errors.append(f"{phase}: {error}")
    return errors


def _collect_3d_log_lines(logs: str) -> List[str]:
    matches = []
    for raw_line in logs.splitlines():
        line = raw_line.strip()
        if "3D" in line or "HybrIK" in line or "servepose" in line:
            matches.append(line)
    return matches[-6:]


def _phase_text(payload: Dict[str, object]) -> str:
    phases = payload.get("phases") if isinstance(payload.get("phases"), dict) else {}
    lines = []
    for name in PHASE_NAMES:
        value = phases.get(name)
        lines.append(f"- `{name}`: {value if value is not None else 'n/a'}")
    return "\n".join(lines)


def _subscore_text(payload: Dict[str, object]) -> str:
    subscores = payload.get("sub_scores") if isinstance(payload.get("sub_scores"), dict) else {}
    if not subscores:
        return "- No sub-scores found."
    lines = []
    for key, value in subscores.items():
        lines.append(f"- `{key}`: {_score_text(value)}")
    return "\n".join(lines)


def _build_report_summary(payload: Dict[str, object], run_dir: str, logs: str) -> str:
    final_score = payload.get("final_score")
    legacy_score = payload.get("legacy_final_score_pose2d", payload.get("legacy_final_score"))
    report_summary = payload.get("report_3d_summary") if isinstance(payload.get("report_3d_summary"), dict) else {}
    keyframe_3d = payload.get("keyframe_3d") if isinstance(payload.get("keyframe_3d"), dict) else {}
    keyframe_phases = keyframe_3d.get("phases") if isinstance(keyframe_3d.get("phases"), dict) else {}
    successful_3d_phases = [
        name
        for name, phase_payload in keyframe_phases.items()
        if isinstance(phase_payload, dict) and not phase_payload.get("error")
    ]

    lines = [
        "## Run Summary",
        f"- Output folder: `{run_dir}`",
        f"- Final score: {_score_text(final_score)}/100",
        f"- 2D pose score: {_score_text(legacy_score)}/100",
    ]
    if isinstance(report_summary.get("coverage_ratio"), (int, float)):
        lines.append(f"- Core metric coverage: {float(report_summary['coverage_ratio']) * 100.0:.0f}%")
    if successful_3d_phases:
        lines.append(f"- 3D keyframes completed: {', '.join(successful_3d_phases)}")
    else:
        lines.append("- 3D keyframes completed: none")
        errors = _collect_3d_errors(payload)
        if errors:
            lines.append(f"- 3D failure: {errors[0]}")
        else:
            log_lines = _collect_3d_log_lines(logs)
            if log_lines:
                lines.append(f"- 3D runtime note: {log_lines[-1]}")

    lines.extend(["", "### Sub-scores", _subscore_text(payload), "", "### Phase Frames", _phase_text(payload)])
    return "\n".join(lines)


def _collect_report_gallery(run_dir: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for phase in PHASE_NAMES:
        keyframe_path = os.path.join(run_dir, "keyframes", f"{phase}.png")
        if os.path.exists(keyframe_path):
            items.append((keyframe_path, f"keyframe: {phase}"))
        compare_path = os.path.join(run_dir, "keyframe_3d", phase, "mesh_overlay_compare.png")
        if os.path.exists(compare_path):
            items.append((compare_path, f"3d overlay: {phase}"))
    return items


def _report_artifacts(run_dir: str, pose_source: str) -> List[str]:
    pose_csv_name = "pose_compare_zero.csv" if pose_source == "compare_zero" else "pose_compare_fast.csv"
    pose_video_name = "pose_compare_zero.mp4" if pose_source == "compare_zero" else "pose_compare_fast.mp4"
    return _existing_files(
        [
            os.path.join(run_dir, "metrics.json"),
            os.path.join(run_dir, "report.html"),
            os.path.join(run_dir, "report.pdf"),
            os.path.join(run_dir, "score_debug.json"),
            os.path.join(run_dir, pose_csv_name),
            os.path.join(run_dir, pose_video_name),
        ]
    )


def _report_video_path(run_dir: str, pose_source: str) -> Optional[str]:
    return _pick_first(
        [
            os.path.join(run_dir, "pose_compare_zero.mp4") if pose_source == "compare_zero" else None,
            os.path.join(run_dir, "pose_compare_fast.mp4"),
            os.path.join(run_dir, "pose_compare_zero.mp4"),
        ]
    )


def _guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }.get(ext, "application/octet-stream")


def _inline_report_html(run_dir: str) -> str:
    html_path = os.path.join(run_dir, "report.html")
    if not os.path.exists(html_path):
        return "<div>report.html not found.</div>"

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    def _replace_src(match: re.Match[str]) -> str:
        quote = match.group(1)
        src = match.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        abs_path = os.path.abspath(os.path.join(run_dir, src.split("?", 1)[0]))
        if not os.path.exists(abs_path):
            return match.group(0)
        with open(abs_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("ascii")
        return f"src={quote}data:{_guess_mime_type(abs_path)};base64,{encoded}{quote}"

    return re.sub(r"src=(['\"])([^'\"]+)\1", _replace_src, html)


def run_serve_report(
    video_path: Optional[str],
    pose_csv_path: Optional[str],
    language: str,
    pose_source: str,
    hand_mode: str,
    quantile_calib: bool,
    verify_3d: str,
    verify_device: str,
) -> Tuple[str, str, Optional[str], List[Tuple[str, str]], List[str], Dict[str, object], str]:
    if not video_path:
        raise gr.Error("Upload a serve video first.")
    if not os.path.exists(MAKE_REPORT_SCRIPT):
        raise gr.Error(f"Missing script: {MAKE_REPORT_SCRIPT}")
    if not os.path.exists(MODEL_PATH):
        raise gr.Error(f"Missing pose model: {MODEL_PATH}")

    lang_code = "zh" if language.startswith("Chinese") else "en"
    run_dir = _make_run_dir("serve_report")
    cmd = [
        sys.executable,
        MAKE_REPORT_SCRIPT,
        "--video",
        os.path.abspath(video_path),
        "--model",
        MODEL_PATH,
        "--out_dir",
        run_dir,
        "--lang",
        lang_code,
        "--pose_source",
        pose_source,
        "--hand_mode",
        hand_mode,
        "--quantile_calib",
        "on" if quantile_calib else "off",
        "--verify_3d",
        verify_3d,
        "--verify_3d_device",
        verify_device,
    ]
    if pose_csv_path:
        cmd.extend(["--pose_csv", os.path.abspath(pose_csv_path)])

    logs = _run_command(cmd)
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise gr.Error(f"Report finished but metrics.json is missing in {run_dir}")

    payload = _read_json(metrics_path)
    summary = _build_report_summary(payload, run_dir, logs)
    html_preview = _inline_report_html(run_dir)
    annotated_video = _report_video_path(run_dir, pose_source)
    gallery = _collect_report_gallery(run_dir)
    downloads = _report_artifacts(run_dir, pose_source)
    return summary, html_preview, annotated_video, gallery, downloads, payload, logs


def _format_trophy_metrics(metrics: Dict[str, object], debug: Dict[str, object]) -> str:
    metric_rows = [
        ("left_knee_angle_deg", "Left knee angle"),
        ("right_knee_angle_deg", "Right knee angle"),
        ("trunk_inclination_deg", "Trunk inclination"),
        ("shoulder_tilt_deg", "Shoulder tilt"),
        ("right_elbow_angle_deg", "Right elbow angle"),
    ]
    lines = ["## 3D Trophy Metrics"]
    for key, label in metric_rows:
        lines.append(f"- {label}: {_score_text(metrics.get(key))}")
    lines.append(f"- Valid 3D joints: {debug.get('num_joints_loaded', 'n/a')}")
    lines.append(f"- Y axis points down: {debug.get('y_axis_down', 'n/a')}")
    return "\n".join(lines)


def _three_d_artifacts(out_dir: str) -> List[str]:
    return _existing_files(
        [
            os.path.join(out_dir, "overlay.jpg"),
            os.path.join(out_dir, "bbox_2d.jpg"),
            os.path.join(out_dir, "joints_2d.jpg"),
            os.path.join(out_dir, "mesh.obj"),
            os.path.join(out_dir, "joints3d.json"),
            os.path.join(out_dir, "result.pkl"),
        ]
    )


def run_single_frame_3d(
    image_path: Optional[str],
    device: str,
) -> Tuple[Optional[str], str, List[str], Dict[str, object], str]:
    if not image_path:
        raise gr.Error("Upload a keyframe image first.")
    if not os.path.exists(RUN_HYBRIK_SCRIPT):
        raise gr.Error(f"Missing script: {RUN_HYBRIK_SCRIPT}")

    ready, issues = _runtime_3d_status()
    if not ready:
        raise gr.Error("3D runtime is not ready: " + "; ".join(issues))

    out_dir = _make_run_dir("single_frame_3d")
    cmd = [
        sys.executable,
        RUN_HYBRIK_SCRIPT,
        "--image",
        os.path.abspath(image_path),
        "--out_dir",
        out_dir,
        "--device",
        device,
    ]
    logs = _run_command(cmd)

    json_path = os.path.join(out_dir, "joints3d.json")
    if not os.path.exists(json_path):
        raise gr.Error(f"3D run finished but joints3d.json is missing in {out_dir}")

    metrics, debug = serve_score.analyze_trophy_pose_3d(json_path)
    summary = _format_trophy_metrics(metrics, debug)
    payload = {
        "metrics": metrics,
        "debug": debug,
        "out_dir": out_dir,
    }
    overlay_path = _pick_first([os.path.join(out_dir, "overlay.jpg")])
    downloads = _three_d_artifacts(out_dir)
    return overlay_path, summary, downloads, payload, logs


with gr.Blocks(title="Serve AI Tracker") as demo:
    gr.Markdown(_runtime_note())

    with gr.Tab("Serve Report"):
        with gr.Row():
            video_input = gr.Video(label="Serve video", format="mp4")
            pose_csv_input = gr.File(label="Existing pose CSV (optional)", type="filepath")

        with gr.Row():
            language_input = gr.Dropdown(
                choices=["Chinese (zh)", "English (en)"],
                value="Chinese (zh)",
                label="Report language",
            )
            pose_source_input = gr.Dropdown(
                choices=["compare_zero", "smooth", "raw"],
                value="compare_zero",
                label="Pose source",
            )
            hand_mode_input = gr.Dropdown(
                choices=["auto", "A", "B"],
                value="auto",
                label="Hand mode",
            )

        with gr.Row():
            quantile_input = gr.Checkbox(value=False, label="Enable quantile calibration")
            verify_3d_input = gr.Dropdown(
                choices=["auto", "on", "off"],
                value="on",
                label="3D keyframe analysis",
            )
            verify_device_input = gr.Dropdown(
                choices=["auto", "cpu", "cuda:0"],
                value="auto",
                label="3D device",
            )

        report_button = gr.Button("Generate report", variant="primary")

        report_summary = gr.Markdown(label="Summary")
        report_html = gr.HTML(label="Report Preview")
        report_video = gr.Video(label="Annotated pose video")
        report_gallery = gr.Gallery(label="Keyframes and 3D overlays", height=420)
        report_downloads = gr.File(label="Download artifacts", file_count="multiple")
        report_json = gr.JSON(label="metrics.json preview")
        report_logs = gr.Textbox(label="Pipeline logs", lines=18)

        report_button.click(
            fn=run_serve_report,
            inputs=[
                video_input,
                pose_csv_input,
                language_input,
                pose_source_input,
                hand_mode_input,
                quantile_input,
                verify_3d_input,
                verify_device_input,
            ],
            outputs=[
                report_summary,
                report_html,
                report_video,
                report_gallery,
                report_downloads,
                report_json,
                report_logs,
            ],
        )

    with gr.Tab("Single Frame 3D"):
        image_input = gr.Image(label="Keyframe image", type="filepath")
        frame_device_input = gr.Dropdown(
            choices=["auto", "cpu", "cuda:0"],
            value="auto",
            label="3D device",
        )
        frame_button = gr.Button("Run 3D modeling", variant="primary")

        frame_overlay = gr.Image(label="3D overlay")
        frame_summary = gr.Markdown(label="3D summary")
        frame_downloads = gr.File(label="Download artifacts", file_count="multiple")
        frame_json = gr.JSON(label="3D JSON preview")
        frame_logs = gr.Textbox(label="3D logs", lines=18)

        frame_button.click(
            fn=run_single_frame_3d,
            inputs=[image_input, frame_device_input],
            outputs=[frame_overlay, frame_summary, frame_downloads, frame_json, frame_logs],
        )


if __name__ == "__main__":
    demo.queue()
    demo.launch()
