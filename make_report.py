import argparse
import json
import time
import os
import subprocess
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

import serve_score
from i18n_zh import zh_label
from i18n_zh import zh_label


LIT_KNEE_MEAN = 64.5
LIT_KNEE_SD = 9.7
LIT_TRUNK_MEAN = 25.0
LIT_TRUNK_SD = 7.1

IMPACT_NORM_RANGE = (1.45, 1.55)
TIMING_RANGES = {
    "timing_toss_to_trophy": (0.3, 1.4),
    "timing_trophy_to_drop": (0.12, 0.26),
    "timing_drop_to_contact": (0.08, 0.16),
}


I18N = {
    "zh": {
        "title": "发球动作评估报告",
        "final_score": "总评分（0–100）",
        "hand_assignment": "手别/镜像判断",
        "data_source": "数据源与参数",
        "biomech": "生物力学指标（软评分）",
        "quality_flags": "数据质量（遮挡/缺失）",
        "subscores": "分项得分",
        "diagnosis": "动作诊断与建议",
        "keyframes": "关键帧",
        "plots": "曲线图",
        "metrics": "指标明细",
        "category": "项目",
        "score": "分数",
        "notes": "建议",
        "metric": "指标",
        "value": "数值",
        "risk_flags": "风险提示",
        "range_lit": "文献均值±1SD",
        "range_heur": "经验范围（可调）",
        "range_calib": "样本分位数范围",
        "selected": "选中",
        "mirror_suspect": "镜像疑似",
        "guess": "自动判断",
        "z_scores": "标准分（Z）",
        "stability": "稳定性",
        "penalties": "扣分项",
        "risk_none": "无明显风险提示",
        "confidence": "置信度",
    },
    "en": {
        "title": "Serve Score Report",
        "final_score": "Final score (0-100)",
        "hand_assignment": "Hand Assignment",
        "data_source": "Data Source",
        "biomech": "Biomechanics (Soft Scoring)",
        "quality_flags": "Quality Flags",
        "subscores": "Sub-scores",
        "diagnosis": "Diagnosis",
        "keyframes": "Keyframes",
        "plots": "Plots",
        "metrics": "Metrics",
        "category": "Category",
        "score": "Score",
        "notes": "Notes",
        "metric": "Metric",
        "value": "Value",
        "risk_flags": "Risk Flags",
        "range_lit": "mean±1SD",
        "range_heur": "heuristic",
        "range_calib": "percentile",
        "selected": "Selected",
        "mirror_suspect": "Mirror Suspect",
        "guess": "Guess",
        "z_scores": "Z-Score",
        "stability": "Stability",
        "penalties": "Penalties",
        "risk_none": "No obvious risk flags",
        "confidence": "Confidence",
    },
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _posix_path(path: str) -> str:
    return path.replace("\\", "/")


def _run_extract(
    video: str,
    model: str,
    out_csv: str,
    out_video: str,
    smooth_mode: str,
    v_thr: float,
    gap_max: int,
    smooth_window: int,
) -> Dict[str, str]:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "extract_pose_compare.py"),
        "--video",
        video,
        "--model",
        model,
        "--out_csv",
        out_csv,
        "--out_video",
        out_video,
        "--smooth_mode",
        smooth_mode,
        "--v_thr",
        str(v_thr),
        "--gap_max",
        str(gap_max),
        "--smooth_window",
        str(smooth_window),
    ]
    subprocess.run(cmd, check=True)
    return {"pose_csv": out_csv, "pose_video": out_video}


def _save_keyframes(video: str, phases: Dict[str, int], out_dir: str) -> Dict[str, str]:
    key_dir = os.path.join(out_dir, "keyframes")
    _ensure_dir(key_dir)
    # Clear old keyframes to avoid browser cache confusion
    for name in ["trophy", "racket_drop", "contact", "finish"]:
        old_path = os.path.join(key_dir, f"{name}.png")
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    outputs = {}
    for name in ["trophy", "racket_drop", "contact", "finish"]:
        idx = phases.get(name)
        if idx is None:
            continue
        idx = int(max(0, min(idx, max(frame_count - 1, 0))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = os.path.join(key_dir, f"{name}.png")
        cv2.imwrite(out_path, frame)
        outputs[name] = _posix_path(os.path.join("keyframes", f"{name}.png"))

    cap.release()
    return outputs


def _plot_series(ts_s: np.ndarray, y: np.ndarray, title: str, ylabel: str, path: str, phases: Dict[str, int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3))
    plt.plot(ts_s, y, color="#2c7fb8", linewidth=2)
    for name, color in [
        ("toss_peak", "#8c510a"),
        ("trophy", "#01665e"),
        ("racket_drop", "#7b3294"),
        ("contact", "#b2182b"),
        ("finish", "#4d4d4d"),
    ]:
        idx = phases.get(name)
        if idx is not None and 0 <= idx < len(ts_s):
            plt.axvline(ts_s[idx], color=color, linestyle="--", linewidth=1, alpha=0.7)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _diagnosis_biomech(
    subscores: Dict[str, float],
    biomech: Dict[str, object],
    lang: str,
) -> List[Dict[str, str]]:
    t = I18N.get(lang, I18N["en"])

    if lang == "zh":
        names = {
            "knee": "膝屈（Knee Flexion）",
            "trunk": "躯干倾角（Trunk Inclination）",
            "impact": "击球高度（Impact Height）",
            "timing": "时序（Timing）",
            "risk": "风险提示（Risk Flags）",
        }
        msgs = {
            "knee": (
                "膝屈幅度良好，发力链条稳定。",
                "膝屈适中，可进一步优化启动深度。",
                "膝屈不足或过大，建议调整下肢负荷。",
            ),
            "trunk": (
                "躯干倾角合理，力量传导顺畅。",
                "躯干倾角尚可，可优化前倾协同。",
                "躯干倾角偏小/偏大，建议调整上体控制。",
            ),
            "impact": (
                "击球高度较高，有利于出球角度。",
                "击球高度尚可，可进一步抬高击球点。",
                "击球高度偏低，建议提高抬臂与蹬伸。",
            ),
            "timing": (
                "时序协调，动作衔接良好。",
                "时序尚可，可优化各相位过渡。",
                "时序偏乱，建议放慢并重建节奏。",
            ),
        }
    else:
        names = {
            "knee": "Knee Flexion",
            "trunk": "Trunk Inclination",
            "impact": "Impact Height",
            "timing": "Timing",
            "risk": "Risk Flags",
        }
        msgs = {
            "knee": (
                "Good knee flexion; stable lower-body loading.",
                "Moderate knee flexion; could deepen loading.",
                "Knee flexion too shallow or too deep; adjust load.",
            ),
            "trunk": (
                "Trunk inclination is well coordinated.",
                "Trunk inclination is okay; improve forward tilt timing.",
                "Trunk inclination too small/large; adjust torso control.",
            ),
            "impact": (
                "Impact height is high; good release angle.",
                "Impact height is acceptable; consider raising the contact.",
                "Impact height is low; extend and raise contact point.",
            ),
            "timing": (
                "Timing is coordinated across phases.",
                "Timing is acceptable; refine phase transitions.",
                "Timing is inconsistent; slow down and rebuild rhythm.",
            ),
        }

    def line(name: str, score: float, good: str, ok: str, bad: str) -> Dict[str, str]:
        if score >= 80:
            text = good
        elif score >= 60:
            text = ok
        else:
            text = bad
        return {"name": name, "score": f"{score:.1f}", "text": text}

    rows = [
        line(names["knee"], subscores.get("knee_flexion", 0.0), *msgs["knee"]),
        line(names["trunk"], subscores.get("trunk_inclination", 0.0), *msgs["trunk"]),
        line(names["impact"], subscores.get("impact_height", 0.0), *msgs["impact"]),
        line(names["timing"], subscores.get("timing", 0.0), *msgs["timing"]),
    ]

    risk_flags = biomech.get("risk_flags") or []
    risk_map = {
        "knee_flexion_out_of_range": (
            "膝屈幅度超出正常范围：可能影响发力或下肢负荷。"
            if lang == "zh"
            else "Knee flexion out of range; may affect loading or power chain."
        ),
        "trunk_inclination_risk": (
            "躯干倾角偏小/偏大：可能影响发力链条与稳定性。"
            if lang == "zh"
            else "Trunk inclination too small/large; may affect power chain stability."
        ),
        "elbow_angle_inconsistent": (
            "肘角波动较大：可能存在发力不连贯/检测不稳定。"
            if lang == "zh"
            else "Elbow angle inconsistent; possible discontinuity or tracking noise."
        ),
        "shoulder_tilt_inconsistent": (
            "肩线倾斜波动较大：可能存在稳定性不足。"
            if lang == "zh"
            else "Shoulder tilt varies widely; possible stability issue."
        ),
    }
    if risk_flags:
        risk_text = "；".join([risk_map.get(r, r) for r in risk_flags])
    else:
        risk_text = t["risk_none"]
    rows.append({"name": names["risk"], "score": "-", "text": risk_text})
    return rows


def _try_export_pdf(html_path: str, pdf_path: str) -> bool:
    try:
        from weasyprint import HTML

        HTML(filename=html_path).write_pdf(pdf_path)
        return True
    except Exception:
        try:
            import pdfkit

            pdfkit.from_file(html_path, pdf_path)
            return True
        except Exception:
            return False


def _find_calibration_dir(repo_root: str) -> Optional[str]:
    candidates = [
        os.path.join(repo_root, "calibration"),
        os.path.join(repo_root, "out", "calibration"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _load_calibration_ranges(repo_root: str) -> Dict[str, tuple]:
    calib_dir = _find_calibration_dir(repo_root)
    if not calib_dir:
        return {}

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
            biomech_raw = biomech.get("raw") or {}

            def pick_metric(key: str) -> float:
                if key == "impact_height_norm":
                    for alt in (
                        biomech_raw.get("impact_height_norm"),
                        metrics.get("impact_height_norm"),
                        metrics.get("contact_above_head"),
                    ):
                        if isinstance(alt, (int, float)) and np.isfinite(alt):
                            return float(alt)
                    return float("nan")
                return float(metrics.get(key, float("nan")))

            for k in targets:
                val = pick_metric(k)
                if np.isfinite(val):
                    values[k].append(val)

    ranges: Dict[str, tuple] = {}
    for k, series in values.items():
        if len(series) < 5:
            continue
        lo, hi = np.nanpercentile(series, [15, 85])
        ranges[k] = (float(lo), float(hi))
    return ranges


def _format_risk_flags(flags: List[str], lang: str) -> str:
    if not flags:
        return I18N.get(lang, I18N["en"])["risk_none"]

    if lang == "zh":
        sep = "；"
        return sep.join([zh_label(r) for r in flags])
    sep = "; "
    return sep.join(flags)


def _render_html(
    out_dir: str,
    final_score: float,
    subscores: Dict[str, float],
    metrics: Dict[str, float],
    keyframes: Dict[str, str],
    plots: Dict[str, str],
    diagnosis: List[Dict[str, str]],
    hand_meta: Dict[str, object],
    quality_flags: Dict[str, object],
    source_meta: Dict[str, object],
    biomech: Dict[str, object],
    phases: Dict[str, object],
    lang: str,
) -> str:
    t = I18N.get(lang, I18N["en"])
    def label(key: str) -> str:
        return zh_label(key) if lang == "zh" else str(key)

    def value_text(value: object) -> str:
        if lang != "zh":
            return str(value)
        if isinstance(value, bool):
            return zh_label("True" if value else "False")
        if isinstance(value, (int, float, np.floating)):
            return str(value)
        if value is None:
            return zh_label("n/a")
        return zh_label(str(value))

    knee_lo = round(LIT_KNEE_MEAN - LIT_KNEE_SD)
    knee_hi = round(LIT_KNEE_MEAN + LIT_KNEE_SD)
    trunk_lo = round(LIT_TRUNK_MEAN - LIT_TRUNK_SD)
    trunk_hi = round(LIT_TRUNK_MEAN + LIT_TRUNK_SD)
    calib_ranges = _load_calibration_ranges(os.path.dirname(__file__))

    impact_unit = "" if lang == "zh" else ""
    recommended_ranges = {
        "knee_flexion_est_deg": (knee_lo, knee_hi, "°", t["range_lit"]),
        "trunk_inclination_est_deg": (trunk_lo, trunk_hi, "°", t["range_lit"]),
        "impact_bh_ratio": (
            IMPACT_NORM_RANGE[0],
            IMPACT_NORM_RANGE[1],
            impact_unit,
            t["range_heur"],
        ),
        "timing_toss_to_trophy": (*TIMING_RANGES["timing_toss_to_trophy"], "s", t["range_heur"]),
        "timing_trophy_to_drop": (*TIMING_RANGES["timing_trophy_to_drop"], "s", t["range_heur"]),
        "timing_drop_to_contact": (*TIMING_RANGES["timing_drop_to_contact"], "s", t["range_heur"]),
    }
    for key, rng in calib_ranges.items():
        if key in recommended_ranges:
            unit = recommended_ranges[key][2]
        else:
            unit = impact_unit if key == "impact_height_norm" else "s"
        recommended_ranges[key] = (rng[0], rng[1], unit, t["range_calib"])

    def fmt(value: float) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return "n/a"
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.3f}"
        return str(value)

    def fmt_with_range(key: str, value: float, unit: str = "") -> str:
        if isinstance(value, (int, float, np.floating)) and key.startswith("timing_") and value < 0:
            return "无效（时间为负）" if lang == "zh" else "invalid (negative time)"
        base = fmt(value)
        rng = recommended_ranges.get(key)
        if rng is None or base == "n/a":
            return f"{base}{unit}"
        lo, hi, unit2, note = rng
        unit_final = unit2 if unit2 else unit
        return f"{base}{unit_final}（建议：{lo}–{hi}{unit_final}，{note}）"

    body_scale = metrics.get("body_scale", 1.0)
    if not isinstance(body_scale, (int, float)) or not np.isfinite(body_scale) or body_scale <= 0:
        body_scale = 1.0

    biomech_raw = dict(biomech.get("raw") or {})
    impact_raw = biomech_raw.get("impact_height")
    if not isinstance(impact_raw, (int, float)) or not np.isfinite(impact_raw):
        impact_raw = metrics.get("contact_above_head_raw")
        if not isinstance(impact_raw, (int, float)) or not np.isfinite(impact_raw):
            impact_raw = metrics.get("contact_above_head")
            if isinstance(impact_raw, (int, float)) and np.isfinite(impact_raw):
                impact_raw = float(impact_raw) * float(body_scale)
    if impact_raw is not None:
        biomech_raw["impact_height"] = float(impact_raw)

    impact_norm = biomech_raw.get("impact_height_norm")
    if not isinstance(impact_norm, (int, float)) or not np.isfinite(impact_norm):
        impact_norm = metrics.get("impact_height_norm", metrics.get("contact_above_head"))
    if impact_norm is not None:
        biomech_raw["impact_height_norm"] = float(impact_norm) if np.isfinite(impact_norm) else float("nan")

    metrics_display = {}
    for key in [
        "knee_flexion_est_deg",
        "trunk_inclination_est_deg",
        "impact_bh_ratio",
    ]:
        metrics_display[key] = metrics.get(key, float("nan"))
    for key in [
        "timing_toss_to_trophy",
        "timing_trophy_to_drop",
        "timing_drop_to_contact",
    ]:
        if key in metrics and np.isfinite(metrics.get(key, float("nan"))):
            metrics_display[key] = metrics[key]
    biomech_raw_display = {k: v for k, v in biomech_raw.items() if k != "impact_height_score_basis"}

    score_block = f"{final_score:.1f}"
    def _score_text(value: object) -> str:
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            return f"{float(value):.1f}"
        return "n/a"

    subs_rows = "\n".join(
        f"<tr><td>{label(k)}</td><td>{_score_text(v)}</td></tr>" for k, v in subscores.items()
    )

    short_note = (
        "该时段非常短，可能表示动作衔接紧凑；若画面遮挡较多或关键帧选择偏差，建议结合关键帧图复核。"
        if lang == "zh"
        else "This interval is very short; consider reviewing keyframes if occlusion or mis-detection is suspected."
    )

    def _zone_note(key: str, value: float) -> str:
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
            return ""
        if key == "knee_flexion_est_deg":
            mu, sd = LIT_KNEE_MEAN, LIT_KNEE_SD
            sub = subscores.get("knee_flexion")
        elif key == "trunk_inclination_est_deg":
            mu, sd = LIT_TRUNK_MEAN, LIT_TRUNK_SD
            sub = subscores.get("trunk_inclination")
        else:
            return ""
        if sd <= 1e-6:
            return ""
        z = abs((value - mu) / sd)
        if z <= 1.0:
            zone = "绿区（文献均值±1SD）"
        elif z <= 2.0:
            zone = "黄区（偏离1–2SD）"
        elif z <= 3.0:
            zone = "橙区（偏离2–3SD）"
        else:
            zone = "红区（偏离>3SD）"
        sub_text = f"{float(sub):.1f}" if isinstance(sub, (int, float, np.floating)) and np.isfinite(sub) else "n/a"
        if lang == "zh":
            return f"文献{zone}，z={z:.2f}，子分={sub_text}"
        return f"Literature zone {zone}, z={z:.2f}, subscore={sub_text}"

    def _timing_note(key: str, value: float) -> str:
        if not key.startswith("timing_"):
            return ""
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
            return ""
        if value < 0.02:
            return short_note
        return ""

    def _metric_note(key: str, value: float) -> str:
        note = _zone_note(key, value)
        timing_note = _timing_note(key, value)
        extra = ""
        if key == "impact_bh_ratio":
            sub = subscores.get("impact_height")
            sub_text = f"{float(sub):.1f}" if isinstance(sub, (int, float, np.floating)) and np.isfinite(sub) else "n/a"
            conf = biomech.get("confidence", {})
            conf_pct = conf.get("impact_confidence")
            conf_text = f"{float(conf_pct) * 100:.0f}%" if isinstance(conf_pct, (int, float, np.floating)) and np.isfinite(conf_pct) else "n/a"
            status = conf.get("impact_status")
            reason = conf.get("impact_reason", "")
            if status == "不可评估":
                return f"不可评估（{reason}）" if reason else "不可评估"
            if lang == "zh":
                extra = f"子分={sub_text}，置信度={conf_text}"
        if key.startswith("timing_"):
            sub = subscores.get("timing")
            sub_text = f"{float(sub):.1f}" if isinstance(sub, (int, float, np.floating)) and np.isfinite(sub) else "n/a"
            conf = biomech.get("confidence", {})
            conf_pct = conf.get("timing_confidence")
            conf_text = f"{float(conf_pct) * 100:.0f}%" if isinstance(conf_pct, (int, float, np.floating)) and np.isfinite(conf_pct) else "n/a"
            status = conf.get("timing_status")
            reason = conf.get("timing_cap_reason", "")
            if status == "不可评估":
                return f"不可评估（{reason}）" if reason else "不可评估"
            if lang == "zh":
                extra = f"子分={sub_text}，置信度={conf_text}"
        parts = [p for p in [note, timing_note, extra] if p]
        if not parts:
            return ""
        return "；".join(parts) if lang == "zh" else "; ".join(parts)

    def _metric_value(key: str, value: float) -> str:
        conf = biomech.get("confidence", {})
        if key == "impact_bh_ratio" and conf.get("impact_status") == "不可评估":
            return "不可评估"
        if key.startswith("timing_") and conf.get("timing_status") == "不可评估":
            return "不可评估"
        return fmt_with_range(key, value)

    metrics_rows = "\n".join(
        f"<tr><td>{label(k)}</td><td>{_metric_value(k, v)}</td><td>{_metric_note(k, v)}</td></tr>"
        for k, v in metrics_display.items()
    )
    z_scores = biomech.get("z_scores") or {}
    z_display = {
        k: v
        for k, v in z_scores.items()
        if k in {"knee_flexion", "trunk_inclination"} and np.isfinite(v)
    }
    z_rows = "\n".join(
        f"<tr><td>{label(k)}</td><td>{fmt(v)}</td></tr>" for k, v in z_display.items()
    )

    confidence = biomech.get("confidence") or {}
    def _pct(val: object) -> str:
        if not isinstance(val, (int, float, np.floating)) or not np.isfinite(val):
            return "n/a"
        return f"{float(val) * 100:.0f}%"

    def _count(valid_key: str, total_key: str) -> str:
        v = confidence.get(valid_key)
        t = confidence.get(total_key)
        if isinstance(v, (int, float)) and isinstance(t, (int, float)) and t:
            return f"{int(v)}/{int(t)}"
        return "n/a"

    if lang == "zh":
        conf_labels = {
            "knee": "膝屈",
            "trunk": "躯干倾角",
            "impact": "击球高度",
            "timing": "时序",
        }
        conf_headers = ["项目", "置信度", "有效帧/总帧", "备注"]
        timing_invalid_note = "存在无效时序" if confidence.get("timing_invalid") else ""
    else:
        conf_labels = {
            "knee": "Knee Flexion",
            "trunk": "Trunk Inclination",
            "impact": "Impact Height",
            "timing": "Timing",
        }
        conf_headers = ["Category", "Confidence", "Valid/Total", "Notes"]
        timing_invalid_note = "invalid timing detected" if confidence.get("timing_invalid") else ""

    confidence_rows = "\n".join(
        [
            f"<tr><td>{conf_labels['knee']}</td><td>{_pct(confidence.get('knee_confidence'))}</td>"
            f"<td>{_count('knee_valid_frames', 'knee_total_frames')}</td><td></td></tr>",
            f"<tr><td>{conf_labels['trunk']}</td><td>{_pct(confidence.get('trunk_confidence'))}</td>"
            f"<td>{_count('trunk_valid_frames', 'trunk_total_frames')}</td><td></td></tr>",
            f"<tr><td>{conf_labels['impact']}</td><td>{_pct(confidence.get('impact_confidence'))}</td>"
            f"<td>{_count('impact_valid_frames', 'impact_total_frames')}</td><td>{value_text(confidence.get('impact_source'))}</td></tr>",
            f"<tr><td>{conf_labels['timing']}</td><td>{_pct(confidence.get('timing_confidence'))}</td>"
            f"<td>{confidence.get('timing_valid_count', 'n/a')}/{confidence.get('timing_total_count', 'n/a')}</td><td>{timing_invalid_note}</td></tr>",
        ]
    )

    def _key_label(name: str) -> str:
        base = label(name)
        status = phases.get(f"{name}_status") if isinstance(phases, dict) else None
        if status == "fallback":
            return f"{base}（自动回退）" if lang == "zh" else f"{base} (fallback)"
        return base

    key_imgs = "\n".join(
        f"<div class='img-card'><div class='label'>{_key_label(name)}</div><img src='{path}'></div>"
        for name, path in keyframes.items()
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{t["title"]}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin: 0 0 8px 0; }}
    .score {{ font-size: 42px; font-weight: bold; color: #1b7837; }}
    .section {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .img-grid {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .img-card {{ width: 320px; }}
    .img-card img {{ width: 100%; border: 1px solid #ddd; }}
    .label {{ font-weight: bold; margin-bottom: 6px; }}
    .note {{ margin-top: 6px; color: #555; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>{t["title"]}</h1>
  <div class="score">{score_block}</div>
  <div>{t["final_score"]}</div>
  {f"<div>评分覆盖率：{(biomech.get('coverage') or 0.0) * 100:.0f}%</div>" if lang == "zh" else f"<div>Coverage: {(biomech.get('coverage') or 0.0) * 100:.0f}%</div>"}

  <div class="section">
    <h2>{t["hand_assignment"]}</h2>
    <table>
      <tr><th>{t["selected"]}</th><th>{t["mirror_suspect"]}</th><th>{t["guess"]}</th></tr>
      <tr>
        <td>{value_text(hand_meta.get("hand_mode_selected", "n/a"))}</td>
        <td>{value_text(hand_meta.get("mirror_suspect", "n/a"))}</td>
        <td>{value_text(hand_meta.get("hand_mode_guess", "n/a"))}</td>
      </tr>
    </table>
  </div>

  <div class="section">
    <h2>{t["data_source"]}</h2>
    <table>
      <tr><th>{t["metric"]}</th><th>{t["value"]}</th></tr>
      {"".join([f"<tr><td>{label(k)}</td><td>{value_text(v)}</td></tr>" for k, v in source_meta.items()])}
    </table>
  </div>

  <div class="section">
    <h2>{t["biomech"]}</h2>
    <table>
      <tr><th>{t["metric"]}</th><th>{t["value"]}</th><th>{t["notes"]}</th></tr>
      {metrics_rows}
    </table>
    {f"<table><tr><th>{t['z_scores']}</th><th>{t['value']}</th></tr>{z_rows}</table>" if z_rows else ""}
    <div class="note">
      {("计算依据：Jacquier-Bret & Gorce 2024 系统综述/元分析（奖杯位膝屈与躯干倾角均值±1SD）；击球高度参照 Mendes 2013；时序参考 Tubez 2019。"
        if lang == "zh"
        else "Basis: Jacquier-Bret & Gorce 2024 meta-analysis (trophy knee flexion & trunk inclination mean±1SD); impact height per Mendes 2013; timing per Tubez 2019.")}
    </div>
  </div>

  <div class="section">
    <h2>{t["subscores"]}</h2>
    <table>
      <tr><th>{t["category"]}</th><th>{t["score"]}</th></tr>
      {subs_rows}
    </table>
  </div>

  <div class="section">
    <h2>{t["confidence"]}</h2>
    <table>
      <tr><th>{conf_headers[0]}</th><th>{conf_headers[1]}</th><th>{conf_headers[2]}</th><th>{conf_headers[3]}</th></tr>
      {confidence_rows}
    </table>
  </div>

  <div class="section">
    <h2>{t["risk_flags"]}</h2>
    <table>
      <tr><th>{t["metric"]}</th><th>{t["value"]}</th></tr>
      <tr><td>{label("risk_flags")}</td><td>{_format_risk_flags(biomech.get("risk_flags") or [], lang)}</td></tr>
    </table>
  </div>

  <div class="section">
    <h2>{t["keyframes"]}</h2>
    <div class="img-grid">
      {key_imgs}
    </div>
    <div class="note">
      {("强制校正：奖杯位=原“拍头下落”帧；拍头下落在奖杯位之后重新搜索（确保 HRP→LRP→Impact）" if lang == "zh" else "Forced correction: HRP uses previous drop frame; LRP re-searched after HRP to enforce HRP→LRP→Impact.")}
    </div>
  </div>
</body>
</html>
"""
    return html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument(
        "--model",
        default=r"D:\Serve_Score\models\pose_landmarker_full.task",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pose_csv", default=None)
    ap.add_argument("--pose_source", choices=["raw", "smooth", "compare_zero"], default="compare_zero",
                    help="Data source selection for report.")
    ap.add_argument("--use_coords", choices=["raw", "smooth"], default="smooth",
                    help="Legacy: used when pose_source=raw/smooth.")
    ap.add_argument("--smooth_mode", choices=["zero_phase", "causal"], default="zero_phase",
                    help="Smoothing mode used when generating compare_zero.")
    ap.add_argument("--v_thr", type=float, default=0.6)
    ap.add_argument("--gap_max", type=int, default=3)
    ap.add_argument("--smooth_window", type=int, default=9)
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--quantile_calib", choices=["on", "off"], default="off",
                    help="Enable per-serve quantile calibration for impact/timing.")
    ap.add_argument("--hand_mode", choices=["auto", "A", "B"], default="auto",
                    help="Force hand assignment: A=left toss/right racket, B=right toss/left racket.")

    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()

    pose_csv = args.pose_csv
    pose_source_used = args.pose_source
    use_coords = args.use_coords

    if args.pose_source == "compare_zero":
        use_coords = "smooth"
        if not pose_csv:
            out_csv = os.path.join(args.out_dir, "pose_compare_zero.csv")
            out_video = os.path.join(args.out_dir, "pose_compare_zero.mp4")
            if not os.path.exists(out_csv):
                _run_extract(
                    args.video,
                    args.model,
                    out_csv,
                    out_video,
                    smooth_mode=args.smooth_mode,
                    v_thr=args.v_thr,
                    gap_max=args.gap_max,
                    smooth_window=args.smooth_window,
                )
            pose_csv = out_csv
    else:
        use_coords = args.pose_source
        if not pose_csv:
            out_csv = os.path.join(args.out_dir, "pose_compare_fast.csv")
            out_video = os.path.join(args.out_dir, "pose_compare_fast.mp4")
            _run_extract(
                args.video,
                args.model,
                out_csv,
                out_video,
                smooth_mode=args.smooth_mode,
                v_thr=args.v_thr,
                gap_max=args.gap_max,
                smooth_window=args.smooth_window,
            )
            pose_csv = out_csv

    df = pd.read_csv(pose_csv, encoding="utf-8-sig")
    result = serve_score.analyze_serve(
        df,
        use_coords=use_coords,
        hand_mode=args.hand_mode,
        v_thr=args.v_thr,
        gap_max=args.gap_max,
        quantile_calib=(args.quantile_calib == "on"),
    )

    metrics = dict(result["metrics"])
    subscores = result["subscores"]
    final_score = result["final_score"]
    biomech_full = result.get("biomech", {})
    biomech = {
        "subscores": biomech_full.get("subscores", {}),
        "final_score": biomech_full.get("final_score"),
        "metrics_min": biomech_full.get("metrics_min", {}),
        "confidence": biomech_full.get("confidence", {}),
        "risk_flags": biomech_full.get("risk_flags", []),
        "coverage": biomech_full.get("coverage"),
    }
    z_src = biomech_full.get("z_scores", {}) or {}
    biomech["z_scores"] = {
        k: v for k, v in z_src.items() if k in {"knee_flexion", "trunk_inclination"}
    }
    legacy_subscores = result.get("legacy_subscores")
    legacy_final = result.get("legacy_final_score")
    phases = result["phases"]
    signals = result["signals"]
    ts_ms = result["ts_ms"]

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    payload = {
        "final_score": final_score,
        "sub_scores": subscores,
        "metrics": metrics,
        "phases": phases,
        "biomech": biomech,
        "legacy_sub_scores": legacy_subscores,
        "legacy_final_score": legacy_final,
        "pose_source_used": pose_source_used,
        "fps": fps,
        "v_thr": args.v_thr,
        "gap_max": args.gap_max,
        "smooth_window": args.smooth_window,
        "smooth_mode": args.smooth_mode,
        "hand_mode_selected": result.get("hand_mode_selected"),
        "mirror_suspect": result.get("mirror_suspect"),
        "hand_mode_guess": result.get("hand_mode_guess"),
        "quality_flags": result.get("quality_flags"),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    debug_path = os.path.join(args.out_dir, "score_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(result.get("score_debug", {}), f, indent=2, ensure_ascii=False)

    plots_dir = os.path.join(args.out_dir, "plots")
    _ensure_dir(plots_dir)
    ts_s = ts_ms / 1000.0 if len(ts_ms) else np.arange(len(signals["left_wrist_y"])) / 30.0

    plots = {}
    plot_specs = [
        ("left_wrist_y", "Left Wrist Y", "left_wrist_y.png", "Y (norm)"),
        ("right_wrist_y", "Right Wrist Y", "right_wrist_y.png", "Y (norm)"),
        ("shoulder_tilt", "Shoulder Tilt (R - L)", "shoulder_tilt.png", "Y diff"),
        ("knee_angle", "Knee Angle", "knee_angle.png", "Degrees"),
        ("elbow_angle", "Elbow Angle", "elbow_angle.png", "Degrees"),
    ]
    for key, title, filename, ylabel in plot_specs:
        out_path = os.path.join(plots_dir, filename)
        _plot_series(ts_s, signals[key], title, ylabel, out_path, phases)
        plots[key] = _posix_path(os.path.join("plots", filename))

    run_id = int(time.time() * 1000)
    keyframes = _save_keyframes(args.video, phases, args.out_dir)
    keyframes = {k: f"{v}?v={run_id}" for k, v in keyframes.items()}
    diagnosis = _diagnosis_biomech(subscores, biomech, args.lang)
    hand_meta = {
        "hand_mode_selected": result.get("hand_mode_selected"),
        "mirror_suspect": result.get("mirror_suspect"),
        "hand_mode_guess": result.get("hand_mode_guess"),
    }
    source_meta = {
        "pose_source_used": pose_source_used,
        "fps": fps,
        "v_thr": args.v_thr,
        "gap_max": args.gap_max,
        "smooth_window": args.smooth_window,
        "smooth_mode": args.smooth_mode,
        "quantile_calib": args.quantile_calib,
    }
    html = _render_html(
        args.out_dir,
        final_score,
        subscores,
        metrics,
        keyframes,
        plots,
        diagnosis,
        hand_meta,
        result.get("quality_flags", {}),
        source_meta,
        biomech,
        phases,
        args.lang,
    )

    html_path = os.path.join(args.out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    pdf_path = os.path.join(args.out_dir, "report.pdf")
    if _try_export_pdf(html_path, pdf_path):
        print(f"[OK] report.pdf: {pdf_path}")
    else:
        print("[INFO] PDF export skipped (optional dependency missing).")

    print(f"[OK] metrics.json: {metrics_path}")
    print(f"[OK] report.html : {html_path}")
    print(f"[OK] score_debug.json: {debug_path}")
    print("[INFO] pose_source_used:", pose_source_used)
    print("[INFO] fps:", fps)
    print("[INFO] v_thr:", args.v_thr, "gap_max:", args.gap_max, "smooth_window:", args.smooth_window,
          "smooth_mode:", args.smooth_mode)
    print("[INFO] coord_system_used:", metrics.get("coord_system_used"))
    missing_count = result.get("score_debug", {}).get("missing_metrics_count")
    print("[INFO] body_scale:", metrics.get("body_scale"), "missing_metrics_count:", missing_count)
    print("[DEBUG] phases:",
          "trophy", phases.get("trophy"),
          "drop", phases.get("racket_drop"),
          "contact", phases.get("contact"),
          "finish", phases.get("finish"))


if __name__ == "__main__":
    main()
