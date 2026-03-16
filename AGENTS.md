# Serve_Score (Tennis Serve Scoring) — Agent Instructions

## Goal
Given a user-uploaded *serve clip* (short mp4), generate a scoring report:
- outputs: JSON metrics + HTML report (+ optional PDF), annotated mp4, keyframes
- right-handed serve by default

## Current Assets (do not break)
- extract_pose_compare.py
  - already produces pose_compare_fast.csv and pose_compare_fast.mp4
  - pose csv is long-format: columns [frame, ts_ms, lm, raw_x, raw_y, smooth_x, smooth_y, vis]
- Environment: conda env name is `servepose` on Windows

## Hard Constraints
- Keep Windows compatibility (paths, encoding utf-8-sig for csv)
- No breaking changes to existing CLI flags in existing scripts
- All new scripts must be runnable from project root with `python <script>.py ...`
- Outputs must go under `D:\Serve_Score\out\...` (configurable via CLI)
- Prefer lightweight deps: numpy, pandas, opencv-python, matplotlib (avoid heavy report deps unless optional)

## New Deliverable (Phase: scoring report, not auto-cut)
Implement:
1) serve_score.py (library): compute phase events + metrics + final score (0-100)
2) make_report.py (CLI):
   - input: --video (serve clip mp4)
   - optional: --pose_csv (if provided, skip extraction)
   - if no pose_csv: call extract_pose_compare.py to generate pose csv + annotated pose video
   - output folder: --out_dir
   - outputs:
     - metrics.json
     - report.html (embeds plots + keyframes)
     - keyframes/*.png (trophy/contact/finish)
     - optional report.pdf if PDF dependency is available; otherwise skip gracefully

## Scoring (right-handed)
Use pose landmarks to estimate:
- Toss quality: left wrist peak height, smoothness, time to peak
- Trophy position: elbow angle (R shoulder-elbow-wrist), elbow height vs shoulder
- Knee bend: min knee angle / hip drop during loading
- Shoulder tilt at trophy: L/R shoulder y diff
- Racket drop depth: R wrist low point vs R shoulder
- Contact: R wrist near-highest + above head; contact height
- Timing: phase durations (toss->trophy->drop->contact)

Provide:
- raw numeric metrics
- sub-scores per category
- weighted final score

## Acceptance Criteria
- Running on a single serve clip produces all outputs without manual editing
- Report shows: final score, sub-scores, plots of key curves, and keyframes
- Code is well-commented and parameterized for future WeChat cloud function integration
