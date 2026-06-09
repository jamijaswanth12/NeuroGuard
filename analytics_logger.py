"""
analytics_logger.py — NeuroGuard Metadata Logger
=================================================
Logs anonymized prediction metadata to a local SQLite database.
NO patient-identifiable data is stored. Only model performance metrics.

Fields logged:
  - timestamp, modality, prediction, confidence, severity
  - per-step timing (preprocess, inference, gradcam, xai)
  - heatmap coverage percentage
"""

import os
import sqlite3
import datetime
import json
from typing import Optional

# ── DB path (next to this script) ─────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuroguard_logs.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    modality         TEXT    NOT NULL,
    prediction       TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    severity         REAL    NOT NULL,
    preprocess_ms    INTEGER DEFAULT 0,
    inference_ms     INTEGER DEFAULT 0,
    gradcam_ms       INTEGER DEFAULT 0,
    xai_ms           INTEGER DEFAULT 0,
    total_ms         INTEGER DEFAULT 0,
    heatmap_coverage REAL    DEFAULT 0.0
);
"""


def _get_conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_SQL)
    conn.commit()
    return conn


def log_prediction(
    modality: str,
    prediction: str,
    confidence: float,
    severity: float,
    timing: dict,
    heatmap_coverage: float = 0.0,
):
    """
    Log a single prediction event. All arguments are model outputs — no patient data.

    timing dict keys (all in milliseconds):
        preprocess_ms, inference_ms, gradcam_ms, xai_ms, total_ms
    """
    try:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        t = timing or {}
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (timestamp, modality, prediction, confidence, severity,
                    preprocess_ms, inference_ms, gradcam_ms, xai_ms, total_ms,
                    heatmap_coverage)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    now,
                    modality,
                    prediction,
                    round(confidence, 4),
                    round(severity, 2),
                    t.get("preprocess_ms", 0),
                    t.get("inference_ms", 0),
                    t.get("gradcam_ms", 0),
                    t.get("xai_ms", 0),
                    t.get("total_ms", 0),
                    round(heatmap_coverage, 2),
                ),
            )
        return True
    except Exception as e:
        print(f"[analytics_logger] Log failed: {e}")
        return False


def get_summary() -> dict:
    """Return aggregate stats for the analytics dashboard."""
    try:
        with _get_conn() as conn:
            rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 500").fetchall()
        if not rows:
            return {"total": 0, "rows": []}

        records = [dict(r) for r in rows]
        total = len(records)

        # Modality split
        ct_count  = sum(1 for r in records if r["modality"] == "CT")
        mri_count = sum(1 for r in records if r["modality"] == "MRI")

        # Prediction distribution
        pred_dist = {"Hemorrhagic": 0, "Ischemic": 0, "Normal": 0}
        for r in records:
            pred_dist[r["prediction"]] = pred_dist.get(r["prediction"], 0) + 1

        # Avg confidence per class
        conf_by_class = {}
        for r in records:
            p = r["prediction"]
            conf_by_class.setdefault(p, []).append(r["confidence"])
        avg_conf = {k: round(sum(v) / len(v) * 100, 1) for k, v in conf_by_class.items()}

        # Avg timing (only non-zero rows)
        timed_rows = [r for r in records if r["total_ms"] > 0]
        if timed_rows:
            avg_pre    = round(sum(r["preprocess_ms"] for r in timed_rows) / len(timed_rows))
            avg_inf    = round(sum(r["inference_ms"]  for r in timed_rows) / len(timed_rows))
            avg_gcam   = round(sum(r["gradcam_ms"]    for r in timed_rows) / len(timed_rows))
            avg_xai    = round(sum(r["xai_ms"]        for r in timed_rows) / len(timed_rows))
            avg_total  = round(sum(r["total_ms"]       for r in timed_rows) / len(timed_rows))
        else:
            avg_pre = avg_inf = avg_gcam = avg_xai = avg_total = 0

        # Recent 20 for timeline
        recent = records[:20]

        return {
            "total":        total,
            "ct_count":     ct_count,
            "mri_count":    mri_count,
            "pred_dist":    pred_dist,
            "avg_conf":     avg_conf,
            "avg_timing": {
                "preprocess_ms": avg_pre,
                "inference_ms":  avg_inf,
                "gradcam_ms":    avg_gcam,
                "xai_ms":        avg_xai,
                "total_ms":      avg_total,
            },
            "recent":       recent,
            "rows":         records,
        }
    except Exception as e:
        print(f"[analytics_logger] Summary failed: {e}")
        return {"total": 0, "rows": []}


def build_charts(summary: dict):
    """
    Build matplotlib figures for the analytics dashboard.
    Returns (fig_dist, fig_timing) or (None, None) on error.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        PALETTE = {
            "Hemorrhagic": "#ef4444",
            "Ischemic":    "#f97316",
            "Normal":      "#22c55e",
        }
        BG   = "#0f172a"
        CARD = "#1e293b"
        TXT  = "#e2e8f0"
        MUTED = "#94a3b8"

        # ── Figure 1: Prediction Distribution ────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(6, 3.5), facecolor=BG)
        ax1.set_facecolor(CARD)

        pred_dist = summary.get("pred_dist", {})
        labels = [k for k in ("Hemorrhagic", "Ischemic", "Normal") if pred_dist.get(k, 0) > 0]
        values = [pred_dist.get(k, 0) for k in labels]
        colors = [PALETTE[k] for k in labels]

        if values:
            bars = ax1.bar(labels, values, color=colors, width=0.5, zorder=3)
            ax1.bar_label(bars, padding=4, color=TXT, fontsize=11, fontweight="bold")
        ax1.set_title("Prediction Distribution", color=TXT, fontsize=13, fontweight="bold", pad=12)
        ax1.set_ylabel("Count", color=MUTED, fontsize=10)
        ax1.tick_params(colors=TXT, labelsize=10)
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.spines[["left", "bottom"]].set_color("#334155")
        ax1.yaxis.label.set_color(MUTED)
        ax1.set_ylim(0, max(values or [1]) * 1.25)
        ax1.grid(axis="y", color="#334155", linestyle="--", alpha=0.5, zorder=0)
        plt.tight_layout(pad=1.5)

        # ── Figure 2: Avg Inference Timing Breakdown ──────────────────────────
        fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor=BG)
        ax2.set_facecolor(CARD)

        timing = summary.get("avg_timing", {})
        t_labels = ["Preprocess", "Inference", "Grad-CAM++", "XAI Gen."]
        t_keys   = ["preprocess_ms", "inference_ms", "gradcam_ms", "xai_ms"]
        t_colors = ["#38bdf8", "#818cf8", "#fb923c", "#34d399"]
        t_values = [timing.get(k, 0) for k in t_keys]

        if any(t_values):
            bars2 = ax2.barh(t_labels, t_values, color=t_colors, height=0.5, zorder=3)
            ax2.bar_label(bars2, fmt="%d ms", padding=4, color=TXT, fontsize=10)
        ax2.set_title("Avg. Inference Timing Breakdown", color=TXT, fontsize=13, fontweight="bold", pad=12)
        ax2.set_xlabel("Milliseconds (ms)", color=MUTED, fontsize=10)
        ax2.tick_params(colors=TXT, labelsize=10)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.spines[["left", "bottom"]].set_color("#334155")
        ax2.set_xlim(0, max(t_values or [1]) * 1.3)
        ax2.grid(axis="x", color="#334155", linestyle="--", alpha=0.5, zorder=0)
        plt.tight_layout(pad=1.5)

        return fig1, fig2

    except Exception as e:
        print(f"[analytics_logger] Chart build failed: {e}")
        return None, None
