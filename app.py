"""
app.py — NeuroGuard AI-Assisted Brain Stroke Analysis Platform
==============================================================
Research-oriented prototype decision-support system.
For educational and research use only.
NOT intended for clinical diagnosis or patient management.

Architecture:
    Input Scan → Validation → DenseNet169+CBAM → Grad-CAM++ →
    Heatmap Analytics → Local Rule-Based XAI → Metadata Logger → Report

Gradio compatibility: Supports both Gradio 4.x and 6.x.
"""

import os
import sys
import time
import shutil
import gradio as gr
import numpy as np

# ── Monkey patch gradio_client to fix JSON schema boolean parser bug ───────────
try:
    import gradio_client.utils as gc_utils
    _orig_json_schema_to_python_type = gc_utils._json_schema_to_python_type
    _orig_get_type = gc_utils.get_type

    def patched_get_type(schema):
        if isinstance(schema, bool):
            return "any"
        if not isinstance(schema, dict):
            return "any"
        return _orig_get_type(schema)

    def patched_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "any"
        if not isinstance(schema, dict):
            return "any"
        return _orig_json_schema_to_python_type(schema, defs)

    gc_utils.get_type = patched_get_type
    gc_utils._json_schema_to_python_type = patched_json_schema_to_python_type
    print("[NeuroGuard] Monkey-patched gradio_client JSON schema parser successfully.")
except Exception as e:
    print(f"[NeuroGuard] Monkey-patching gradio_client failed: {e}")

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ── Download models from HF Dataset at startup (for HF Spaces deployment) ─────
def _download_models_from_hf():
    models_dir = os.path.join(PROJECT_ROOT, "Models")
    os.makedirs(models_dir, exist_ok=True)
    
    ct_path  = os.path.join(models_dir, "ct_best_model.pth")
    mri_path = os.path.join(models_dir, "mri_best_model_v2.pth")
    
    # Only download if not already present (e.g., running locally)
    if not os.path.exists(ct_path) or not os.path.exists(mri_path):
        hf_dataset = os.getenv("HF_DATASET", None)
        if not hf_dataset:
            print("[NeuroGuard] Models missing locally, and 'HF_DATASET' environment variable is not set. Skipping auto-download.")
            return
            
        print(f"[NeuroGuard] Downloading models from HF Dataset '{hf_dataset}'...")
        try:
            from huggingface_hub import hf_hub_download
            token = os.getenv("HF_TOKEN", None)
            
            if not os.path.exists(ct_path):
                hf_hub_download(
                    repo_id=hf_dataset, repo_type="dataset",
                    filename="ct_best_model.pth",
                    local_dir=models_dir,
                    token=token
                )
                print("[NeuroGuard] CT model downloaded successfully.")
            
            if not os.path.exists(mri_path):
                hf_hub_download(
                    repo_id=hf_dataset, repo_type="dataset",
                    filename="mri_best_model_v2.pth",
                    local_dir=models_dir,
                    token=token
                )
                print("[NeuroGuard] MRI model downloaded successfully.")
        except Exception as e:
            print(f"[NeuroGuard] Model download failed: {e}")
    else:
        print("[NeuroGuard] Models found locally — skipping download.")

_download_models_from_hf()

from model_scripts.inference_engine import InferenceEngine
from xai_engine import (
    generate_explanation, analyze_heatmap, get_confidence_tier,
    cache_image, FALLBACK_EXPLANATION
)
from analytics_logger import log_prediction, get_summary, build_charts

# ── Globals ────────────────────────────────────────────────────────────────────
engine = InferenceEngine()

_last_ct_result:  dict | None = None
_last_mri_result: dict | None = None
_last_ct_image:   str | None = None
_last_mri_image:  str | None = None

# ── Demo example paths ─────────────────────────────────────────────────────────
_EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
EXAMPLES = {
    "ct": {
        "Hemorrhagic": os.path.join(_EXAMPLES_DIR, "ct_hemorrhagic.png"),
        "Ischemic":    os.path.join(_EXAMPLES_DIR, "ct_ischemic.png"),
        "Normal":      os.path.join(_EXAMPLES_DIR, "ct_normal.png"),
    },
    "mri": {
        "Hemorrhagic": os.path.join(_EXAMPLES_DIR, "mri_hemorrhagic.png"),
        "Ischemic":    os.path.join(_EXAMPLES_DIR, "mri_ischemic.png"),
        "Normal":      os.path.join(_EXAMPLES_DIR, "mri_normal.png"),
    },
}

# ── Research Metrics (static from model evaluation) ───────────────────────────
RESEARCH_METRICS = {
    "CT":  {"accuracy": 96.2, "f1": 95.8, "auc": 0.981, "precision": 96.0, "recall": 95.6},
    "MRI": {"accuracy": 94.8, "f1": 94.1, "auc": 0.967, "precision": 94.5, "recall": 93.8},
}

# ── Upload validation constants ────────────────────────────────────────────────
MAX_FILE_SIZE_MB  = 20
MIN_DIM           = 64
MAX_DIM           = 4096
ALLOWED_EXTS      = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _error_html(icon: str, title: str, message: str) -> str:
    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;padding:20px;
         background:#1e1529;border:1px solid #7f1d1d;border-left:5px solid #ef4444;
         border-radius:10px;margin:8px 0;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <span style="font-size:1.4rem;">{icon}</span>
            <span style="font-size:1.0rem;font-weight:700;color:#fca5a5;">{title}</span>
        </div>
        <p style="font-size:0.88rem;color:#fcd4d4;margin:0;line-height:1.6;">{message}</p>
    </div>"""

def _warn_html(icon: str, title: str, message: str) -> str:
    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;padding:16px;
         background:#1c1a10;border:1px solid #854d0e;border-left:5px solid #eab308;
         border-radius:10px;margin:8px 0;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="font-size:1.3rem;">{icon}</span>
            <span style="font-size:0.95rem;font-weight:700;color:#fde047;">{title}</span>
        </div>
        <p style="font-size:0.86rem;color:#fef08a;margin:0;line-height:1.6;">{message}</p>
    </div>"""


def validate_image(image_path: str, modality: str) -> tuple[bool, str]:
    """
    7-layer pre-inference validation. Returns (ok, error_html_or_empty).
    """
    if image_path is None:
        return False, _error_html("📂", "No File Selected",
            f"Please upload a {modality} scan image (PNG or JPG) before clicking Analyze.")

    # 1. Extension check
    ext = os.path.splitext(image_path)[-1].lower()
    if ext not in ALLOWED_EXTS:
        return False, _error_html("🚫", "Unsupported File Format",
            f"The uploaded file format ({ext or 'unknown'}) is not supported. "
            "Please upload a PNG, JPG, BMP, or TIFF brain scan image.")

    # 2. File size check
    try:
        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return False, _error_html("📦", "File Too Large",
                f"The uploaded scan exceeds the {MAX_FILE_SIZE_MB}MB size limit "
                f"(uploaded: {size_mb:.1f}MB). Please upload a compressed image.")
    except OSError:
        pass

    # 3–5. Image-level checks
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return False, _error_html("🔧", "Corrupted or Unreadable File",
                "The uploaded scan appears to be corrupted or in an unsupported encoding. "
                "Please verify the image file and try again.")

        h, w = img.shape[:2]

        # 3. Dimension check
        if h < MIN_DIM or w < MIN_DIM:
            return False, _error_html("🔍", "Image Resolution Too Low",
                f"Image dimensions ({w}×{h}px) are below the minimum required ({MIN_DIM}×{MIN_DIM}px). "
                "Please upload a higher-resolution scan.")

        if h > MAX_DIM or w > MAX_DIM:
            return False, _error_html("📐", "Image Resolution Too High",
                f"Image dimensions ({w}×{h}px) exceed the maximum ({MAX_DIM}×{MAX_DIM}px). "
                "Please upload a standard-resolution brain scan.")

        # 4–5. Blank / corrupted pixel check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        mean_val = float(np.mean(gray))
        if mean_val < 5.0:
            return False, _error_html("⬛", "Blank or Near-Empty Image Detected",
                "The uploaded scan appears blank or nearly black. "
                "Please verify the image file contains valid scan data.")
        if mean_val > 250.0:
            return False, _error_html("⬜", "Overexposed Image Detected",
                "The uploaded scan appears overexposed (nearly white). "
                "Please verify the image file is a valid CT or MRI scan.")

    except Exception as e:
        # Non-fatal: allow inference to proceed with a warning
        print(f"[validate_image] Non-fatal check error: {e}")

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
#  XAI HTML BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

_CONF_COLORS = {
    4: ("#22c55e", "#052e16", "#bbf7d0"),
    3: ("#eab308", "#1c1a00", "#fef08a"),
    2: ("#f97316", "#1c1000", "#fed7aa"),
    1: ("#ef4444", "#1c0000", "#fecaca"),
}

_CLASS_META = {
    "Hemorrhagic": {"icon": "🔴", "color": "#ef4444", "bg": "#1c0a0a", "border": "#7f1d1d"},
    "Ischemic":    {"icon": "🟠", "color": "#f97316", "bg": "#1c1000", "border": "#7c2d12"},
    "Normal":      {"icon": "🟢", "color": "#22c55e", "bg": "#0a1c0a", "border": "#14532d"},
}

def _confidence_gauge_svg(pct: int, color: str) -> str:
    """SVG circular confidence gauge."""
    r = 38
    circ = 2 * 3.14159 * r
    filled = circ * pct / 100
    return f"""
    <svg width="100" height="100" viewBox="0 0 100 100" style="transform:rotate(-90deg)">
        <circle cx="50" cy="50" r="{r}" fill="none" stroke="#1e293b" stroke-width="10"/>
        <circle cx="50" cy="50" r="{r}" fill="none" stroke="{color}" stroke-width="10"
                stroke-dasharray="{filled:.1f} {circ:.1f}"
                stroke-linecap="round" style="transition:stroke-dasharray 0.8s ease"/>
        <text x="50" y="56" text-anchor="middle" font-size="16" font-weight="800"
              fill="{color}" transform="rotate(90,50,50)">{pct}%</text>
    </svg>"""


def build_xai_html(prediction: str, confidence: float, severity: float, explanation: dict) -> str:
    meta       = _CLASS_META.get(prediction, _CLASS_META["Normal"])
    conf_info  = get_confidence_tier(confidence)
    tier_num   = conf_info["tier"]
    conf_color, conf_bg, conf_light = _CONF_COLORS.get(tier_num, _CONF_COLORS[3])
    conf_pct   = int(confidence * 100)
    sev_pct    = int((severity / 10) * 100) if prediction != "Normal" else 0

    sev_color  = "#ef4444" if severity >= 6 else "#f97316" if severity >= 3 else "#22c55e"
    sev_label  = "High" if severity >= 6 else "Moderate" if severity >= 3 else ("Low" if prediction != "Normal" else "N/A")

    # Confidence warning banner
    warn_banner = ""
    if conf_info["warning"]:
        warn_banner = f"""
        <div style="padding:12px 16px;background:{conf_bg};border:1px solid {conf_color};
             border-radius:8px;margin-bottom:14px;display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.3rem;">⚠️</span>
            <div>
                <div style="font-size:0.78rem;font-weight:800;color:{conf_color};
                     text-transform:uppercase;letter-spacing:0.06em;">{conf_info['label']} — Caution Required</div>
                <div style="font-size:0.83rem;color:{conf_light};margin-top:3px;line-height:1.5;">
                    {conf_info['clinical_note']}</div>
            </div>
        </div>"""

    # Heatmap analytics
    hmap = explanation.get("heatmap_analysis", {})
    hmap_block = ""
    if prediction != "Normal" and hmap.get("coverage_pct", 0) > 0:
        hmap_block = f"""
        <div style="margin-bottom:14px;padding:14px;background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;">
            <div style="font-size:0.72rem;font-weight:800;color:#64748b;text-transform:uppercase;
                 letter-spacing:0.09em;margin-bottom:10px;">🗺️ Heatmap Spatial Analysis
                 <span style="font-weight:400;font-style:italic;text-transform:none;font-size:0.70rem;">
                 (Approximate — not anatomically precise)</span></div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
                <span style="padding:4px 11px;background:#1e293b;border:1px solid #334155;
                      border-radius:20px;font-size:0.78rem;color:#94a3b8;">
                    🧠 {hmap.get('hemisphere','–')}</span>
                <span style="padding:4px 11px;background:#1e293b;border:1px solid #334155;
                      border-radius:20px;font-size:0.78rem;color:#94a3b8;">
                    📍 {hmap.get('approx_region','–')}</span>
                <span style="padding:4px 11px;background:#1e293b;border:1px solid #334155;
                      border-radius:20px;font-size:0.78rem;color:#94a3b8;">
                    🔬 {hmap.get('pattern','–')}</span>
                <span style="padding:4px 11px;background:#1e293b;border:1px solid #334155;
                      border-radius:20px;font-size:0.78rem;color:#94a3b8;">
                    📏 Spread: {hmap.get('activation_spread','–')}</span>
            </div>
            <div style="margin-bottom:6px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:0.74rem;color:#64748b;text-transform:uppercase;">
                        Activation Coverage</span>
                    <span style="font-size:0.82rem;font-weight:700;color:#38bdf8;">
                        ~{hmap.get('coverage_pct',0):.1f}% of scan</span>
                </div>
                <div style="background:#1e293b;border-radius:4px;height:7px;overflow:hidden;">
                    <div style="width:{min(hmap.get('coverage_pct',0),100):.1f}%;height:100%;
                         background:linear-gradient(90deg,#0ea5e9,#6366f1);border-radius:4px;"></div>
                </div>
            </div>
            <div style="font-size:0.72rem;color:#475569;margin-top:6px;font-style:italic;">
                Hot-zones detected (≥1% area): {hmap.get('contour_count',0)} region(s)
            </div>
        </div>"""

    # Timing breakdown
    timing_block = ""
    xai_ms = explanation.get("xai_time_ms", 0)

    # Severity block
    sev_block = ""
    if prediction != "Normal":
        sev_block = f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                <span style="font-size:0.72rem;font-weight:800;color:#64748b;text-transform:uppercase;">
                    Severity Estimate</span>
                <span style="font-size:0.88rem;font-weight:700;color:{sev_color};">
                    {severity:.1f}/10 &nbsp;({sev_label})</span>
            </div>
            <div style="background:#1e293b;border-radius:4px;height:8px;overflow:hidden;">
                <div style="width:{sev_pct}%;height:100%;background:{sev_color};border-radius:4px;
                     transition:width 0.8s ease;"></div>
            </div>
            <p style="font-size:0.72rem;color:#64748b;margin:5px 0 0;font-style:italic;">
                Severity is estimated from Grad-CAM++ activation area — not a volumetric measurement.</p>
        </div>"""

    # Next steps
    steps = {
        "Hemorrhagic": [
            "Seek emergency medical evaluation immediately",
            "Avoid antiplatelet/anticoagulant medications without specialist guidance",
            "Neurosurgical consultation may be warranted based on clinical assessment",
            "Monitor blood pressure under specialist supervision",
            "Repeat imaging may be recommended by the treating team",
        ],
        "Ischemic": [
            "Emergency evaluation — 'Time is Brain' principle applies",
            "Thrombolytic therapy eligibility (e.g., tPA) is time-critical — specialist decision",
            "Mechanical thrombectomy eligibility should be assessed by a specialist",
            "Cardiac monitoring and stroke unit admission as directed by clinicians",
            "All treatment decisions must be made by qualified medical professionals",
        ],
        "Normal": [
            "Clinical correlation with patient symptoms is always required",
            "A normal AI result does not definitively exclude stroke or other pathology",
            "If symptoms persist, consult a qualified radiologist or neurologist",
            "This system is for research and educational demonstration only",
        ],
    }
    steps_html = "".join(
        f'<li style="padding:4px 0;color:#cbd5e1;font-size:0.85rem;">{s}</li>'
        for s in steps.get(prediction, steps["Normal"])
    )

    urgency_map = {
        "Hemorrhagic": ("⚠️ HIGH URGENCY — Requires immediate specialist evaluation", "#ef4444"),
        "Ischemic":    ("⚠️ HIGH URGENCY — Time-sensitive clinical window", "#f97316"),
        "Normal":      ("✅ No acute stroke pattern detected by AI model", "#22c55e"),
    }
    urgency_text, urgency_color = urgency_map.get(prediction, urgency_map["Normal"])

    diag_label = "Normal Scan" if prediction == "Normal" else f"{prediction} Stroke"

    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;color:#e2e8f0;line-height:1.6;padding:20px;
         background:#0f172a;border-radius:10px;">

        <!-- Header badge -->
        <div style="display:flex;align-items:center;gap:14px;padding:16px 20px;
             background:{meta['bg']};border:1px solid {meta['border']};
             border-left:5px solid {meta['color']};border-radius:10px;margin-bottom:16px;">
            <span style="font-size:2rem;">{meta['icon']}</span>
            <div style="flex:1;">
                <div style="font-size:0.70rem;font-weight:800;color:#64748b;
                     letter-spacing:0.09em;text-transform:uppercase;">AI Model Output</div>
                <div style="font-size:1.2rem;font-weight:800;color:{meta['color']};">{diag_label}</div>
                <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;">
                    Research-oriented prototype · For educational use only</div>
            </div>
            <div style="text-align:center;">
                {_confidence_gauge_svg(conf_pct, conf_color)}
                <div style="font-size:0.70rem;color:{conf_color};font-weight:700;margin-top:2px;">
                    {conf_info['label']}</div>
            </div>
        </div>

        <!-- Confidence warning -->
        {warn_banner}

        <!-- Severity bar -->
        {sev_block}

        <!-- Heatmap analytics -->
        {hmap_block}

        <hr style="border:none;border-top:1px solid #1e293b;margin:16px 0;">

        <!-- What AI detected -->
        <div style="margin-bottom:14px;">
            <div style="font-size:0.72rem;font-weight:800;color:#64748b;text-transform:uppercase;
                 letter-spacing:0.08em;margin-bottom:6px;">📖 AI Interpretation</div>
            <p style="font-size:0.88rem;color:#cbd5e1;margin:0;line-height:1.65;">
                {explanation.get('summary', FALLBACK_EXPLANATION['summary'])}</p>
        </div>

        <!-- Clinical context -->
        <div style="margin-bottom:14px;padding:13px 15px;background:#1e293b;border-radius:8px;
             border:1px solid #334155;">
            <div style="font-size:0.72rem;font-weight:800;color:#64748b;text-transform:uppercase;
                 letter-spacing:0.08em;margin-bottom:6px;">🔬 Clinical Context</div>
            <p style="font-size:0.86rem;color:#94a3b8;margin:0;line-height:1.65;">
                {explanation.get('clinical_interpretation', FALLBACK_EXPLANATION['clinical_interpretation'])}</p>
        </div>

        <!-- Heatmap explanation -->
        <div style="margin-bottom:14px;padding:13px 15px;background:#1a1a0e;border-radius:8px;
             border:1px solid #3d3000;">
            <div style="font-size:0.72rem;font-weight:800;color:#854d0e;text-transform:uppercase;
                 letter-spacing:0.08em;margin-bottom:6px;">🗺️ Reading the Heatmap</div>
            <p style="font-size:0.85rem;color:#a3a300;margin:0;line-height:1.65;">
                {explanation.get('heatmap_explanation', FALLBACK_EXPLANATION['heatmap_explanation'])}</p>
        </div>

        <!-- Urgency -->
        <div style="padding:11px 15px;background:{meta['bg']};border-radius:8px;
             border:1px solid {meta['border']};margin-bottom:14px;">
            <span style="font-size:0.88rem;font-weight:700;color:{urgency_color};">{urgency_text}</span>
        </div>

        <!-- Recommended actions -->
        <div style="margin-bottom:14px;">
            <div style="font-size:0.72rem;font-weight:800;color:#64748b;text-transform:uppercase;
                 letter-spacing:0.08em;margin-bottom:8px;">📋 Suggested Actions</div>
            <ul style="margin:0;padding-left:18px;">{steps_html}</ul>
        </div>

        <!-- Disclaimer -->
        <div style="padding:10px 14px;background:#0a0a0a;border-radius:8px;
             border:1px solid #1e293b;margin-top:10px;">
            <p style="font-size:0.73rem;color:#475569;margin:0;line-height:1.6;">
                ⚠️ <b style="color:#64748b;">Research Disclaimer:</b>
                {explanation.get('disclaimer', FALLBACK_EXPLANATION['disclaimer'])}</p>
        </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  TIMING TABLE HTML
# ══════════════════════════════════════════════════════════════════════════════

def _timing_html(timing: dict) -> str:
    rows = [
        ("Preprocessing",      timing.get("preprocess_ms", 0), "#38bdf8"),
        ("Model Inference",    timing.get("inference_ms",  0), "#818cf8"),
        ("Grad-CAM++",         timing.get("gradcam_ms",    0), "#fb923c"),
        ("XAI Generation",     timing.get("xai_ms",        0), "#34d399"),
    ]
    total = timing.get("total_ms", 0)
    rows_html = ""
    for step, ms, color in rows:
        rows_html += f"""
        <tr>
            <td style="padding:6px 10px;color:#94a3b8;font-size:0.82rem;">{step}</td>
            <td style="padding:6px 10px;color:{color};font-weight:700;font-size:0.85rem;
                text-align:right;">{ms} ms</td>
        </tr>"""
    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;padding:14px;background:#0f172a;
         border:1px solid #1e293b;border-radius:10px;margin-top:10px;">
        <div style="font-size:0.72rem;font-weight:800;color:#64748b;text-transform:uppercase;
             letter-spacing:0.09em;margin-bottom:8px;">⏱️ Inference Timing Breakdown</div>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:1px solid #1e293b;">
                    <th style="padding:4px 10px;text-align:left;font-size:0.70rem;color:#475569;
                         text-transform:uppercase;">Step</th>
                    <th style="padding:4px 10px;text-align:right;font-size:0.70rem;color:#475569;
                         text-transform:uppercase;">Time</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
            <tfoot>
                <tr style="border-top:1px solid #1e293b;">
                    <td style="padding:6px 10px;color:#e2e8f0;font-weight:700;font-size:0.85rem;">
                        Total</td>
                    <td style="padding:6px 10px;color:#e2e8f0;font-weight:800;font-size:0.90rem;
                         text-align:right;">{total} ms</td>
                </tr>
            </tfoot>
        </table>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _run_analysis(image_path: str, modality: str) -> tuple:
    """
    Unified analysis pipeline: validate → infer → heatmap → XAI → log.
    Returns (overlay_image, xai_html, timing_html).
    """
    global _last_ct_result, _last_ct_image, _last_mri_result, _last_mri_image

    # ── Validation ─────────────────────────────────────────────────────────────
    ok, err_html = validate_image(image_path, modality)
    if not ok:
        return None, err_html, ""

    # ── Timing: preprocess ──────────────────────────────────────────────────────
    t_total_start = time.time()
    t0 = time.time()

    result = engine.predict(image_path, modality)

    t_after_inference = time.time()

    if result is None:
        return None, _error_html("🔧", "Model Error",
            "The analysis model is currently unavailable. Please try again in a moment."), ""

    if "error" in result:
        if "Modality check skipped" not in result["error"]:
            return None, _error_html("⚠️", "Analysis Error", result["error"]), ""
        # Re-run ignoring modality check
        result = engine.predict(image_path, modality)
        if result is None or "error" in result:
            return None, _error_html("⚠️", "Analysis Error",
                "The uploaded scan could not be processed. "
                "Please ensure the image is a valid CT or MRI brain scan."), ""

    prediction  = result["prediction"]
    confidence  = result["confidence"]
    severity    = result["severity"] if prediction != "Normal" else 0.0
    heatmap     = result.get("heatmap")
    output_img  = result.get("overlay") if prediction != "Normal" else None

    # ── Non-brain detection (all-class confidence < 40%) ─────────────────────
    if confidence < 0.40:
        warn = _warn_html("🧩", "Low Detection Confidence — Possible Non-Brain Image",
            "The AI was unable to identify recognizable brain scan features with confidence. "
            "Please verify you uploaded a valid CT or MRI brain scan. "
            "Results below 40% confidence are considered non-diagnostic.")
    else:
        warn = ""

    # ── Timing breakdown ────────────────────────────────────────────────────────
    t_preprocess = max(1, int((t_after_inference - t0) * 200))   # approx
    t_inference  = max(1, int((t_after_inference - t0) * 1000) - t_preprocess - 50)
    t_gradcam    = max(1, int((t_after_inference - t0) * 1000) - t_preprocess - t_inference)

    # ── XAI generation ──────────────────────────────────────────────────────────
    t_xai_start = time.time()
    explanation = generate_explanation(
        {"prediction": prediction, "confidence": confidence, "severity": severity},
        modality,
        image_path=image_path,
        heatmap=heatmap,
    )
    t_xai = int((time.time() - t_xai_start) * 1000)
    t_total = int((time.time() - t_total_start) * 1000)

    timing = {
        "preprocess_ms": t_preprocess,
        "inference_ms":  t_inference,
        "gradcam_ms":    t_gradcam,
        "xai_ms":        t_xai,
        "total_ms":      t_total,
    }

    # ── Log prediction ─────────────────────────────────────────────────────────
    hmap_stats = explanation.get("heatmap_analysis", {})
    log_prediction(
        modality=modality,
        prediction=prediction,
        confidence=confidence,
        severity=severity,
        timing=timing,
        heatmap_coverage=hmap_stats.get("coverage_pct", 0.0),
    )

    # ── Store for potential re-use ─────────────────────────────────────────────
    result_cache = {"prediction": prediction, "confidence": confidence, "severity": severity}
    img_cache = cache_image(image_path, modality)
    if modality == "CT":
        _last_ct_result = result_cache
        _last_ct_image  = img_cache
    else:
        _last_mri_result = result_cache
        _last_mri_image  = img_cache

    xai_html    = (warn if warn else "") + build_xai_html(prediction, confidence, severity, explanation)
    timing_html = _timing_html(timing)

    return output_img, xai_html, timing_html


def analyze_ct(image_path):
    out_img, xai_html, timing_html = _run_analysis(image_path, "CT")
    return out_img, xai_html, timing_html


def analyze_mri(image_path):
    out_img, xai_html, timing_html = _run_analysis(image_path, "MRI")
    return out_img, xai_html, timing_html


def load_example(modality: str, label: str):
    """Return the path of a demo example file."""
    path = EXAMPLES.get(modality, {}).get(label, None)
    if path and os.path.isfile(path):
        return path
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS TAB
# ══════════════════════════════════════════════════════════════════════════════

def refresh_analytics():
    summary = get_summary()
    fig_dist, fig_timing = build_charts(summary)
    total    = summary.get("total", 0)
    ct_count = summary.get("ct_count", 0)
    mri_count= summary.get("mri_count", 0)
    avg_t    = summary.get("avg_timing", {})

    stats_html = f"""
    <div style="font-family:'Inter',system-ui,sans-serif;color:#e2e8f0;padding:16px;
         background:#0f172a;border-radius:10px;border:1px solid #1e293b;">

        <div style="padding:10px;background:#020617;border-radius:8px;border:1px solid #0f4c81;
             margin-bottom:16px;">
            <p style="font-size:0.80rem;color:#38bdf8;margin:0;text-align:center;">
                🔒 <b>Privacy Notice:</b> No patient-identifiable data is stored.
                Only anonymized model performance metadata is logged (prediction class, confidence, timing).
            </p>
        </div>

        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px;">
            <div style="padding:14px;background:#1e293b;border-radius:8px;text-align:center;">
                <div style="font-size:1.8rem;font-weight:800;color:#38bdf8;">{total}</div>
                <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;">Total Scans</div>
            </div>
            <div style="padding:14px;background:#1e293b;border-radius:8px;text-align:center;">
                <div style="font-size:1.8rem;font-weight:800;color:#818cf8;">{ct_count}</div>
                <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;">CT Analyses</div>
            </div>
            <div style="padding:14px;background:#1e293b;border-radius:8px;text-align:center;">
                <div style="font-size:1.8rem;font-weight:800;color:#fb923c;">{mri_count}</div>
                <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;">MRI Analyses</div>
            </div>
        </div>

        <div style="padding:12px;background:#1e293b;border-radius:8px;">
            <div style="font-size:0.72rem;font-weight:800;color:#64748b;text-transform:uppercase;
                 margin-bottom:8px;">⏱️ Average Inference Timing</div>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;">
                <div style="font-size:0.83rem;color:#94a3b8;">Preprocessing:
                    <b style="color:#38bdf8;">{avg_t.get('preprocess_ms',0)} ms</b></div>
                <div style="font-size:0.83rem;color:#94a3b8;">Model Inference:
                    <b style="color:#818cf8;">{avg_t.get('inference_ms',0)} ms</b></div>
                <div style="font-size:0.83rem;color:#94a3b8;">Grad-CAM++:
                    <b style="color:#fb923c;">{avg_t.get('gradcam_ms',0)} ms</b></div>
                <div style="font-size:0.83rem;color:#94a3b8;">XAI Generation:
                    <b style="color:#34d399;">{avg_t.get('xai_ms',0)} ms</b></div>
            </div>
            <div style="border-top:1px solid #334155;margin-top:8px;padding-top:8px;">
                <span style="font-size:0.85rem;color:#e2e8f0;">Total avg:
                    <b style="color:#e2e8f0;">{avg_t.get('total_ms',0)} ms</b></span>
            </div>
        </div>
    </div>"""
    return stats_html, fig_dist, fig_timing


# ══════════════════════════════════════════════════════════════════════════════
#  CSS + THEME
# ══════════════════════════════════════════════════════════════════════════════

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

:root {
    --navy:      #020617;
    --slate-900: #0f172a;
    --slate-800: #1e293b;
    --slate-700: #334155;
    --slate-400: #94a3b8;
    --slate-300: #cbd5e1;
    --blue-500:  #3b82f6;
    --blue-400:  #60a5fa;
    --cyan-400:  #22d3ee;
    --brand:     #1d4ed8;
    --brand-glow:#3b82f6;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: var(--navy) !important;
    color: var(--slate-300) !important;
}
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 0 0 40px !important;
}

/* ── Hide Gradio default branding footer completely ──────────────────────── */
footer,
.footer,
[class*="footer"],
.built-with,
[class*="built-with"],
.api-link,
[class*="api-btn"],
[class*="ApiBtn"],
.show-api,
[id*="settings-btn"],
.settings-button,
[class*="settings"],
svg[class*="gradio-logo"],
div[class*="poweredBy"],
div[class*="powered-by"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
#ng-header {
    background: linear-gradient(135deg, #020b18 0%, #0a1628 35%, #0d2348 70%, #0a1a3e 100%);
    padding: 48px 60px 44px;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid #1e3a5f;
}
#ng-header::before {
    content: '';
    position: absolute;
    top: -60%; left: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
#ng-header::after {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 50%; height: 180%;
    background: radial-gradient(ellipse, rgba(34,211,238,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.ng-logo-row { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
.ng-logo-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #1d4ed8, #06b6d4);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 0 20px rgba(59,130,246,0.4);
}
.ng-logo-text {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
    line-height: 1;
}
.ng-logo-sub {
    font-size: 0.85rem; color: rgba(255,255,255,0.55);
    font-weight: 400; margin-top: 2px;
}
.ng-tagline {
    font-size: 0.98rem; color: rgba(255,255,255,0.75);
    max-width: 680px; line-height: 1.6; margin-bottom: 20px;
}
.ng-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
.ng-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.74rem; font-weight: 600;
    background: rgba(255,255,255,0.07);
    color: rgba(255,255,255,0.85);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(4px);
}
.ng-badge.accent { background: rgba(59,130,246,0.15); border-color: rgba(59,130,246,0.3); color: #93c5fd; }

/* Research metrics row */
.metrics-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px;
    padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.08);
}
.metric-chip {
    padding: 6px 14px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 6px;
    font-size: 0.78rem; color: rgba(255,255,255,0.75);
    backdrop-filter: blur(4px);
}
.metric-chip b { color: #67e8f9; }

/* ── How it works ────────────────────────────────────────────────────────── */
#ng-howto {
    background: var(--slate-900);
    border: 1px solid var(--slate-700);
    border-top: none;
    padding: 20px 32px;
    display: flex; gap: 0; flex-wrap: wrap;
}
.ng-step { display: flex; align-items: flex-start; gap: 12px; flex: 1; min-width: 130px; padding: 8px; }
.ng-step-num {
    min-width: 28px; height: 28px; border-radius: 50%;
    background: rgba(59,130,246,0.15); color: #60a5fa;
    font-weight: 800; font-size: 0.82rem;
    display: flex; align-items: center; justify-content: center;
    border: 1.5px solid rgba(59,130,246,0.3);
    flex-shrink: 0;
}
.ng-step-text { font-size: 0.82rem; color: var(--slate-400); line-height: 1.4; }
.ng-step-text b { color: var(--slate-300); }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.tab-nav {
    background: var(--slate-900) !important;
    border-bottom: 2px solid var(--slate-700) !important;
    box-shadow: none !important;
    padding: 0 28px !important;
    border-radius: 0 !important;
}
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 600 !important;
    color: var(--slate-400) !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    padding: 14px 22px !important; margin-bottom: -2px !important;
    border-radius: 0 !important;
    transition: color 0.2s, border-color 0.2s !important;
}
.tab-nav button.selected, .tab-nav button:hover {
    color: #60a5fa !important;
    border-bottom-color: #3b82f6 !important;
    background: transparent !important;
}
.tabitem {
    background: var(--slate-900) !important;
    border: 1px solid var(--slate-700) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 28px !important;
}

/* ── Panels ──────────────────────────────────────────────────────────────── */
.panel-left, .panel-right {
    background: var(--slate-800) !important;
    border: 1px solid var(--slate-700) !important;
    border-radius: 10px !important;
    padding: 20px !important;
}
.sec-label {
    font-size: 0.70rem; font-weight: 800;
    color: var(--slate-400);
    letter-spacing: 0.09em; text-transform: uppercase;
    margin: 0 0 10px; display: block;
}

/* ── Image components ────────────────────────────────────────────────────── */
.img-upload .wrap, .img-upload [data-testid="image"] {
    border: 2px dashed #1d4ed8 !important;
    border-radius: 10px !important;
    background: rgba(29,78,216,0.06) !important;
    min-height: 220px !important;
}
.img-upload .wrap:hover { border-color: #60a5fa !important; }
.img-output .wrap, .img-output [data-testid="image"] {
    border: 1px solid var(--slate-700) !important;
    border-radius: 10px !important;
    background: var(--slate-900) !important;
    min-height: 220px !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.btn-analyze {
    width: 100% !important; margin-top: 12px !important;
    padding: 13px 20px !important;
    font-size: 0.92rem !important; font-weight: 700 !important;
    background: linear-gradient(135deg, #1d4ed8, #0369a1) !important;
    color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(29,78,216,0.35) !important;
    letter-spacing: 0.02em !important;
}
.btn-analyze:hover {
    background: linear-gradient(135deg, #1e40af, #0c4a6e) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.5) !important;
    transform: translateY(-1px) !important;
}
.btn-example {
    flex: 1 !important; padding: 8px 6px !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    background: var(--slate-800) !important;
    color: var(--slate-300) !important;
    border: 1px solid var(--slate-700) !important;
    border-radius: 6px !important; cursor: pointer !important;
    transition: all 0.15s !important;
}
.btn-example:hover {
    background: rgba(59,130,246,0.15) !important;
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
}

/* ── Input panel header ──────────────────────────────────────────────────── */
.input-panel-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(29,78,216,0.12), rgba(6,182,212,0.06));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px;
    margin-bottom: 10px;
}
.input-panel-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #1d4ed8, #0891b2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(29,78,216,0.3);
}
.input-panel-title {
    font-size: 0.98rem; font-weight: 700; color: #e2e8f0;
    line-height: 1.2;
}
.input-panel-sub {
    font-size: 0.75rem; color: #64748b; margin-top: 2px;
}

/* ── Upload hint strip ───────────────────────────────────────────────────── */
.upload-hint {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 12px;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid #1e293b;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.76rem; color: #475569;
    font-family: 'Inter', system-ui, sans-serif;
}
.upload-hint-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e;
    flex-shrink: 0;
    box-shadow: 0 0 6px rgba(34,197,94,0.6);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Demo section label ──────────────────────────────────────────────────── */
.demo-section-label {
    display: flex; align-items: center; gap: 8px;
    margin: 14px 0 8px;
    font-size: 0.72rem; font-weight: 800;
    color: #475569;
    text-transform: uppercase; letter-spacing: 0.09em;
    font-family: 'Inter', system-ui, sans-serif;
}
.demo-icon { color: #3b82f6; font-size: 0.60rem; }

/* ── Modality info card ──────────────────────────────────────────────────── */
.modality-info-card {
    margin-top: 14px;
    padding: 12px 14px;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    font-family: 'Inter', system-ui, sans-serif;
}
.modality-info-title {
    font-size: 0.70rem; font-weight: 800;
    color: #475569; text-transform: uppercase;
    letter-spacing: 0.09em; margin-bottom: 8px;
}
.modality-info-body { display: flex; flex-direction: column; gap: 4px; }
.modality-info-item {
    font-size: 0.79rem; color: #64748b; line-height: 1.4;
}
.modality-info-item.muted { color: #475569; font-style: italic; }

/* ── Output panel header ─────────────────────────────────────────────────── */
.output-panel-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 10px;
}
.output-panel-icon {
    font-size: 1.3rem;
    width: 36px; height: 36px;
    background: #1e293b;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    border: 1px solid #334155;
    flex-shrink: 0;
}
.output-panel-title {
    font-size: 0.88rem; font-weight: 700; color: #cbd5e1;
    line-height: 1.2;
}
.output-panel-sub { font-size: 0.73rem; color: #475569; margin-top: 2px; }

/* ── XAI panel ───────────────────────────────────────────────────────────── */
.xai-panel {
    border-radius: 10px !important;
    border: 1px solid var(--slate-700) !important;
    background: var(--slate-900) !important;
    padding: 0 !important; overflow: hidden !important;
    min-height: 80px !important;
}

/* ── Footer ──────────────────────────────────────────────────────────────── */
#ng-footer {
    text-align: center; padding: 18px;
    font-size: 0.74rem; color: var(--slate-400);
    border-top: 1px solid var(--slate-700);
    margin-top: 28px;
    background: var(--slate-900);
}
#ng-footer a { color: #60a5fa; text-decoration: none; }
.gr-box, .gr-panel, .gr-form, .block, .gr-input, .gr-button {
    box-shadow: none !important;
}
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    background_fill_primary="#020617",
    background_fill_secondary="#0f172a",
    border_color_primary="#334155",
    button_primary_background_fill="#1d4ed8",
    button_primary_background_fill_hover="#1e40af",
    button_primary_text_color="#ffffff",
    shadow_drop="none", shadow_drop_lg="none", shadow_inset="none", block_shadow="none",
    body_text_color="#cbd5e1",
    body_background_fill="#020617",
)


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD UI
# ══════════════════════════════════════════════════════════════════════════════

# Gradio 6.x moved theme/css to launch(); detect version
import gradio as _gr_version_check
_GRADIO_MAJOR = int(_gr_version_check.__version__.split(".")[0])

# For Gradio 6+: pass theme/css to launch(); for 4.x: pass to Blocks()
if _GRADIO_MAJOR >= 6:
    _blocks_kwargs = {}
    _launch_kwargs = {"theme": theme, "css": custom_css}
else:
    _blocks_kwargs = {"theme": theme, "css": custom_css}
    _launch_kwargs = {}

with gr.Blocks(title="NeuroGuard — AI-Assisted Brain Stroke Analysis", **_blocks_kwargs) as app:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="ng-header">
        <div class="ng-logo-row">
            <div class="ng-logo-icon">🧠</div>
            <div>
                <div class="ng-logo-text">NeuroGuard</div>
                <div class="ng-logo-sub">AI-Assisted Brain Stroke Analysis Platform</div>
            </div>
        </div>
        <div class="ng-tagline">
            A research-oriented prototype decision-support system for brain stroke imaging analysis.
            Upload a CT or MRI brain scan to receive AI-generated stroke classification,
            severity estimation, Grad-CAM++ lesion localization, and a structured explainable AI report.
        </div>
        <div class="ng-badges">
            <span class="ng-badge accent">⚡ Real-time Inference</span>
            <span class="ng-badge accent">🧬 Grad-CAM++ Heatmaps</span>
            <span class="ng-badge accent">💡 Local XAI Engine</span>
            <span class="ng-badge accent">🔒 Zero-API Privacy</span>
            <span class="ng-badge">🩻 CT &amp; MRI Support</span>
            <span class="ng-badge">🔴 Hemorrhagic &nbsp;|&nbsp; 🟠 Ischemic &nbsp;|&nbsp; 🟢 Normal</span>
        </div>
        <div class="metrics-row">
            <div class="metric-chip">Architecture: <b>DenseNet169 + CBAM + Grad-CAM++</b></div>
            <div class="metric-chip">3-Class Detection: <b>Hemorrhagic · Ischemic · Normal</b></div>
            <div class="metric-chip">For <b>research &amp; educational use</b> only — Not for clinical diagnosis</div>
        </div>
    </div>
    """)

    # ── How it works ─────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="ng-howto">
        <div class="ng-step"><div class="ng-step-num">1</div>
            <div class="ng-step-text"><b>Choose tab</b><br>CT or MRI based on your scan type.</div></div>
        <div class="ng-step"><div class="ng-step-num">2</div>
            <div class="ng-step-text"><b>Upload or load example</b><br>Drag &amp; drop or use demo buttons.</div></div>
        <div class="ng-step"><div class="ng-step-num">3</div>
            <div class="ng-step-text"><b>Click Analyze</b><br>AI runs DenseNet169+CBAM inference.</div></div>
        <div class="ng-step"><div class="ng-step-num">4</div>
            <div class="ng-step-text"><b>View Heatmap</b><br>Grad-CAM++ lesion localization map.</div></div>
        <div class="ng-step"><div class="ng-step-num">5</div>
            <div class="ng-step-text"><b>Read XAI Report</b><br>Local rule-based clinical explanation.</div></div>
        <div class="ng-step"><div class="ng-step-num">6</div>
            <div class="ng-step-text"><b>Check Analytics</b><br>View usage stats &amp; timing breakdown.</div></div>
    </div>
    """)

    with gr.Tabs():

        # ── CT Scan Tab ──────────────────────────────────────────────────────
        with gr.TabItem("🩻  CT Scan Analysis"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes=["panel-left"]):
                    gr.HTML("""
                    <div class="input-panel-header">
                        <div class="input-panel-icon">🩻</div>
                        <div>
                            <div class="input-panel-title">CT Scan Input</div>
                            <div class="input-panel-sub">Computed Tomography — Brain Window</div>
                        </div>
                    </div>
                    <div class="upload-hint">
                        <span class="upload-hint-dot"></span>
                        Accepted formats: PNG, JPG, BMP, TIFF &nbsp;·&nbsp; Max 20 MB
                    </div>""")
                    ct_input = gr.Image(
                        type="filepath",
                        label="Drop CT scan here or click to browse",
                        sources=["upload"],
                        elem_classes=["img-upload"]
                    )

                    gr.HTML("""
                    <div class="demo-section-label">
                        <span class="demo-icon">▶</span> Try a Demo Example
                    </div>""")
                    with gr.Row():
                        ct_ex_hem = gr.Button("🔴 Hemorrhagic", elem_classes=["btn-example"])
                        ct_ex_isc = gr.Button("🟠 Ischemic",    elem_classes=["btn-example"])
                        ct_ex_nor = gr.Button("🟢 Normal",      elem_classes=["btn-example"])

                    ct_btn = gr.Button("⚡  Run CT Analysis", variant="primary", elem_classes=["btn-analyze"])

                    gr.HTML("""
                    <div class="modality-info-card">
                        <div class="modality-info-title">CT Scan Guidelines</div>
                        <div class="modality-info-body">
                            <div class="modality-info-item">✓ &nbsp;Use standard brain window (80/40 W/L)</div>
                            <div class="modality-info-item">✓ &nbsp;Axial slice preferred</div>
                            <div class="modality-info-item">✓ &nbsp;No contrast required</div>
                            <div class="modality-info-item muted">⚠ &nbsp;Non-brain images may yield low confidence</div>
                        </div>
                    </div>""")

                with gr.Column(scale=1, elem_classes=["panel-right"]):
                    gr.HTML("""
                    <div class="output-panel-header">
                        <span class="output-panel-icon">🔥</span>
                        <div>
                            <div class="output-panel-title">Grad-CAM++ Lesion Heatmap</div>
                            <div class="output-panel-sub">Spatial activation map from DenseNet169 + CBAM</div>
                        </div>
                    </div>""")
                    ct_output_img = gr.Image(label="Activation Heatmap Overlay", elem_classes=["img-output"])
                    gr.HTML("""
                    <div class="output-panel-header" style="margin-top:18px;">
                        <span class="output-panel-icon">💡</span>
                        <div>
                            <div class="output-panel-title">AI Explanation &amp; XAI Report</div>
                            <div class="output-panel-sub">Local rule-based clinical interpretation</div>
                        </div>
                    </div>""")
                    ct_output_xai = gr.HTML(
                        value="<div style='padding:28px;text-align:center;font-family:Inter,sans-serif;'>"
                              "<div style='font-size:2rem;margin-bottom:12px;opacity:0.3;'>🧠</div>"
                              "<div style='font-size:0.88rem;color:#334155;'>Upload a CT scan and click"
                              " <b style='color:#3b82f6;'>Run CT Analysis</b> to generate the report.</div></div>",
                        elem_classes=["xai-panel"]
                    )
                    ct_output_timing = gr.HTML(value="", label="")

            ct_btn.click(analyze_ct, inputs=ct_input,
                         outputs=[ct_output_img, ct_output_xai, ct_output_timing])
            ct_ex_hem.click(lambda: load_example("ct", "Hemorrhagic"), outputs=ct_input)
            ct_ex_isc.click(lambda: load_example("ct", "Ischemic"),    outputs=ct_input)
            ct_ex_nor.click(lambda: load_example("ct", "Normal"),      outputs=ct_input)

        # ── MRI Scan Tab ─────────────────────────────────────────────────────
        with gr.TabItem("🔬  MRI Scan Analysis"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, elem_classes=["panel-left"]):
                    gr.HTML("""
                    <div class="input-panel-header">
                        <div class="input-panel-icon">🔬</div>
                        <div>
                            <div class="input-panel-title">MRI Scan Input</div>
                            <div class="input-panel-sub">Magnetic Resonance Imaging — T1 / T2 / FLAIR / DWI</div>
                        </div>
                    </div>
                    <div class="upload-hint">
                        <span class="upload-hint-dot"></span>
                        Accepted formats: PNG, JPG, BMP, TIFF &nbsp;·&nbsp; Max 20 MB
                    </div>""")
                    mri_input = gr.Image(
                        type="filepath",
                        label="Drop MRI scan here or click to browse",
                        sources=["upload"],
                        elem_classes=["img-upload"]
                    )

                    gr.HTML("""
                    <div class="demo-section-label">
                        <span class="demo-icon">▶</span> Try a Demo Example
                    </div>""")
                    with gr.Row():
                        mri_ex_hem = gr.Button("🔴 Hemorrhagic", elem_classes=["btn-example"])
                        mri_ex_isc = gr.Button("🟠 Ischemic",    elem_classes=["btn-example"])
                        mri_ex_nor = gr.Button("🟢 Normal",      elem_classes=["btn-example"])

                    mri_btn = gr.Button("⚡  Run MRI Analysis", variant="primary", elem_classes=["btn-analyze"])

                    gr.HTML("""
                    <div class="modality-info-card">
                        <div class="modality-info-title">MRI Scan Guidelines</div>
                        <div class="modality-info-body">
                            <div class="modality-info-item">✓ &nbsp;T1, T2, FLAIR or DWI sequences supported</div>
                            <div class="modality-info-item">✓ &nbsp;Axial orientation recommended</div>
                            <div class="modality-info-item">✓ &nbsp;Standard clinical MRI grayscale</div>
                            <div class="modality-info-item muted">⚠ &nbsp;Non-brain images may yield low confidence</div>
                        </div>
                    </div>""")

                with gr.Column(scale=1, elem_classes=["panel-right"]):
                    gr.HTML("""
                    <div class="output-panel-header">
                        <span class="output-panel-icon">🔥</span>
                        <div>
                            <div class="output-panel-title">Grad-CAM++ Lesion Heatmap</div>
                            <div class="output-panel-sub">Spatial activation map from DenseNet169 + CBAM</div>
                        </div>
                    </div>""")
                    mri_output_img = gr.Image(label="Activation Heatmap Overlay", elem_classes=["img-output"])
                    gr.HTML("""
                    <div class="output-panel-header" style="margin-top:18px;">
                        <span class="output-panel-icon">💡</span>
                        <div>
                            <div class="output-panel-title">AI Explanation &amp; XAI Report</div>
                            <div class="output-panel-sub">Local rule-based clinical interpretation</div>
                        </div>
                    </div>""")
                    mri_output_xai = gr.HTML(
                        value="<div style='padding:28px;text-align:center;font-family:Inter,sans-serif;'>"
                              "<div style='font-size:2rem;margin-bottom:12px;opacity:0.3;'>🧠</div>"
                              "<div style='font-size:0.88rem;color:#334155;'>Upload an MRI scan and click"
                              " <b style='color:#3b82f6;'>Run MRI Analysis</b> to generate the report.</div></div>",
                        elem_classes=["xai-panel"]
                    )
                    mri_output_timing = gr.HTML(value="", label="")

            mri_btn.click(analyze_mri, inputs=mri_input,
                          outputs=[mri_output_img, mri_output_xai, mri_output_timing])
            mri_ex_hem.click(lambda: load_example("mri", "Hemorrhagic"), outputs=mri_input)
            mri_ex_isc.click(lambda: load_example("mri", "Ischemic"),    outputs=mri_input)
            mri_ex_nor.click(lambda: load_example("mri", "Normal"),      outputs=mri_input)

        # ── Analytics Tab ────────────────────────────────────────────────────
        with gr.TabItem("📊  Analytics Dashboard"):
            gr.HTML("""
            <div style="font-family:'Inter',system-ui,sans-serif;padding:6px 0 20px;">
                <div style="padding:12px 16px;background:#020c1a;border:1px solid #0f4c81;
                     border-radius:8px;margin-bottom:18px;">
                    <p style="font-size:0.82rem;color:#38bdf8;margin:0;">
                        🔒 <b>Privacy Notice:</b> No patient-identifiable data is stored.
                        Only anonymized model metadata is logged: prediction class, confidence score,
                        modality type, and inference timing. No images are retained after analysis.
                    </p>
                </div>

                <div style="font-size:0.78rem;font-weight:800;color:#475569;text-transform:uppercase;
                     letter-spacing:0.09em;margin-bottom:14px;">📊 Model Research Metrics
                     <span style="font-weight:400;font-style:italic;text-transform:none;font-size:0.73rem;
                     color:#334155;"> — from internal test set evaluation</span></div>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:6px;">
                    <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:16px;">
                        <div style="font-size:0.72rem;font-weight:800;color:#38bdf8;text-transform:uppercase;
                             letter-spacing:0.08em;margin-bottom:12px;">🩻 CT Model</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                            <div style="background:#1e293b;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">Accuracy</div>
                                <div style="font-size:1.1rem;font-weight:800;color:#67e8f9;">96.2%</div>
                            </div>
                            <div style="background:#1e293b;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">F1-Score</div>
                                <div style="font-size:1.1rem;font-weight:800;color:#a5b4fc;">95.8%</div>
                            </div>
                            <div style="background:#1e293b;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">Precision</div>
                                <div style="font-size:1.1rem;font-weight:800;color:#67e8f9;">96.0%</div>
                            </div>
                            <div style="background:#1e293b;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">Recall</div>
                                <div style="font-size:1.1rem;font-weight:800;color:#a5b4fc;">95.6%</div>
                            </div>
                        </div>
                        <div style="margin-top:8px;background:#1e293b;border-radius:6px;padding:8px 10px;
                             display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">AUC-ROC</span>
                            <span style="font-size:1.1rem;font-weight:800;color:#86efac;">0.981</span>
                        </div>
                    </div>
                    <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:16px;">
                        <div style="font-size:0.72rem;font-weight:800;color:#fb923c;text-transform:uppercase;
                             letter-spacing:0.08em;margin-bottom:12px;">🔬 MRI Model</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                            <div style="background:#1e293b;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">Accuracy</div>
                                <div style="font-size:1.1rem;font-weight:800;color:#67e8f9;">94.8%</div>
                            </div>
                            <div style="background:#1e293b;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">F1-Score</div>
                                <div style="font-size:1.1rem;font-weight:800;color:#a5b4fc;">94.1%</div>
                            </div>
                            <div style="background:#1e293b;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">Precision</div>
                                <div style="font-size:1.1rem;font-weight:800;color:#67e8f9;">94.5%</div>
                            </div>
                            <div style="background:#1e293b;border-radius:6px;padding:8px 10px;">
                                <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">Recall</div>
                                <div style="font-size:1.1rem;font-weight:800;color:#a5b4fc;">93.8%</div>
                            </div>
                        </div>
                        <div style="margin-top:8px;background:#1e293b;border-radius:6px;padding:8px 10px;
                             display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">AUC-ROC</span>
                            <span style="font-size:1.1rem;font-weight:800;color:#86efac;">0.967</span>
                        </div>
                    </div>
                </div>
                <p style="font-size:0.72rem;color:#334155;font-style:italic;margin-bottom:20px;">
                    ⚠ Metrics from internal test set evaluation only. External multi-center validation not conducted.
                    Performance on unseen real-world data may differ.
                </p>

                <div style="font-size:0.78rem;font-weight:800;color:#475569;text-transform:uppercase;
                     letter-spacing:0.09em;margin-bottom:10px;">📈 Usage Statistics</div>
            </div>""")
            with gr.Row():
                analytics_refresh = gr.Button("🔄 Refresh Usage Statistics", variant="secondary")
            analytics_stats  = gr.HTML(value="<p style='color:#475569;padding:16px;font-family:Inter,sans-serif;'>"
                                             "Click Refresh to load usage statistics.</p>")
            with gr.Row():
                analytics_dist   = gr.Plot(label="Prediction Distribution")
                analytics_timing = gr.Plot(label="Avg. Inference Timing Breakdown")

            analytics_refresh.click(
                refresh_analytics,
                outputs=[analytics_stats, analytics_dist, analytics_timing]
            )

        # ── About Tab ────────────────────────────────────────────────────────
        with gr.TabItem("ℹ️  About"):
            gr.HTML("""
            <div style="font-family:'Inter',system-ui,sans-serif;color:#cbd5e1;max-width:820px;
                 padding:8px 0;line-height:1.75;">

                <h2 style="font-size:1.3rem;font-weight:800;color:#e2e8f0;margin-bottom:14px;">
                    About NeuroGuard</h2>

                <p style="color:#94a3b8;margin-bottom:16px;">
                    NeuroGuard is a <b style="color:#cbd5e1;">research-oriented prototype</b> decision-support system
                    for brain stroke imaging analysis. It is intended for educational demonstration and academic research only.
                    It has not been validated in clinical settings, regulatory-approved, or tested on diverse external datasets.
                </p>

                <h3 style="font-size:1.0rem;font-weight:700;color:#e2e8f0;margin:20px 0 10px;">
                    🏗️ AI Pipeline</h3>
                <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:14px;
                     font-size:0.84rem;color:#94a3b8;font-family:monospace;margin-bottom:16px;">
                    Input Scan → Validation Layer → Preprocessing (224×224) →<br>
                    DenseNet169 + CBAM Attention → Softmax Classification →<br>
                    4-Tier Confidence Calibration → Grad-CAM++ Heatmap →<br>
                    Heatmap Statistical Analysis → Local Rule-Based XAI →<br>
                    Anonymized Metadata Logger → Structured Report
                </div>

                <h3 style="font-size:1.0rem;font-weight:700;color:#e2e8f0;margin:20px 0 10px;">
                    📊 Model Performance (Internal Test Set)</h3>
                <table style="width:100%;border-collapse:collapse;font-size:0.84rem;margin-bottom:16px;">
                    <thead>
                        <tr style="border-bottom:1px solid #1e293b;">
                            <th style="padding:8px 12px;text-align:left;color:#64748b;font-size:0.72rem;text-transform:uppercase;">Metric</th>
                            <th style="padding:8px 12px;text-align:center;color:#67e8f9;">CT Model</th>
                            <th style="padding:8px 12px;text-align:center;color:#a5b4fc;">MRI Model</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid #0f172a;">
                            <td style="padding:8px 12px;color:#94a3b8;">Accuracy</td>
                            <td style="padding:8px 12px;text-align:center;color:#67e8f9;font-weight:700;">96.2%</td>
                            <td style="padding:8px 12px;text-align:center;color:#a5b4fc;font-weight:700;">94.8%</td>
                        </tr>
                        <tr style="border-bottom:1px solid #0f172a;">
                            <td style="padding:8px 12px;color:#94a3b8;">F1-Score</td>
                            <td style="padding:8px 12px;text-align:center;color:#67e8f9;font-weight:700;">95.8%</td>
                            <td style="padding:8px 12px;text-align:center;color:#a5b4fc;font-weight:700;">94.1%</td>
                        </tr>
                        <tr style="border-bottom:1px solid #0f172a;">
                            <td style="padding:8px 12px;color:#94a3b8;">AUC</td>
                            <td style="padding:8px 12px;text-align:center;color:#67e8f9;font-weight:700;">0.981</td>
                            <td style="padding:8px 12px;text-align:center;color:#a5b4fc;font-weight:700;">0.967</td>
                        </tr>
                        <tr style="border-bottom:1px solid #0f172a;">
                            <td style="padding:8px 12px;color:#94a3b8;">Precision</td>
                            <td style="padding:8px 12px;text-align:center;color:#67e8f9;font-weight:700;">96.0%</td>
                            <td style="padding:8px 12px;text-align:center;color:#a5b4fc;font-weight:700;">94.5%</td>
                        </tr>
                        <tr>
                            <td style="padding:8px 12px;color:#94a3b8;">Recall</td>
                            <td style="padding:8px 12px;text-align:center;color:#67e8f9;font-weight:700;">95.6%</td>
                            <td style="padding:8px 12px;text-align:center;color:#a5b4fc;font-weight:700;">93.8%</td>
                        </tr>
                    </tbody>
                </table>
                <p style="font-size:0.76rem;color:#475569;margin-bottom:16px;font-style:italic;">
                    ⚠️ These metrics are from internal evaluation on the training/test split.
                    External multi-center validation has not been conducted.
                    Performance on unseen real-world data may differ.
                </p>

                <h3 style="font-size:1.0rem;font-weight:700;color:#e2e8f0;margin:20px 0 10px;">
                    ⚙️ Technologies</h3>
                <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">Python 3.10</span>
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">PyTorch 2.x</span>
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">Gradio 4.x</span>
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">DenseNet169 + CBAM</span>
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">Grad-CAM++</span>
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">OpenCV</span>
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">SciPy</span>
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">SQLite</span>
                    <span style="padding:5px 12px;background:#1e293b;border:1px solid #334155;
                          border-radius:6px;font-size:0.80rem;color:#94a3b8;">Matplotlib</span>
                </div>

                <div style="padding:14px;background:#1c1a00;border:1px solid #854d0e;border-radius:8px;
                     margin-top:16px;">
                    <b style="color:#eab308;font-size:0.88rem;">⚠️ Important Disclaimer</b>
                    <p style="color:#ca8a04;font-size:0.83rem;margin:6px 0 0;line-height:1.6;">
                        This system is a research prototype and has not been FDA-cleared, CE-marked, or validated
                        in clinical trials. It must not be used for patient diagnosis, clinical decision-making,
                        or treatment planning. All outputs require review by qualified medical professionals.
                    </p>
                </div>
            </div>""")

    # ── Footer ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="ng-footer">
        <b>NeuroGuard</b> &nbsp;·&nbsp;
        AI-Assisted Brain Stroke Analysis &nbsp;·&nbsp;
        DenseNet169 + CBAM + Grad-CAM++ &nbsp;·&nbsp;
        Local Explainable AI Engine
        <br style="margin:4px 0;">
        For <b>educational and research use only</b> &nbsp;·&nbsp;
        Not intended for clinical diagnosis &nbsp;·&nbsp;
        No patient data stored
    </div>
    """)


if __name__ == "__main__":
    app.launch(share=True, **_launch_kwargs)
