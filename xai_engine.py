"""
xai_engine.py — NeuroGuard Local Explainable AI Engine
=======================================================
Fully local, zero-API explainable AI for brain stroke classification.
No external services, no rate limits, no internet required.

Pipeline:
  1. Heatmap Statistical Analyzer — centroid, hemisphere, approx. region,
     contour analysis (Gaussian-smoothed, noise-filtered), coverage %
  2. Clinical Narrative Generator — medically responsible rule-based NLP
     Uses language: "consistent with", "may suggest", "requires clinical validation"
     Never: "diagnosis confirmed", "patient has stroke", "treatment required"
  3. Returns same JSON schema as previous Gemini-based system
"""

import os
import time
import shutil
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Stable image cache ─────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".image_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

FALLBACK_EXPLANATION = {
    "summary":                "AI-generated interpretation is temporarily unavailable. Refer to the local XAI panel above.",
    "clinical_interpretation": "The local XAI panel provides a structured interpretation of the model's prediction, confidence, and severity estimate.",
    "severity_explanation":   "Severity is estimated from Grad-CAM++ heatmap activation coverage. Refer to the progress bar.",
    "heatmap_explanation":    "The Grad-CAM++ overlay highlights approximate brain regions with highest model attention. Red/orange zones indicate high activation; blue zones indicate low activation.",
    "disclaimer":             "This is an AI-generated output for educational and research purposes only. It does not constitute a medical diagnosis. Clinical correlation with a qualified radiologist or neurologist is mandatory.",
    "_source":                "fallback",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Heatmap Statistical Analyzer
# ══════════════════════════════════════════════════════════════════════════════

def analyze_heatmap(heatmap: np.ndarray) -> dict:
    """
    Compute spatial statistics from a Grad-CAM++ heatmap (0–1 normalized, 224×224).

    Returns a dict with:
        hemisphere, approx_region, pattern, coverage_pct,
        peak_coord, contour_count, activation_spread, centroid
    """
    default = {
        "hemisphere":       "Indeterminate",
        "approx_region":    "Indeterminate",
        "pattern":          "Indeterminate",
        "coverage_pct":     0.0,
        "peak_coord":       (112, 112),
        "contour_count":    0,
        "activation_spread":"Confined",
        "centroid":         (112, 112),
    }

    if heatmap is None or heatmap.size == 0:
        return default

    try:
        import cv2
        from scipy.ndimage import gaussian_filter

        h = np.array(heatmap, dtype=np.float32)
        if h.max() < 1e-6:
            return default
        h = (h - h.min()) / (h.max() - h.min() + 1e-8)

        H, W = h.shape  # typically 224, 224

        # ── 1. Gaussian smoothing (reduces noise before contour analysis) ──────
        h_smooth = gaussian_filter(h, sigma=5.0)

        # ── 2. Coverage % (thresholded at 0.4 on smoothed map) ───────────────
        threshold    = 0.40
        active_mask  = h_smooth > threshold
        coverage_pct = round(float(np.sum(active_mask)) / (H * W) * 100, 1)

        # ── 3. Peak coordinate ────────────────────────────────────────────────
        peak_flat  = np.argmax(h_smooth)
        peak_y     = int(peak_flat // W)
        peak_x     = int(peak_flat %  W)
        peak_coord = (peak_x, peak_y)

        # ── 4. Centroid of active region ──────────────────────────────────────
        ys, xs = np.where(active_mask)
        if len(xs) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
        else:
            cx, cy = W // 2, H // 2
        centroid = (cx, cy)

        # ── 5. Hemisphere (left/right of image midline) ───────────────────────
        if cx < W * 0.40:
            hemisphere = "Left (image)"
        elif cx > W * 0.60:
            hemisphere = "Right (image)"
        else:
            hemisphere = "Bilateral / Central"

        # ── 6. Approximate region (quadrant mapping — NOT anatomically precise) ─
        # Prefixed "Approx." to indicate this is a spatial approximation only
        norm_x = cx / W
        norm_y = cy / H
        if norm_y < 0.33:
            region_v = "Superior"
        elif norm_y > 0.66:
            region_v = "Inferior"
        else:
            region_v = "Mid"

        if norm_x < 0.40:
            region_h = "Left"
        elif norm_x > 0.60:
            region_h = "Right"
        else:
            region_h = "Central"

        approx_region = f"Approx. {region_v}-{region_h}"

        # ── 7. Contour analysis (noise-filtered) ──────────────────────────────
        # Adaptive threshold on smoothed map; ignore contours < 1% of image area
        min_contour_area = int(H * W * 0.01)   # 1% threshold
        h_u8 = np.uint8(h_smooth * 255)
        _, thresh = cv2.threshold(h_u8, int(threshold * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        contour_count = len(valid_contours)

        # ── 8. Activation pattern ─────────────────────────────────────────────
        if contour_count <= 1:
            pattern = "Focal"
        elif contour_count <= 3:
            pattern = "Multi-focal"
        else:
            pattern = "Diffuse"

        # ── 9. Activation spread ──────────────────────────────────────────────
        if coverage_pct < 8:
            activation_spread = "Confined"
        elif coverage_pct < 25:
            activation_spread = "Moderate"
        else:
            activation_spread = "Extensive"

        return {
            "hemisphere":        hemisphere,
            "approx_region":     approx_region,
            "pattern":           pattern,
            "coverage_pct":      coverage_pct,
            "peak_coord":        peak_coord,
            "contour_count":     contour_count,
            "activation_spread": activation_spread,
            "centroid":          centroid,
        }

    except Exception as e:
        logger.warning(f"[xai_engine] Heatmap analysis failed: {e}")
        return default


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Calibrated Confidence Helper
# ══════════════════════════════════════════════════════════════════════════════

def get_confidence_tier(confidence: float) -> dict:
    """
    4-tier calibrated confidence classification.
    Returns label, color, clinical_note.
    """
    pct = confidence * 100
    if pct >= 90:
        return {
            "label":         "Highly Confident",
            "color":         "#22c55e",
            "clinical_note": "Model output shows strong activation patterns consistent with the predicted class.",
            "tier":          4,
            "warning":       False,
        }
    elif pct >= 75:
        return {
            "label":         "Moderately Confident",
            "color":         "#eab308",
            "clinical_note": "Clinical correlation is strongly recommended. Moderate confidence may reflect image quality or borderline features.",
            "tier":          3,
            "warning":       True,
        }
    elif pct >= 60:
        return {
            "label":         "Low Confidence",
            "color":         "#f97316",
            "clinical_note": "Radiologist review recommended. AI confidence is limited — result should be interpreted with caution.",
            "tier":          2,
            "warning":       True,
        }
    else:
        return {
            "label":         "Non-Diagnostic",
            "color":         "#ef4444",
            "clinical_note": "Uncertain prediction. This result should NOT be used for clinical decision-making. Please verify with a qualified specialist.",
            "tier":          1,
            "warning":       True,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Clinical Narrative Generator (Rule-Based, No API)
# ══════════════════════════════════════════════════════════════════════════════

# All templates use medically responsible language:
#   ✅ "consistent with", "may suggest", "activation regions indicate"
#   ❌ Never: "confirmed", "patient has", "treatment required"

_SUMMARIES = {
    ("Hemorrhagic", 4): (
        "The model demonstrates high-confidence imaging features consistent with hemorrhagic stroke presentation. "
        "Grad-CAM++ activation regions indicate dense, focal hyperdense patterns commonly associated with intraparenchymal blood products. "
        "This AI-generated output requires urgent clinical validation."
    ),
    ("Hemorrhagic", 3): (
        "Imaging features identified by the model are consistent with hemorrhagic stroke characteristics, "
        "though moderate confidence warrants careful clinical correlation. "
        "The activation pattern suggests the presence of hyperdense regions that may indicate blood leakage."
    ),
    ("Hemorrhagic", 2): (
        "The model identified some features that may be consistent with hemorrhagic changes, "
        "however confidence is limited. Results should be interpreted with caution and supplemented by expert radiological review."
    ),
    ("Hemorrhagic", 1): (
        "The model produced an uncertain output with features loosely resembling hemorrhagic patterns. "
        "This result is non-diagnostic and should not be used for any clinical assessment."
    ),
    ("Ischemic", 4): (
        "High-confidence model activation is consistent with ischemic stroke imaging patterns. "
        "The Grad-CAM++ heatmap indicates reduced tissue density zones that may suggest an area of reduced perfusion or infarction. "
        "Clinical correlation and time-sensitive evaluation are strongly advised."
    ),
    ("Ischemic", 3): (
        "The model detected imaging features that may be consistent with ischemic stroke. "
        "Moderate confidence suggests borderline activation patterns — radiological review and clinical correlation are recommended."
    ),
    ("Ischemic", 2): (
        "Some imaging features loosely consistent with ischemic changes were identified, "
        "but confidence is limited. Expert review is recommended before drawing any clinical conclusions."
    ),
    ("Ischemic", 1): (
        "The model output is uncertain and inconclusive. Features show some similarity to ischemic patterns "
        "but fall below a reliable detection threshold. This result is non-diagnostic."
    ),
    ("Normal", 4): (
        "No dominant imaging features consistent with acute hemorrhagic or ischemic stroke were identified by the model. "
        "The scan appears within normal range for the features analyzed. Clinical correlation with patient symptoms remains necessary."
    ),
    ("Normal", 3): (
        "The model did not detect features strongly consistent with acute stroke. "
        "A normal AI result does not definitively exclude stroke — clinical and symptom correlation is advised."
    ),
    ("Normal", 2): (
        "The model identified no clear stroke features, though confidence is limited. "
        "This result should be supplemented by specialist evaluation if clinical symptoms persist."
    ),
    ("Normal", 1): (
        "The model output is inconclusive. No definitive stroke features were identified, "
        "but confidence is insufficient for reliable interpretation."
    ),
}

_CLINICAL_INTERP = {
    "Hemorrhagic": (
        "Hemorrhagic stroke imaging is typically characterized by hyperdense regions on CT or "
        "T1-hyperintense/T2-hypointense signal on MRI, reflecting blood products. "
        "The model's activation regions suggest localized intensity changes consistent with this pattern. "
        "No medications, surgical interventions, or specific clinical recommendations are implied by this output."
    ),
    "Ischemic": (
        "Ischemic stroke on imaging typically presents as hypodense regions on CT or diffusion restriction on MRI DWI sequences, "
        "reflecting cytotoxic edema in the affected zone. "
        "The model's activation pattern may suggest areas of reduced perfusion or tissue density changes. "
        "This interpretation is model-generated and does not replace specialist assessment."
    ),
    "Normal": (
        "No significant focal abnormalities consistent with acute stroke were detected. "
        "The model assessed the scan as within expected normal range for the features it was trained to identify. "
        "A normal AI result does not constitute a radiological clearance — clinical presentation and expert review remain essential."
    ),
}

_SEVERITY_EXPLANATIONS = {
    ("Hemorrhagic", "High"): (
        "The Grad-CAM++ heatmap exhibits extensive high-activation coverage, suggesting a relatively large region of imaging abnormality "
        "consistent with hemorrhagic changes. Estimated activation coverage exceeds the threshold for high-severity classification."
    ),
    ("Hemorrhagic", "Moderate"): (
        "A moderate-sized activation region was identified by the heatmap, suggesting a localized area of imaging abnormality. "
        "This may correspond to a contained hemorrhagic zone, though spatial approximations should be interpreted cautiously."
    ),
    ("Hemorrhagic", "Low"): (
        "Heatmap activation is relatively limited in coverage, suggesting a small or diffuse abnormality. "
        "Severity scoring is an approximation based on activation area — not a volumetric measurement."
    ),
    ("Ischemic", "High"): (
        "Extensive Grad-CAM++ activation is consistent with a large region of potential ischemic involvement. "
        "This may reflect a large vessel occlusion pattern; however, activation extent in AI heatmaps does not directly correspond to clinical infarct volume."
    ),
    ("Ischemic", "Moderate"): (
        "Moderate heatmap coverage suggests a mid-sized region of potential ischemic change. "
        "Spatial extent is approximated from activation density and should not be treated as a precise lesion measurement."
    ),
    ("Ischemic", "Low"): (
        "Limited heatmap activation suggests a small or early-stage ischemic pattern, or possible lacunar involvement. "
        "Small infarcts may be underrepresented in Grad-CAM++ due to the model's resolution constraints."
    ),
    ("Normal", "N/A"): (
        "No heatmap severity scoring is applicable for Normal predictions. "
        "The model did not identify a dominant focal abnormality requiring activation-based severity assessment."
    ),
}

_HEATMAP_EXPLANATIONS = {
    "Hemorrhagic": (
        "The Grad-CAM++ heatmap highlights approximate brain regions where the model's DenseNet169+CBAM architecture "
        "assigned the highest attention weights for the hemorrhagic class prediction. "
        "Red/orange zones indicate strong feature activation — typically corresponding to regions of high image intensity. "
        "Blue zones represent areas of low model attention. "
        "Note: Heatmap regions are spatial approximations and do not represent precise anatomical boundaries."
    ),
    "Ischemic": (
        "The Grad-CAM++ heatmap indicates approximate regions where the model detected features consistent with ischemic changes, "
        "such as subtle hypodensity or diffusion signal alterations. "
        "Warm colors (red/orange) denote highest activation; cool colors (blue) indicate minimal attention. "
        "The activation map is an approximation of the model's decision basis, not an anatomical segmentation."
    ),
    "Normal": (
        "No Grad-CAM++ heatmap is generated for Normal predictions, as the model did not identify a dominant focal abnormality. "
        "The absence of a heatmap is consistent with a normal prediction — it does not imply the absence of subtle pathology beyond the model's detection scope."
    ),
}

_DISCLAIMER = (
    "This output is generated by an AI prototype for educational and research purposes only. "
    "It has not been validated in clinical trials, regulatory-cleared, or tested against diverse multi-center datasets. "
    "It must not be used as a substitute for professional medical diagnosis, radiological reporting, or clinical decision-making. "
    "All outputs require review by a qualified neurologist or radiologist."
)


def _get_severity_label(score: float, prediction: str) -> str:
    if prediction == "Normal":
        return "N/A"
    if score >= 6.0:
        return "High"
    elif score >= 3.0:
        return "Moderate"
    else:
        return "Low"


def generate_explanation(
    result_dict: dict,
    modality: str,
    image_path: str = None,
    heatmap: np.ndarray = None,
) -> dict:
    """
    Generate a fully local, rule-based clinical explanation.
    No API, no internet, no rate limits.

    Args:
        result_dict: {'prediction', 'confidence', 'severity'}
        modality:    'CT' or 'MRI'
        image_path:  path to original scan (unused locally but kept for API compat)
        heatmap:     numpy heatmap array for spatial analysis

    Returns:
        dict with keys: summary, clinical_interpretation, severity_explanation,
                        heatmap_explanation, disclaimer, _source, heatmap_analysis, xai_time_ms
    """
    t0 = time.time()

    if not result_dict:
        fb = FALLBACK_EXPLANATION.copy()
        fb["heatmap_analysis"] = analyze_heatmap(None)
        fb["xai_time_ms"] = 0
        return fb

    prediction = str(result_dict.get("prediction", "Normal"))
    confidence = float(result_dict.get("confidence", 0.0))
    severity   = float(result_dict.get("severity", 0.0))

    # Confidence tier (4-level)
    conf_tier  = get_confidence_tier(confidence)
    tier_num   = conf_tier["tier"]

    # Severity label
    sev_label  = _get_severity_label(severity, prediction)

    # Heatmap spatial analysis
    hmap_stats = analyze_heatmap(heatmap)

    # ── Build narrative ────────────────────────────────────────────────────────
    summary_key = (prediction, tier_num)
    summary     = _SUMMARIES.get(summary_key, _SUMMARIES.get((prediction, 3),
                  FALLBACK_EXPLANATION["summary"]))

    clinical    = _CLINICAL_INTERP.get(prediction, FALLBACK_EXPLANATION["clinical_interpretation"])

    sev_key     = (prediction, sev_label)
    sev_expl    = _SEVERITY_EXPLANATIONS.get(sev_key,
                  _SEVERITY_EXPLANATIONS.get((prediction, "Moderate"),
                  FALLBACK_EXPLANATION["severity_explanation"]))

    hmap_expl   = _HEATMAP_EXPLANATIONS.get(prediction, FALLBACK_EXPLANATION["heatmap_explanation"])

    # Append spatial info to heatmap explanation (for non-Normal predictions)
    if prediction != "Normal" and hmap_stats["coverage_pct"] > 0:
        spatial_note = (
            f" Approximate spatial analysis: {hmap_stats['pattern']} activation pattern detected "
            f"in the {hmap_stats['approx_region']} zone ({hmap_stats['hemisphere']} dominant), "
            f"covering ~{hmap_stats['coverage_pct']:.1f}% of the scan area. "
            f"Activation spread: {hmap_stats['activation_spread']}. "
            f"(Note: Region mapping is a coarse spatial approximation — no skull stripping or atlas registration was applied.)"
        )
        hmap_expl = hmap_expl + spatial_note

    xai_ms = int((time.time() - t0) * 1000)

    return {
        "summary":                summary,
        "clinical_interpretation": clinical,
        "severity_explanation":   sev_expl,
        "heatmap_explanation":    hmap_expl,
        "disclaimer":             _DISCLAIMER,
        "_source":                "local_rule_based",
        "heatmap_analysis":       hmap_stats,
        "confidence_tier":        conf_tier,
        "xai_time_ms":            xai_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Utility helpers (kept for backwards compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def cache_image(src_path: str, modality: str) -> str:
    """Copy uploaded scan to stable local dir so Gradio cleanup can't delete it."""
    try:
        ext = os.path.splitext(src_path)[-1] or ".png"
        dst = os.path.join(_CACHE_DIR, f"last_{modality.lower()}_scan{ext}")
        shutil.copy2(src_path, dst)
        return dst
    except Exception as e:
        print(f"[xai_engine] Image cache failed: {e}")
        return src_path


def clear_cache():
    print("[xai_engine] Local engine — no persistent cache.")


def cache_stats() -> dict:
    return {"mode": "local_rule_based (zero-API)"}
