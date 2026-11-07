from __future__ import annotations

import contextlib
import csv
import base64
import html
import importlib
import io
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")
os.environ.setdefault("STREAMLIT_CLIENT_SHOW_ERROR_DETAILS", "type")

import numpy as np
import pandas as pd
import streamlit as st
import torch
from src import cha_parser
from database.db import (
    init_db,
    save_patient,
    save_speech_analysis,
    save_brain_analysis,
    save_report,
    get_patient_history,
    get_latest_speech_analysis,
    get_latest_brain_analysis,
    get_reports,
    get_recent_patient_cases,
)
from src.combined_report_generator import (
    build_combined_case_data,
    generate_combined_rag_summary,
    save_combined_multimodal_report,
)


PROJECT_DIR = Path(__file__).resolve().parent
NLP_SRC_DIR = PROJECT_DIR / "nlp_rag_module" / "src"
CNN_SRC_DIR = PROJECT_DIR / "cnn_module" / "src"
OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
CNN_OUTPUT_DIR = PROJECT_DIR / "cnn_module" / "outputs"
CNN_REPORTS_DIR = CNN_OUTPUT_DIR / "reports"
MRI_EXPLANATIONS_DIR = CNN_OUTPUT_DIR / "mri_explanations"
SKULL_STRIPPED_OUTPUT_DIR = CNN_OUTPUT_DIR / "skull_stripped"
GRADCAM_OUTPUT_DIR = OUTPUT_DIR / "gradcam"
MRI_UPLOAD_DIR = CNN_OUTPUT_DIR / "uploads"
HISTORY_PATH = OUTPUT_DIR / "history.csv"
HISTORY_COLUMNS = [
    "date_time",
    "patient_case_id",
    "analysis_id",
    "analysis_type",
    "result",
    "confidence",
    "report_name",
    "notes",
]
# Legacy column names kept only so that older history.csv files can still be read.
LEGACY_HISTORY_ALIASES = {
    "patient_case_id": ["patient_id"],
    "result": ["prediction"],
}

MEDICAL_NOTE = (
    "This result is not a medical diagnosis. It is only a decision-support "
    "output and must be interpreted by a healthcare professional."
)
GRADCAM_MEDICAL_NOTE = (
    "This visualization is not a medical diagnosis. It is a visual explanation "
    "tool and must be interpreted by a healthcare professional."
)

_VECTORSTORE_CACHE = None
_MRI_VOLUME_CACHE: dict[str, np.ndarray] = {}


for module_dir in (NLP_SRC_DIR, CNN_SRC_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)


# ---------------------------------------------------------------------------
# Patient-friendly text cleaning (UI + reports)
# ---------------------------------------------------------------------------
# Technical terms that must never appear in patient-facing text.
_TECHNICAL_REPLACEMENTS = {
    "grad-cam": "visual explanation",
    "gradcam": "visual explanation",
    "grad cam": "visual explanation",
    "3d cam": "3D visual explanation",
    "vector store": "medical knowledge base",
    "vectorstore": "medical knowledge base",
    "embeddings": "medical knowledge base",
    "embedding": "medical knowledge base",
    "model output": "result",
    "model prediction": "result",
    "ai output label": "result",
    "the model": "the system",
    " model ": " system ",
    "faiss": "medical knowledge base",
    "rag": "medical knowledge base",
    "llm": "language helper",
    "cnn": "image analysis",
    "nlp": "language analysis",
    "xai": "visual explanation",
    "heatmap": "visual color overlay",
    "tokenizer": "language processing component",
    "scaler": "data processing component",
    "tensor": "internal data",
    "logits": "internal scores",
}

# Whole lines containing any of these tokens are removed from reports.
_FORBIDDEN_LINE_TOKENS = (
    ".pth",
    "faiss",
    "embedding",
    "traceback",
    "c:\\users",
    "/users/",
    "site-packages",
    "stderr",
    "stdout",
    "stack trace",
    "warning:",
    "warnings",
    "exception:",
    "module not found",
    "task_nlp_cha",
    "now execute",
    "do not continue",
    "search / replace",
    "tokenizer",
    "scaler",
    "tensor",
    "logits",
)

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s\"']+")
_POSIX_PATH_RE = re.compile(r"(?:/[\w.-]+){3,}")

# Chatbot-style openers that must never appear in clinical responses.
_CHATBOT_OPENER_RE = re.compile(
    r"^(?:"
    r"okay[,!.]?\s*|ok[,!.]?\s*|sure[,!.]?\s*|certainly[,!.]?\s*|"
    r"of course[,!.]?\s*|great[,!.]?\s*|absolutely[,!.]?\s*|"
    r"based on (?:your|the|my|our|these|this)\s+(?:guidelines?|context|information|data|"
    r"results?|parameters?|assessment|case|analysis|imaging|scan|question)[,!.]?\s*|"
    r"here(?:'s| is)[,!.]?\s*|"
    r"let me (?:provide|explain|clarify|answer)[,!.]?\s*|"
    r"i(?:'ll| will) (?:provide|explain|clarify|answer)[,!.]?\s*|"
    r"as requested[,!.]?\s*"
    r")",
    re.IGNORECASE,
)


def strip_chatbot_preamble(text: str) -> str:
    """Strip chatbot-style openers from LLM answers before clinical display."""
    cleaned = (text or "").strip()
    cleaned = _CHATBOT_OPENER_RE.sub("", cleaned).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def replace_technical_terms(text: str) -> str:
    cleaned = str(text or "")
    for technical, friendly in _TECHNICAL_REPLACEMENTS.items():
        pattern = re.escape(technical)
        if re.fullmatch(r"[A-Za-z0-9-]+", technical):
            pattern = rf"\b{pattern}\b"
        cleaned = re.sub(pattern, friendly, cleaned, flags=re.IGNORECASE)
    return cleaned


def patient_friendly_text(text: str) -> str:
    """Replace technical jargon in any displayed text with friendly wording."""
    if not text:
        return ""
    cleaned = replace_technical_terms(text)
    cleaned = _WINDOWS_PATH_RE.sub("", cleaned)
    return cleaned.strip()


def clean_report_text(text: str) -> str:
    """
    Remove technical lines (paths, logs, warnings, traces) and replace jargon.
    Used as a final guard before any report content is shown or saved.
    """
    if not text:
        return ""

    kept_lines = []
    for raw_line in str(text).splitlines():
        if raw_line.strip().startswith("!["):
            kept_lines.append(raw_line)
            continue

        lower = raw_line.lower()
        if any(token in lower for token in _FORBIDDEN_LINE_TOKENS):
            continue
        line = _WINDOWS_PATH_RE.sub("", raw_line)
        # Only strip long posix paths that are clearly filesystem paths, not URLs.
        if "http" not in line:
            line = _POSIX_PATH_RE.sub("", line)
        kept_lines.append(line)

    cleaned_lines = []
    for line in kept_lines:
        if line.strip().startswith("!["):
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(replace_technical_terms(line))
    cleaned = "\n".join(cleaned_lines)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Filesystem / pipeline helpers (backend logic unchanged)
# ---------------------------------------------------------------------------
def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CNN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CNN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MRI_EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    SKULL_STRIPPED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MRI_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for legacy_report in OUTPUT_DIR.glob("patient_report_*.md"):
        legacy_name = legacy_report.stem.replace("patient_report_", "")
        target = REPORTS_DIR / f"nlp_report_legacy_{legacy_name}.md"
        if not target.exists():
            target.write_text(legacy_report.read_text(encoding="utf-8"), encoding="utf-8")


def run_quietly(function, *args, **kwargs):
    """
    Run model functions without leaking internal print output into Streamlit logs.
    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        return function(*args, **kwargs)


def run_hd_bet_cpu(input_path: str, output_path: str) -> str:
    """
    Run HD-BET brain extraction in CPU-only mode.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hd_bet_command = "hd-bet"
    local_hd_bet = PROJECT_DIR / ".venv" / "Scripts" / "hd-bet.exe"
    if local_hd_bet.exists():
        hd_bet_command = str(local_hd_bet)

    command = [
        hd_bet_command,
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-device",
        "cpu",
        "--disable_tta",
    ]
    print("Running HD-BET command:", command)
    print("HD-BET input:", input_path)
    print("HD-BET expected output:", output_path)

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    print("HD-BET stdout:", result.stdout)
    print("HD-BET stderr:", result.stderr)
    print("HD-BET output directory files:", list(output_path.parent.glob("*")))

    actual_output = find_hd_bet_output(output_path)
    print("HD-BET actual output:", actual_output)

    return str(actual_output)


def hd_bet_output_path(analysis_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(analysis_id or "brain"))
    return SKULL_STRIPPED_OUTPUT_DIR / f"skull_stripped_{safe_id}.nii.gz"


def nifti_stem_and_suffix(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    name = path.name
    lower_name = name.lower()
    if lower_name.endswith(".nii.gz"):
        return name[:-7], ".nii.gz"
    if lower_name.endswith(".nii"):
        return name[:-4], ".nii"
    return path.stem, path.suffix


def find_hd_bet_output(expected_output_path: Path) -> Path:
    """
    HD-BET may write either the requested path or a *_bet variant.
    This helper handles .nii and .nii.gz without relying on Path.suffix alone.
    """
    expected_output_path = Path(expected_output_path)
    stem, suffix = nifti_stem_and_suffix(expected_output_path)
    possible_outputs = [
        expected_output_path,
        expected_output_path.with_name(f"{stem}_bet{suffix}"),
    ]

    for candidate in possible_outputs:
        if candidate.exists():
            return candidate

    print("HD-BET expected output not found")
    print("Expected:", expected_output_path)
    print("Possible outputs:", possible_outputs)
    print("Existing files:", list(expected_output_path.parent.glob("*")))
    raise FileNotFoundError("HD-BET expected output not found.")


def log_brain_analysis_error(exc: Exception) -> None:
    try:
        ensure_output_dirs()
        error_path = OUTPUT_DIR / "last_brain_analysis_error.txt"
        error_path.write_text(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def run_cnn_prediction(mri_path: str | Path):
    importlib.invalidate_caches()
    cnn_model = importlib.import_module("cnn_model")
    importlib.reload(cnn_model)
    cnn_predictor = importlib.import_module("cnn_predictor")
    cnn_predictor = importlib.reload(cnn_predictor)
    torch = importlib.import_module("torch")
    cnn_predictor.DEVICE = torch.device("cpu")

    return run_quietly(cnn_predictor.predict_mri, mri_path)


def run_gradcam_generation(
    mri_path: str | Path,
    orientation: str = "multi",
    num_slices: int = 5,
    display_mode: str = "overlay",
    output_dir: str | Path | None = None,
):
    importlib.invalidate_caches()
    cnn_model = importlib.import_module("cnn_model")
    importlib.reload(cnn_model)
    cnn_predictor = importlib.import_module("cnn_predictor")
    importlib.reload(cnn_predictor)
    torch = importlib.import_module("torch")
    cnn_predictor.DEVICE = torch.device("cpu")
    gradcam_3d = importlib.import_module("gradcam_3d")

    gradcam_3d = importlib.reload(gradcam_3d)
    gradcam_3d.DEVICE = torch.device("cpu")
    return run_quietly(
        gradcam_3d.generate_gradcam_slices,
        mri_path,
        orientation=orientation,
        num_slices=num_slices,
        output_dir=output_dir or GRADCAM_OUTPUT_DIR,
        display_mode=display_mode,
        alpha=0.60,
        threshold=0.48,
        percentile=88.0,
        colormap="turbo",
    )


def run_mri_rag_explanation(
    cnn_result: dict,
    gradcam_info: dict | None = None,
    question: str | None = None,
) -> dict:
    mri_rag_explainer = importlib.import_module("mri_rag_explainer")
    mri_rag_explainer = importlib.reload(mri_rag_explainer)

    return run_quietly(
        mri_rag_explainer.generate_mri_explanation_with_rag,
        cnn_result,
        gradcam_info=gradcam_info,
        question=question,
    )


def run_nlp_prediction(transcript: str, feature_values: list):
    numeric_features = [float(value) for value in feature_values]
    print("Speech features sent to model:", numeric_features)
    print("Feature types:", [type(value) for value in numeric_features])

    if len(numeric_features) != len(CHA_FEATURE_ORDER):
        raise ValueError(f"Speech feature vector must contain {len(CHA_FEATURE_ORDER)} values.")

    predict_nlp_model = importlib.import_module("predict_nlp_model")
    predict_nlp_model = importlib.reload(predict_nlp_model)
    predict_nlp_model.get_device = lambda: torch.device("cpu")

    prediction, confidence = run_quietly(
        predict_nlp_model.predict_patient,
        transcript=transcript,
        feature_values=numeric_features,
    )

    try:
        rag_explainer = importlib.import_module("rag_explainer")
        rag_explainer = importlib.reload(rag_explainer)
        return run_quietly(
            rag_explainer.explain_prediction_with_rag,
            transcript=transcript,
            prediction=prediction,
            confidence=confidence,
            k=4,
        )
    except Exception as exc:
        print("Speech RAG explanation error:", repr(exc))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "context": "",
            "explanation": "",
            "sources": [],
        }


def percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def display_value(value) -> str:
    if value is None:
        return "Not provided"

    if hasattr(value, "strftime"):
        return value.strftime("%d/%m/%Y")

    text = str(value).strip()
    return text if text else "Not provided"


def patient_identifier(patient_id: str) -> str:
    patient_id = str(patient_id or "").strip()
    return patient_id if patient_id else "Not provided"


def generate_patient_case_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4].upper()
    return f"PAT-{timestamp}-{suffix}"


def generate_analysis_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4].upper()
    return f"AN-{timestamp}-{suffix}"


def ensure_patient_case_id() -> str:
    """
    Return the single global Patient Case ID for this session, creating it once
    if needed. The same Patient Case ID is reused for every analysis, report and
    history entry until the user starts a new patient case.
    """
    case_id = str(st.session_state.get("patient_case_id", "")).strip()
    if not case_id:
        case_id = generate_patient_case_id()
        st.session_state["patient_case_id"] = case_id
    return case_id


# Backwards-compatible alias: every previous call site now resolves to the
# global Patient Case ID instead of a manually entered patient id.
def ensure_patient_id() -> str:
    return ensure_patient_case_id()


def safe_filename_part(value: str, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text or text == "Not provided":
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or fallback


def report_display_name(report_path: str | Path | None) -> str:
    if not report_path:
        return "No report"
    name = Path(str(report_path)).name
    name = re.sub(r"^cnn_report_", "brain_scan_report_", name, flags=re.IGNORECASE)
    name = re.sub(r"^nlp_report_", "speech_language_report_", name, flags=re.IGNORECASE)
    name = re.sub(
        r"^combined_multimodal_report_",
        "combined_multimodal_summary_",
        name,
        flags=re.IGNORECASE,
    )
    return name


def report_image_reference(image_path: str | Path) -> str:
    path = Path(image_path)
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.name


def gradcam_item_path(item) -> Path | None:
    if isinstance(item, dict):
        path_value = item.get("image_path") or item.get("path")
        return Path(path_value) if path_value else None
    return Path(item) if item else None


def gradcam_item_caption(
    item,
    fallback_number: int,
    fallback_orientation: str | None = None,
) -> str:
    if isinstance(item, dict):
        caption = str(item.get("caption") or "").strip()
        stale_patterns = [
            "the colors are more diffuse across this image",
            "moderate visual influence",
            "stronger highlighted areas",
            "mainly central highlights",
        ]
        if caption and not any(pattern in caption.lower() for pattern in stale_patterns):
            return caption

        slice_number = int(item.get("slice_number") or fallback_number)
        orientation = str(item.get("orientation") or fallback_orientation or "")
        metrics = {
            "intensity_level": item.get("intensity_level")
            or item.get("highlight_level")
            or "moderate",
            "highlight_area_percent": item.get("highlight_area_percent") or 0.0,
            "brightest_zone": item.get("brightest_zone")
            or item.get("dominant_zone")
            or "center",
            "spread_type": item.get("spread_type") or "medium",
        }
        if metrics["brightest_zone"] == "diffuse":
            metrics["brightest_zone"] = "center"
            metrics["spread_type"] = "diffuse"
        try:
            gradcam_3d = importlib.import_module("gradcam_3d")
            gradcam_3d = importlib.reload(gradcam_3d)
            return gradcam_3d.build_specific_slice_caption(
                slice_number,
                orientation,
                metrics,
            )
        except Exception:
            pass

        return get_visual_explanation_caption(slice_number, orientation)

    return get_visual_explanation_caption(fallback_number, fallback_orientation)


def valid_gradcam_items(items) -> list:
    valid_items = []
    for item in items or []:
        image_path = gradcam_item_path(item)
        if image_path and image_path.exists() and image_path.is_file():
            valid_items.append(item)
    return valid_items


def visual_explanation_slice_count(items) -> int:
    total = 0
    for item in valid_gradcam_items(items or []):
        if isinstance(item, dict):
            displayed_slices = item.get("displayed_slices")
            if displayed_slices:
                total += int(displayed_slices)
                continue
            view_count = int(item.get("view_count") or 0)
            slices_per_view = int(item.get("slices_per_view") or 0)
            if view_count and slices_per_view:
                total += view_count * slices_per_view
                continue
        total += 1
    return total


def visual_explanation_orientation_label(items, fallback: str | None = None) -> str:
    valid_items = valid_gradcam_items(items or [])
    if valid_items:
        first_item = valid_items[0]
        if isinstance(first_item, dict) and first_item.get("orientation") == "multi-axis":
            return "Axial, sagittal and coronal views"
    return fallback or "Not generated"


def visual_explanation_display_mode_label(items, fallback: str | None = None) -> str:
    valid_items = valid_gradcam_items(items or [])
    if valid_items:
        first_item = valid_items[0]
        if isinstance(first_item, dict):
            display_mode = str(first_item.get("display_mode") or "").lower()
            if display_mode == "heatmap":
                return "Heatmap only"
            if display_mode == "overlay":
                return "Overlay view"
    return fallback or "Not generated"


def history_report_name_value(report_path: str | Path | None) -> str:
    if not report_path:
        return ""
    return Path(str(report_path)).name


def report_kind(report_path: Path) -> str:
    name = report_path.name.lower()
    if name.startswith("combined_multimodal_report_") or name.startswith("combined_multimodal_summary_"):
        return "Combined Multimodal"
    if name.startswith("brain_scan_report_") or name.startswith("cnn_report_"):
        return "Brain Scan"
    if name.startswith("speech_language_report_") or name.startswith("nlp_report_"):
        return "Speech and Language"
    return "Report"


def report_download_label(report_path: Path) -> str:
    kind = report_kind(report_path)
    if kind == "Brain Scan":
        return "Download Brain Scan Report"
    if kind == "Speech and Language":
        return "Download Speech and Language Report"
    return "Download Report"


def report_download_filename(report_path: Path) -> str:
    return report_display_name(report_path)


def friendly_analysis_type(value: str) -> str:
    text = str(value or "").strip()
    if text in {"Brain MRI", "MRI Analysis", "Brain MRI Analysis", "Brain Scan Analysis"}:
        return "Brain Scan Analysis"
    if text in {
        "Speech Transcript",
        "Speech Analysis",
        "Speech Transcript Analysis",
        "Speech and Language Analysis",
    }:
        return "Speech and Language Analysis"
    return text


def _date_from_id(identifier: str) -> str | None:
    match = re.search(r"(\d{8})-(\d{6})", str(identifier or ""))
    if not match:
        return None
    try:
        return datetime.strptime(
            match.group(1) + match.group(2), "%Y%m%d%H%M%S"
        ).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def report_card_metadata(report_path: Path) -> dict:
    """
    Return clean, patient-friendly metadata for a generated report file.
    Never exposes file paths, function names or internal labels.
    """
    stem = report_path.stem
    case_id = "Not available"
    analysis_id = "Not available"
    date_text = None

    new_match = re.match(
        r"(?:brain_scan|speech_language|combined_multimodal)_report_(PAT-[0-9A-Za-z-]+?)_(AN-[0-9A-Za-z-]+)$",
        stem,
    )
    if new_match:
        case_id = new_match.group(1)
        analysis_id = new_match.group(2)
        date_text = _date_from_id(analysis_id) or _date_from_id(case_id)
    else:
        legacy_match = re.match(r"(?:cnn|nlp)_report_(.+)_(\d{8}_\d{6})$", stem)
        if legacy_match:
            case_id = legacy_match.group(1)
            try:
                date_text = datetime.strptime(
                    legacy_match.group(2), "%Y%m%d_%H%M%S"
                ).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                date_text = None

    if not date_text:
        date_text = datetime.fromtimestamp(report_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return {
        "patient_case_id": case_id,
        "analysis_id": analysis_id,
        "analysis_type": report_kind(report_path),
        "date": date_text,
    }


def report_metadata(report_path: Path) -> tuple[str, str]:
    meta = report_card_metadata(report_path)
    return meta["patient_case_id"], meta["date"]


def append_history(
    analysis_type: str,
    patient_case_id: str,
    analysis_id: str,
    result: str,
    confidence: float,
    report_path: str | Path | None,
    notes: str,
) -> None:
    ensure_output_dirs()
    row = {
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_case_id": patient_identifier(patient_case_id),
        "analysis_id": str(analysis_id or "").strip() or "Not available",
        "analysis_type": friendly_analysis_type(analysis_type),
        "result": str(result),
        "confidence": percent(confidence),
        "report_name": history_report_name_value(report_path),
        "notes": " ".join(str(notes).split()),
    }

    history = load_history()
    if not history.empty and "analysis_id" in history.columns:
        # Each analysis has a unique Analysis ID, so we only block exact repeats.
        duplicate_mask = history["analysis_id"].astype(str) == row["analysis_id"]
        if duplicate_mask.any():
            history = history.loc[~duplicate_mask].reset_index(drop=True)

    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    save_history(history)


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    def pick(raw_row: dict, column: str) -> str:
        value = str(raw_row.get(column, "") or "").strip()
        if value:
            return value
        for alias in LEGACY_HISTORY_ALIASES.get(column, []):
            alias_value = str(raw_row.get(alias, "") or "").strip()
            if alias_value:
                return alias_value
        return ""

    rows = []
    with HISTORY_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw_row in reader:
            if not raw_row:
                continue

            row = {column: "" for column in HISTORY_COLUMNS}
            row["date_time"] = pick(raw_row, "date_time")
            row["patient_case_id"] = pick(raw_row, "patient_case_id") or "Not available"
            row["analysis_id"] = pick(raw_row, "analysis_id") or "Not available"
            row["analysis_type"] = friendly_analysis_type(pick(raw_row, "analysis_type"))
            row["result"] = pick(raw_row, "result")
            row["confidence"] = pick(raw_row, "confidence")

            report_name = str(
                raw_row.get("report_name", "") or raw_row.get("report_path", "")
            ).strip()
            notes = str(raw_row.get("notes", "")).strip()
            extra_values = raw_row.get(None) or []

            if extra_values:
                if report_name and not notes:
                    notes = str(extra_values[0]).strip()
                elif notes and not report_name and (
                    "report" in notes.lower() or notes.lower().endswith(".md")
                ):
                    report_name = notes
                    notes = str(extra_values[0]).strip()
                else:
                    notes = " ".join([notes, *map(str, extra_values)]).strip()

            row["report_name"] = report_display_name(report_name) if report_name else ""
            row["notes"] = friendly_history_notes(notes)
            rows.append(row)

    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return history.drop_duplicates().reset_index(drop=True)


def save_history(history: pd.DataFrame) -> None:
    ensure_output_dirs()
    history = history.copy()
    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = ""
    history[HISTORY_COLUMNS].to_csv(HISTORY_PATH, index=False, encoding="utf-8")


def delete_history_row(row_index: int) -> None:
    history = load_history().reset_index(drop=True)
    if 0 <= row_index < len(history):
        history = history.drop(index=row_index).reset_index(drop=True)
        save_history(history)


def friendly_history_notes(notes: str) -> str:
    text = str(notes or "").strip()
    replacements = {
        "CNN image analysis completed.": "MRI analysis completed.",
        "Speech transcript NLP analysis completed.": "Speech analysis completed.",
        "CNN report generated successfully.": "Brain scan report generated successfully.",
        "NLP report generated successfully.": "Speech and language report generated successfully.",
    }
    return replacements.get(text, text if text else "Analysis completed.")


def db_clean_value(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"not provided", "none", "nan"}:
        return None
    return text


def db_available() -> bool:
    return bool(st.session_state.get("db_available", False))


def refresh_database_status() -> bool:
    try:
        init_db()
        st.session_state.db_available = True
        return True
    except Exception as exc:
        print("Database connection check failed:", repr(exc))
        st.session_state.db_available = False
        return False


def log_database_debug(db_connected: bool) -> None:
    print("=== DATABASE DEBUG ===")
    print("DATABASE_URL exists:", bool(os.getenv("DATABASE_URL")))
    print("Database connected:", db_connected)
    print("Current patient_case_id:", st.session_state.get("patient_case_id"))


def warn_database_not_saved(saved: bool) -> None:
    if not saved:
        note("Patient history is currently unavailable. Results were generated but not saved.", "warning")


def current_patient_context(patient_info: dict | None = None) -> dict:
    patient_info = patient_info or {}
    study_date = patient_info.get("Study Date") or st.session_state.get("speech_study_date")
    if not study_date:
        study_date = datetime.now().date()

    return {
        "patient_name": db_clean_value(
            patient_info.get("Patient Name") or st.session_state.get("speech_patient_name")
        ),
        "study_date": db_clean_value(study_date),
        "responsible_clinician": db_clean_value(
            patient_info.get("Responsible Clinician") or st.session_state.get("speech_clinician")
        ),
        "clinical_notes": db_clean_value(
            sanitize_clinical_notes(
                patient_info.get("Clinical Notes") or st.session_state.get("speech_notes", "")
            )
        ),
    }


def save_patient_to_database(patient_case_id: str, patient_info: dict | None = None) -> bool:
    if not db_available():
        return False

    context = current_patient_context(patient_info)
    return save_patient(
        patient_case_id=patient_identifier(patient_case_id),
        patient_name=context["patient_name"],
        study_date=context["study_date"],
        responsible_clinician=context["responsible_clinician"],
        clinical_notes=context["clinical_notes"],
    )


def first_visual_explanation_path() -> Path | None:
    for item in valid_gradcam_items(st.session_state.get("gradcam_paths") or []):
        image_path = gradcam_item_path(item)
        if image_path and image_path.exists():
            return image_path
    return None


def save_report_to_database(
    patient_case_id: str,
    analysis_id: str,
    analysis_type: str,
    report_paths: dict | None = None,
    report_path: str | Path | None = None,
) -> bool:
    if not db_available() and not refresh_database_status():
        return False

    formats = report_paths.copy() if isinstance(report_paths, dict) else {}
    if report_path and not formats:
        formats = report_sibling_files(Path(report_path))

    return save_report(
        patient_case_id=patient_identifier(patient_case_id),
        analysis_id=str(analysis_id or "").strip(),
        analysis_type=friendly_analysis_type(analysis_type),
        report_md_path=formats.get("md"),
        report_html_path=formats.get("html"),
        report_pdf_path=formats.get("pdf"),
    )


def save_existing_report_to_database(report_path: str | Path) -> bool:
    if not db_available():
        return False

    path = Path(report_path)
    metadata = report_card_metadata(path)
    patient_case_id = metadata.get("patient_case_id") or ensure_patient_case_id()
    analysis_id = metadata.get("analysis_id") or ""
    if patient_case_id == "Not available":
        patient_case_id = ensure_patient_case_id()
    if analysis_id == "Not available":
        analysis_id = (
            st.session_state.get("cnn_analysis_id")
            or st.session_state.get("nlp_analysis_id")
            or ""
        )

    return save_report_to_database(
        patient_case_id=patient_case_id,
        analysis_id=analysis_id,
        analysis_type=metadata.get("analysis_type") or report_kind(path),
        report_path=path,
    )


def save_speech_result_to_database(
    patient_case_id: str,
    analysis_id: str,
    patient_info: dict,
    uploaded_file_name: str,
    cleaned_transcript: str,
    extracted_features: dict,
    result: dict,
    simple_explanation: str,
    report_paths: dict | None = None,
) -> bool:
    if not db_available() and not refresh_database_status():
        return False

    patient_saved = save_patient_to_database(patient_case_id, patient_info)
    analysis_saved = save_speech_analysis(
        patient_case_id=patient_identifier(patient_case_id),
        analysis_id=str(analysis_id),
        uploaded_file_name=uploaded_file_name,
        speech_file_path=st.session_state.get("nlp_uploaded_file_path"),
        cleaned_transcript=cleaned_transcript,
        extracted_features=extracted_features,
        result=result.get("prediction"),
        confidence=result.get("confidence"),
        simple_explanation=simple_explanation,
    )
    report_saved = True
    if report_paths:
        report_saved = save_report_to_database(
            patient_case_id=patient_case_id,
            analysis_id=analysis_id,
            analysis_type="Speech and Language Analysis",
            report_paths=report_paths,
        )
    print(
        "Speech database save:",
        {
            "patient_case_id": patient_identifier(patient_case_id),
            "analysis_id": str(analysis_id),
            "patient_saved": patient_saved,
            "analysis_saved": analysis_saved,
            "report_saved": report_saved,
        },
    )
    return bool(patient_saved and analysis_saved and report_saved)


def save_brain_result_to_database(
    patient_case_id: str,
    analysis_id: str,
    result: dict,
    uploaded_file_name: str | None = None,
    mri_file_path: str | Path | None = None,
    report_paths: dict | None = None,
    save_report_entry: bool = True,
) -> bool:
    if not db_available() and not refresh_database_status():
        return False

    patient_saved = save_patient_to_database(patient_case_id)
    analysis_saved = save_brain_analysis(
        patient_case_id=patient_identifier(patient_case_id),
        analysis_id=str(analysis_id),
        uploaded_file_name=uploaded_file_name or st.session_state.get("mri_source_name"),
        mri_file_path=mri_file_path or st.session_state.get("mri_path"),
        result=result.get("prediction"),
        confidence=result.get("confidence"),
        prob_cn=result.get("prob_cn"),
        prob_ad=result.get("prob_ad"),
        visual_explanation_path=first_visual_explanation_path(),
        simple_explanation=brain_scan_simple_explanation(result.get("prediction", "")),
    )
    report_saved = True
    if save_report_entry and report_paths:
        report_saved = save_report_to_database(
            patient_case_id=patient_case_id,
            analysis_id=analysis_id,
            analysis_type="Brain Scan Analysis",
            report_paths=report_paths,
        )
    return bool(patient_saved and analysis_saved and report_saved)


def stored_path_to_local_path(value: str | Path | None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path if path.exists() else None


def database_report_paths(patient_case_id: str) -> list[Path]:
    if not db_available():
        return []

    paths: list[Path] = []
    seen = set()
    for row in get_reports(patient_identifier(patient_case_id)):
        for key in ("report_html_path", "report_md_path", "report_pdf_path"):
            path = stored_path_to_local_path(row.get(key))
            if path and path not in seen:
                paths.append(path)
                seen.add(path)
                break
    return paths


def database_history(patient_case_id: str) -> pd.DataFrame:
    if not db_available():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    report_rows = get_reports(patient_identifier(patient_case_id))
    reports_by_analysis = {}
    for report in report_rows:
        analysis_id = str(report.get("analysis_id") or "")
        if analysis_id and analysis_id not in reports_by_analysis:
            report_path = (
                report.get("report_html_path")
                or report.get("report_md_path")
                or report.get("report_pdf_path")
                or ""
            )
            reports_by_analysis[analysis_id] = report_display_name(report_path)

    rows = []
    for row in get_patient_history(patient_identifier(patient_case_id)):
        created_at = row.get("created_at")
        if hasattr(created_at, "strftime"):
            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        analysis_id = str(row.get("analysis_id") or "Not available")
        rows.append({
            "date_time": str(created_at or ""),
            "patient_case_id": patient_identifier(patient_case_id),
            "analysis_id": analysis_id,
            "analysis_type": friendly_analysis_type(row.get("analysis_type", "")),
            "result": str(row.get("result") or ""),
            "confidence": percent(row.get("confidence") or 0.0),
            "report_name": reports_by_analysis.get(analysis_id, ""),
            "notes": "Analysis saved in database.",
        })

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def recent_patient_case_label(row: dict) -> str:
    patient_id = str(row.get("patient_case_id") or "Not available")
    patient_name = str(row.get("patient_name") or "Unnamed patient")
    latest = row.get("latest_created_at")
    if hasattr(latest, "strftime"):
        latest = latest.strftime("%Y-%m-%d %H:%M:%S")
    latest_text = str(latest or "No date")
    brain = "Brain: yes" if row.get("has_brain_analysis") else "Brain: no"
    speech = "Speech: yes" if row.get("has_speech_analysis") else "Speech: no"
    combined = "Combined: yes" if row.get("has_combined_report") else "Combined: no"
    return f"{patient_id} | {patient_name} | {latest_text} | {brain} | {speech} | {combined}"


def render_patient_case_selector(current_patient_id: str) -> str:
    if not db_available():
        return current_patient_id

    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
    section(
        "Search / Select Previous Patient Case",
        "Saved patient cases can be recovered by selecting a previous Patient Case ID.",
    )

    selected_patient_id = current_patient_id
    with st.container(border=True):
        typed_id = st.text_input(
            "Paste Patient Case ID",
            value="",
            placeholder="Example: PAT-20260625-024256-A69E",
            key="reports_patient_case_search",
        )
        if st.button("Load Patient Case ID", key="load_typed_patient_case", use_container_width=True):
            candidate = typed_id.strip()
            if candidate:
                st.session_state.patient_case_id = candidate
                rerun_app()

        recent_cases = get_recent_patient_cases(limit=20)
        if recent_cases:
            labels = ["Select a recent patient case"]
            case_by_label = {}
            for row in recent_cases:
                label = recent_patient_case_label(row)
                labels.append(label)
                case_by_label[label] = str(row.get("patient_case_id") or "")

            selected_label = st.selectbox(
                "Recent patient cases",
                options=labels,
                index=0,
                key="reports_recent_patient_case",
            )
            selected_from_list = case_by_label.get(selected_label, "")
            if selected_from_list and selected_from_list != current_patient_id:
                if st.button("Load Selected Patient Case", key="load_selected_patient_case", use_container_width=True):
                    st.session_state.patient_case_id = selected_from_list
                    rerun_app()
        else:
            note("No saved patient cases were found yet.", "info")

    return selected_patient_id


def combined_report_paths(patient_case_id: str) -> list[Path]:
    if not db_available():
        return [
            path for path in generated_reports()
            if path.name.lower().startswith("combined_multimodal_report_")
        ]

    paths: list[Path] = []
    seen = set()
    for row in get_reports(patient_identifier(patient_case_id)):
        analysis_type = str(row.get("analysis_type") or "")
        if analysis_type != "Combined Multimodal Summary":
            continue
        for key in ("report_html_path", "report_md_path", "report_pdf_path"):
            path = stored_path_to_local_path(row.get(key))
            if path and path not in seen:
                paths.append(path)
                seen.add(path)
                break
    return paths


def combined_status_label(row: dict | None) -> str:
    return "Available" if row else "Missing"


def render_combined_status_cards(brain_row: dict | None, speech_row: dict | None) -> None:
    brain_ok  = brain_row  is not None
    speech_ok = speech_row is not None

    def _card(ok: bool, label: str, detail: str) -> str:
        val_cls    = "assess-val-ok"    if ok else "assess-val-miss"
        badge_cls  = "assess-badge-ok"  if ok else "assess-badge-miss"
        val_text   = "Available"        if ok else "Missing"
        badge_text = "Ready"            if ok else "Pending"
        return (
            f'<div class="assess-status-card fade-in">'
            f'  <div>'
            f'    <div class="assess-lbl">{html.escape(label)}</div>'
            f'    <div class="assess-val {val_cls}">{val_text}</div>'
            f'    <div class="assess-{badge_cls}">{badge_text}</div>'
            f'    <div style="font-size:0.74rem;color:#6B7C75;margin-top:4px;">{html.escape(detail)}</div>'
            f'  </div>'
            f'</div>'
        )

    brain_detail = (
        f"Latest result: {brain_row.get('result', 'Available')}"
        if brain_row
        else "Brain imaging assessment has not been completed for this patient case."
    )
    speech_detail = (
        f"Latest result: {speech_row.get('result', 'Available')}"
        if speech_row
        else "Speech and language assessment has not been completed for this patient case."
    )

    st.markdown(
        f'<div class="assess-status-grid">'
        f'{_card(brain_ok,  "Brain Imaging Analysis",      brain_detail)}'
        f'{_card(speech_ok, "Speech & Language Assessment", speech_detail)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_combined_report_downloads(report_path: Path, key_prefix: str = "combined") -> None:
    formats = report_sibling_files(report_path)
    if not formats:
        note("No downloadable combined report is available yet.", "info")
        return

    with st.container(border=True):
        cols = st.columns(3)
        order = [
            ("html", "Download Combined HTML", "text/html"),
            ("pdf", "Download Combined PDF", "application/pdf"),
            ("md", "Download Combined Markdown", "text/markdown"),
        ]
        for col, (ext, label, mime) in zip(cols, order):
            path = formats.get(ext)
            if path and Path(path).exists():
                with Path(path).open("rb") as handle:
                    col.download_button(
                        label,
                        data=handle.read(),
                        file_name=report_download_filename(Path(path)),
                        mime=mime,
                        key=f"dl_{key_prefix}_{ext}_{Path(path).name}",
                        use_container_width=True,
                    )
            else:
                col.button(
                    label,
                    key=f"dl_{key_prefix}_{ext}_missing",
                    use_container_width=True,
                    disabled=True,
                )


def render_combined_multimodal_report_section(patient_id: str) -> None:
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    section(
        "Integrated Patient Summary",
        "Consolidate brain imaging and speech-language outputs into a structured clinical summary.",
    )
    note(
        "Brain imaging analysis and speech-language assessment are independent. "
        "This report summarizes both outputs but does not combine them into a medical diagnosis.",
        "info",
    )

    if not db_available():
        note(
            "Patient history is currently unavailable. Please check the database connection and try again.",
            "error",
        )
        return

    patient_id = patient_identifier(patient_id)
    brain_row = get_latest_brain_analysis(patient_id)
    speech_row = get_latest_speech_analysis(patient_id)
    render_combined_status_cards(brain_row, speech_row)

    if not brain_row and not speech_row:
        note("No assessment result is available for this patient case yet.", "info")
        generate_disabled = True
    else:
        generate_disabled = False
        if not brain_row:
            note(
                "Brain imaging assessment has not been completed for this patient case. "
                "A partial integrated summary can be generated with the available assessment.",
                "warning",
            )
        if not speech_row:
            note(
                "Speech and language assessment has not been completed for this patient case. "
                "A partial integrated summary can be generated with the available assessment.",
                "warning",
            )

    if st.button(
        "Generate Integrated Patient Summary",
        key="generate_combined_report",
        type="primary",
        disabled=generate_disabled,
        use_container_width=True,
    ):
        try:
            with st.spinner("Generating combined report..."):
                case_data = build_combined_case_data(patient_id)
                rag_summary = generate_combined_rag_summary(case_data)
                paths = save_combined_multimodal_report(case_data, rag_summary)
            st.session_state.combined_report_path = paths.get("md")
            st.session_state.combined_report_html_path = paths.get("html")
            st.session_state.combined_report_pdf_path = paths.get("pdf")
            note("Integrated Patient Summary generated successfully.", "success")
        except Exception as exc:
            show_error("Combined report could not be generated. Please try again.", exc)

    medical_note_card(
        "This report is not a medical diagnosis. The results must be reviewed by a qualified healthcare professional."
    )
    current_combined_path = st.session_state.get("combined_report_path")
    if current_combined_path and Path(current_combined_path).exists():
        render_combined_report_downloads(Path(current_combined_path), key_prefix="combined_current")

    current_path_resolved = Path(current_combined_path).resolve() if current_combined_path else None
    existing_combined = [
        path for path in combined_report_paths(patient_id)
        if current_path_resolved is None or path.resolve() != current_path_resolved
    ]
    if existing_combined:
        st.markdown("##### Generated combined reports")
        for index, report_path in enumerate(existing_combined, start=1):
            meta = report_card_metadata(report_path)
            report_card({
                "kind": "Combined Multimodal",
                "title": "Combined Multimodal report",
                "date": meta["date"],
            })
            render_combined_report_downloads(report_path, key_prefix=f"combined_existing_{index}")


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def source_to_google_link(source):
    source_name = Path(str(source)).name
    title = (
        Path(source_name).stem
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )
    title = patient_friendly_source_title(title)
    url = "https://www.google.com/search?q=" + quote_plus(title + " PDF")
    return title, url


def patient_friendly_source_title(title: str) -> str:
    text = str(title or "medical source").strip() or "medical source"
    replacements = {
        "gradcam": "visual explanation",
        "grad cam": "visual explanation",
        "3d cam": "3D visual explanation",
        "xai": "explainable AI",
    }
    for old, new in replacements.items():
        text = re.sub(old, new, text, flags=re.IGNORECASE)
    return text


def sources_markdown(sources):
    lines = []
    seen = set()

    for item in sources:
        source = item.get("source", "unknown") if isinstance(item, dict) else item
        title, url = source_to_google_link(source)
        key = title.casefold()

        if title and key not in seen:
            lines.append(f"- [{title}]({url})")
            seen.add(key)

    return "\n".join(lines) if lines else "- No medical sources were retrieved."


def mri_sources_markdown(sources) -> str:
    lines = []
    seen = set()

    for item in sources or []:
        if isinstance(item, dict):
            raw_title = str(
                item.get("title") or item.get("source") or ""
            ).strip()
            if not raw_title and not str(item.get("url") or "").strip():
                continue
            title = patient_friendly_source_title(raw_title or "Medical source")
            url = str(item.get("url") or "").strip()
        else:
            if not str(item or "").strip():
                continue
            title, url = source_to_google_link(item)

        if not title:
            title = "Medical source"
        if not url:
            url = "https://www.google.com/search?q=" + quote_plus(title + " MRI Alzheimer PDF")

        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{title}]({url})")

    return "\n".join(lines) if lines else (
        "This response was generated from the current assessment result. "
        "No external reference document was retrieved for this question."
    )


def render_mri_sources(sources) -> None:
    if not sources:
        note(
            "This response was generated from the current assessment result. "
            "No external reference document was retrieved for this question.",
            "info",
        )
        return

    with st.expander("Reference sources"):
        st.markdown(mri_sources_markdown(sources))


def has_retrieved_sources(sources) -> bool:
    for source in sources or []:
        if isinstance(source, dict):
            if any(str(source.get(key) or "").strip() for key in ("title", "source", "url")):
                return True
        elif str(source or "").strip():
            return True
    return False


def brain_result_interpretation(result: dict) -> str:
    prediction = str((result or {}).get("prediction", "")).strip()
    if prediction == "CN":
        return (
            "The brain imaging assessment classified this scan as a Cognitively "
            "Normal Profile. This means that the system did not detect a brain "
            "scan pattern strongly associated with Alzheimer's disease in this image."
        )
    if prediction == "AD":
        return (
            "The brain imaging assessment classified this scan as compatible with "
            "an Alzheimer's disease profile. This means the current image pattern "
            "may require clinical review, but it does not confirm a diagnosis."
        )
    return (
        "The brain imaging assessment produced a decision-support output that "
        "should be interpreted in the full clinical context."
    )


def brain_next_steps_answer(result: dict) -> str:
    return (
        "**Current interpretation:**\n\n"
        f"{brain_result_interpretation(result)}\n\n"
        "**Recommended next clinical steps:**\n\n"
        "1. Review the result together with the patient's clinical history, "
        "symptoms, cognitive complaints, and neurological examination.\n"
        "2. Compare the imaging result with cognitive screening tests, previous "
        "imaging, laboratory findings, or specialist evaluation if available.\n"
        "3. If symptoms are present or progressing, consider referral to a "
        "neurologist or memory clinic for further assessment.\n"
        "4. If there are no concerning symptoms, document the result as a "
        "decision-support finding and continue routine clinical follow-up if "
        "clinically appropriate.\n\n"
        "**Important note:**\n\n"
        "This result is not a medical diagnosis. It should support, not replace, "
        "clinical judgment by a qualified healthcare professional."
    )


def sanitize_clinical_interpretation_answer(answer: str, result: dict) -> str:
    """
    Remove non-professional or medically unsafe phrasing from panel responses.
    """
    text = strip_chatbot_preamble(str(answer or "")).strip()
    text = patient_friendly_text(text)
    if not text:
        return (
            "**Current interpretation:**\n\n"
            f"{brain_result_interpretation(result)}\n\n"
            "**Important note:**\n\n"
            "This result is not a medical diagnosis. It should support, not replace, "
            "clinical judgment by a qualified healthcare professional."
        )

    forbidden_patterns = (
        r"\byour overall mental activity\b.*",
        r"\bhealthy thoughts and feelings\b.*",
        r"\bwe['’]?ll discuss\b.*",
        r"\bwe will discuss\b.*",
        r"\bthe ai has identified\b.*",
        r"\bthe ai\b",
        r"\bhealthy\b",
        r"\bthoughts?\b",
        r"\bfeelings?\b",
    )
    for pattern in forbidden_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    replacements = {
        "AI": "the decision-support system",
        "artificial intelligence": "the decision-support system",
        "the model": "the brain imaging assessment",
        "the system confirms": "the current result suggests",
        "confirms Alzheimer's disease": "requires clinical review",
        "the patient has Alzheimer's disease": "the result requires clinical review",
    }
    for old, new in replacements.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line.strip()]
    text = "\n\n".join(cleaned_lines).strip()

    if "medical diagnosis" not in text.lower():
        text += (
            "\n\n**Important note:**\n\n"
            "This result is not a medical diagnosis. It should support, not replace, "
            "clinical judgment by a qualified healthcare professional."
        )

    return text


def build_brain_panel_answer(question: str, result: dict, rag: dict) -> tuple[str, list]:
    """
    Build a safe answer for the Brain Imaging Clinical Interpretation Panel.
    RAG sources are displayed only when actually returned by the retrieval step.
    """
    normalized_question = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    sources = rag.get("sources", []) if isinstance(rag, dict) else []

    if "recommended" in normalized_question and "next" in normalized_question and "step" in normalized_question:
        return brain_next_steps_answer(result), sources

    raw_answer = str((rag or {}).get("answer", "")).strip() if isinstance(rag, dict) else ""
    unavailable = not raw_answer or raw_answer == "MRI explanation is currently unavailable."
    if unavailable:
        answer = (
            "**Current interpretation:**\n\n"
            f"{brain_result_interpretation(result)}\n\n"
            "**Clinical review:**\n\n"
            "This output should be reviewed together with symptoms, clinical history, "
            "neurological examination, and any available cognitive testing or previous imaging.\n\n"
            "**Important note:**\n\n"
            "This result is not a medical diagnosis. It should support, not replace, "
            "clinical judgment by a qualified healthcare professional."
        )
    else:
        answer = sanitize_clinical_interpretation_answer(raw_answer, result)

    return answer, sources


def safe_generate_with_llm(prompt: str) -> str:
    try:
        llm_generator = importlib.import_module("llm_generator")

        return llm_generator.generate_with_llm(prompt).strip()
    except Exception:
        return (
            "The local language explanation is currently unavailable. The result "
            "and retrieved medical context can still be reviewed by a "
            "healthcare professional."
        )


def get_vectorstore():
    global _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is not None:
        return _VECTORSTORE_CACHE

    rag_explainer = importlib.import_module("rag_explainer")
    _VECTORSTORE_CACHE = rag_explainer.load_vectorstore()

    return _VECTORSTORE_CACHE


def retrieve_rag_context(question, transcript, prediction, k=4):
    vectorstore = get_vectorstore()
    query = f"""
    Patient prediction: {prediction}

    Patient transcript:
    {transcript}

    User question:
    {question}

    Retrieve medical evidence about Alzheimer's disease, dementia, language
    impairment, pauses, hesitations, repetitions, word-finding difficulties,
    reduced information content, discourse organization, speech biomarkers,
    and NLP-based dementia detection.
    """
    docs = vectorstore.similarity_search(query, k=k)

    context_parts = []
    sources = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        text = " ".join(doc.page_content.replace("\n", " ").split())
        context_parts.append(
            f"Document {index}\n"
            f"Source: {Path(str(source)).name}\n"
            f"Content: {text[:1200]}"
        )
        sources.append(source)

    return "\n\n".join(context_parts), sources


def prediction_display(prediction):
    if prediction == "ProbableAD":
        return "ProbableAD"
    if prediction == "Control":
        return "Control"
    return str(prediction)


def cnn_explanation(prediction: str) -> str:
    if prediction == "CN":
        return (
            "The brain scan analysis classified this scan as a cognitively normal "
            "profile. CN means Cognitively Normal profile and is not a brain "
            "region. The visual explanation highlights the image areas that "
            "influenced this result."
        )

    if prediction == "AD":
        return (
            "The brain scan analysis classified this scan as compatible with an "
            "Alzheimer's disease profile. This does not confirm Alzheimer's disease. "
            "The visual explanation highlights the image areas that influenced "
            "this result."
        )

    return (
        "The brain scan analysis was completed. The visual explanation highlights the "
        "image areas that influenced the result."
    )


def mri_prediction_meaning(prediction: str) -> str:
    if prediction == "CN":
        return "Cognitively Normal profile"
    if prediction == "AD":
        return "Alzheimer's Disease compatible profile"
    return "Unknown profile"


def cnn_patient_summary(prediction: str) -> str:
    if prediction == "CN":
        return (
            "In simple terms, the system did not detect a brain scan pattern strongly "
            "associated with Alzheimer's disease in this scan. However, this result "
            "must always be checked by a medical specialist."
        )

    if prediction == "AD":
        return (
            "In simple terms, the system detected a brain scan pattern that may be "
            "associated with Alzheimer's disease. This does not confirm the disease "
            "by itself and must be reviewed by a medical specialist."
        )

    return (
        "In simple terms, the system analyzed the brain scan and produced a "
        "decision-support result that must be reviewed by a medical specialist."
    )


def mri_general_explanation(prediction: str) -> str:
    if prediction == "CN":
        return (
            "The brain scan analysis classified this scan as a Cognitively Normal "
            "profile. The confidence score shows how sure the system was about "
            "this result. The visual explanation highlights the areas of the image "
            "that influenced the result. Blue or dark colors indicate lower influence, "
            "while yellow or red colors indicate stronger influence. These colors do "
            "not mean that these areas are diseased.\n\n"
            f"{MEDICAL_NOTE}"
        )

    if prediction == "AD":
        return (
            "The brain scan analysis classified this scan as compatible with an "
            "Alzheimer's disease profile. This does not mean that the patient has "
            "Alzheimer's disease. It means that the system detected a brain scan "
            "pattern that may be associated with Alzheimer's disease. The visual "
            "explanation highlights the areas that influenced the result.\n\n"
            f"{MEDICAL_NOTE}"
        )

    return (
        "The brain scan analysis produced a decision-support result. The visual "
        "explanation highlights the areas of the image that influenced the result. "
        f"{MEDICAL_NOTE}"
    )


def gradcam_prediction_explanation(prediction: str) -> str:
    if prediction == "CN":
        return (
            "The system classified this scan as a cognitively normal profile. The "
            "visual explanation images show the highlighted areas that supported this "
            "classification."
        )

    if prediction == "AD":
        return (
            "The system classified this scan as compatible with an Alzheimer's "
            "disease profile. The visual explanation images show the highlighted areas "
            "that influenced this classification."
        )

    return (
        "The visual explanation images show the highlighted areas that influenced the "
        "classification."
    )


def current_gradcam_info() -> dict:
    paths = valid_gradcam_items(st.session_state.get("gradcam_paths") or [])
    orientation = visual_explanation_orientation_label(
        paths,
        st.session_state.get("gradcam_generated_orientation") or "Not generated",
    )
    return {
        "orientation": orientation,
        "display_mode": visual_explanation_display_mode_label(
            paths,
            st.session_state.get("gradcam_generated_display_mode") or "Not generated",
        ),
        "number_of_slices": visual_explanation_slice_count(paths),
        "status": "Generated" if paths else "Not generated",
        "interpretation": (
            "Blue or dark colors indicate lower influence; yellow or red colors "
            "indicate stronger influence on the result. "
            "Highlighted areas are not automatically diseased areas."
        ),
    }


def write_updated_cnn_report(result: dict, patient_id: str) -> None:
    report_path_value = st.session_state.get("cnn_report_path")
    if not report_path_value:
        return

    analysis_id = st.session_state.get("cnn_analysis_id") or generate_analysis_id()
    st.session_state.cnn_analysis_id = analysis_id
    gradcam_paths = valid_gradcam_items(st.session_state.get("gradcam_paths") or [])

    paths = save_brain_scan_reports(
        result=result,
        patient_case_id=patient_id,
        analysis_id=analysis_id,
        gradcam_paths=gradcam_paths,
        gradcam_orientation=visual_explanation_orientation_label(
            gradcam_paths,
            st.session_state.get("gradcam_generated_orientation"),
        ),
        mri_preview_path=st.session_state.get("cnn_preview_path"),
        mri_source_name=st.session_state.get("mri_source_name"),
        mri_rag_explanation=st.session_state.get("mri_rag_explanation"),
        mri_rag_sources=st.session_state.get("mri_rag_sources"),
        preprocessing_note=st.session_state.get("mri_preprocessing_note"),
    )
    st.session_state.cnn_report_path = paths["md"]
    st.session_state.cnn_report_html_path = paths["html"]
    st.session_state.cnn_report_pdf_path = paths["pdf"]
    db_saved = save_brain_result_to_database(
        patient_case_id=patient_id,
        analysis_id=analysis_id,
        result=result,
        uploaded_file_name=st.session_state.get("mri_source_name"),
        mri_file_path=st.session_state.get("mri_path"),
        report_paths=paths,
        save_report_entry=False,
    )
    warn_database_not_saved(db_saved)


def required_nlp_explanation(prediction: str, confidence: float) -> str:
    if prediction == "Control":
        return (
            f"The system classified this case as a Control profile with a "
            f"confidence score of {percent(confidence)}. This means that, based "
            "on the patient transcript and the extracted speech features, the "
            "system did not detect enough language signs associated with a "
            "probable Alzheimer's disease profile."
        )

    if prediction == "ProbableAD":
        return (
            "The system detected language patterns that may be associated "
            "with cognitive decline, such as hesitations, repetitions, "
            "corrections, pauses, reduced information content, or difficulties "
            "organizing speech."
        )

    return "The system produced a speech and language analysis result for professional review."


def is_prompt_like_clinical_note(text: str) -> bool:
    lowered = str(text or "").lower()
    prompt_markers = (
        "task_nlp_cha",
        "now execute",
        "do not continue",
        "search / replace",
        "trained nlp model",
        "tokenizer",
        "scaler",
        "model weights",
        "after finishing",
        "codex",
    )
    return any(marker in lowered for marker in prompt_markers)


def sanitize_clinical_notes(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned or is_prompt_like_clinical_note(cleaned):
        return "No clinical notes provided."
    return cleaned


def build_patient_info(
    patient_name,
    patient_id,
    study_date,
    age,
    sex_label,
    education_years,
    clinician,
    clinical_notes,
    analysis_id="",
):
    return {
        "Patient Name": display_value(patient_name),
        "Patient Case ID": display_value(patient_id),
        "Analysis ID": display_value(analysis_id),
        "Study Date": display_value(study_date),
        "Age": display_value(age),
        "Sex": display_value(sex_label),
        "Years of Education": display_value(education_years),
        "Responsible Clinician": display_value(clinician),
        "Clinical Notes": sanitize_clinical_notes(clinical_notes),
    }


def build_analysis_features(
    n_filled_pauses,
    n_phon_fragments,
    n_paralinguistic,
    n_retracings,
    n_unintelligible,
    n_pauses,
    entryage,
    sex,
    educ,
):
    return {
        "Filled pauses": n_filled_pauses,
        "Phonological fragments": n_phon_fragments,
        "Paralinguistic markers": n_paralinguistic,
        "Repetitions / corrections": n_retracings,
        "Unintelligible words": n_unintelligible,
        "Pauses": n_pauses,
        "Age": entryage,
        "Sex": "Female" if sex == 0 else "Male",
        "Years of education": educ,
    }


CHA_FEATURE_ORDER = [
    "n_filled_pauses",
    "n_phon_fragments",
    "n_paralinguistic",
    "n_retracings",
    "n_unintelligible",
    "n_pauses",
    "entryage",
    "sex",
    "educ",
]

CHA_FEATURE_LABELS = {
    "n_filled_pauses": "Hesitation words",
    "n_phon_fragments": "Interrupted words",
    "n_paralinguistic": "Non-verbal speech markers",
    "n_retracings": "Self-corrections",
    "n_unintelligible": "Unclear words",
    "n_pauses": "Pauses",
    "entryage": "Age",
    "sex": "Sex",
    "educ": "Years of education",
}


def read_cha_file(uploaded_file) -> str:
    """Read an uploaded CHAT transcript with UTF-8 and latin-1 fallback."""
    if uploaded_file is None:
        return ""

    data = uploaded_file.getvalue()
    if not data:
        return ""

    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def extract_participant_transcript(cha_text: str) -> str:
    """
    Extract the patient/participant utterances from a CHAT transcript.

    The preferred speaker code is *PAR:. If it is not present, common patient
    speaker codes are used as a conservative fallback.
    """
    if not cha_text:
        return ""

    def collect_for_codes(codes: set[str]) -> str:
        collected = []
        keep_continuation = False

        for raw_line in cha_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("@") or line.startswith("%"):
                keep_continuation = False
                continue

            match = re.match(r"^\*([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
            if match:
                speaker = match.group(1).upper()
                keep_continuation = speaker in codes
                if keep_continuation:
                    collected.append(match.group(2).strip())
                continue

            if keep_continuation and not line.startswith("*"):
                collected.append(line)

        return " ".join(part for part in collected if part)

    participant_text = collect_for_codes({"PAR"})
    if participant_text.strip():
        return participant_text.strip()

    return collect_for_codes({"PAR", "PAT", "CHI", "SUB", "P"}).strip()


def clean_cha_transcript(raw_transcript: str) -> str:
    """Remove CHAT annotations while keeping readable speech text."""
    text = str(raw_transcript or "")
    if not text.strip():
        return ""

    text = re.sub(r"\x15[^\x15]*\x15", " ", text)
    text = re.sub(r"\b\d+_\d+\b", " ", text)
    text = re.sub(r"&-(um|uh)\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"&\+([A-Za-z]+)", r"\1", text)
    text = re.sub(r"&=\w+", " ", text)
    text = re.sub(r"\[\s*(?:/|//|x\s*\d+)\s*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[=![^\]]*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\((?:\s*\.+\s*|\s*\d+(?:\.\d+)?\s*)\)", " ", text)
    text = re.sub(r"\b(?:xxx|yyy|www)\b", " unclear ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b0\b", " ", text)
    text = re.sub(r"[<>+/#=_~^]", " ", text)
    text = re.sub(r"\b([A-Za-z]+)-\s+", r"\1 ", text)
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n-")


def _metadata_line_value(cha_text: str, names: tuple[str, ...]) -> str:
    for name in names:
        match = re.search(rf"(?im)^@{re.escape(name)}\s*:\s*(.+)$", cha_text)
        if match:
            return match.group(1).strip()
    return ""


def _participant_id_fields(cha_text: str) -> list[list[str]]:
    fields_by_line = []
    for raw_line in cha_text.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("@id:"):
            continue
        value = line.split(":", 1)[1].strip()
        fields = [part.strip() for part in value.split("|")]
        fields_by_line.append(fields)

    preferred = [
        fields for fields in fields_by_line
        if len(fields) > 2 and fields[2].upper() in {"PAR", "PAT", "CHI", "SUB", "P"}
    ]
    return preferred or fields_by_line


def _parse_number(value: str, max_value: float | None = None) -> float | None:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    number = float(match.group(0))
    if max_value is not None and number > max_value:
        return None
    return number


def _parse_age_value(value: str) -> float | None:
    if not value:
        return None
    match = re.search(r"\b(\d{1,3})(?:;|\.\d+)?", str(value))
    if not match:
        return None
    age = float(match.group(1))
    return age if 0 < age <= 120 else None


def _parse_sex_value(value) -> int | None:
    # integer 0/1 come directly from cha_parser extraction — must not use `or`
    # because `0 or ""` evaluates to "" and loses the Female value.
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    if text in {"female", "f", "woman", "girl"}:
        return 0
    if text in {"male", "m", "man", "boy"}:
        return 1
    return None


def _extract_age_from_metadata(cha_text: str) -> float | None:
    direct = _metadata_line_value(cha_text, ("Age", "EntryAge"))
    parsed = _parse_age_value(direct)
    if parsed is not None:
        return parsed

    for fields in _participant_id_fields(cha_text):
        if len(fields) > 3:
            parsed = _parse_age_value(fields[3])
            if parsed is not None:
                return parsed
        for field in fields:
            if ";" in field:
                parsed = _parse_age_value(field)
                if parsed is not None:
                    return parsed
    return None


def _extract_sex_from_metadata(cha_text: str) -> int | None:
    direct = _metadata_line_value(cha_text, ("Sex", "Gender"))
    parsed = _parse_sex_value(direct)
    if parsed is not None:
        return parsed

    for fields in _participant_id_fields(cha_text):
        for field in fields:
            parsed = _parse_sex_value(field)
            if parsed is not None:
                return parsed
    return None


def _extract_education_from_metadata(cha_text: str) -> float | None:
    direct = _metadata_line_value(cha_text, ("Education", "Educ"))
    parsed = _parse_number(direct, max_value=40)
    if parsed is not None:
        return parsed

    for fields in _participant_id_fields(cha_text):
        for index in (8, 9, 7):
            if len(fields) > index:
                parsed = _parse_number(fields[index], max_value=40)
                if parsed is not None:
                    return parsed
    return None


def extract_speech_features_from_cha(cha_text: str) -> dict:
    """Extract the 9 speech features expected by the existing NLP model."""
    participant_text = extract_participant_transcript(cha_text)
    source = participant_text or cha_text or ""
    lower_source = source.lower()

    filled_pauses = len(
        re.findall(r"(?<!\w)&-(?:um|uh)\b|\b(?:um|uh|erm|er)\b", lower_source)
    )
    phon_fragments = len(re.findall(r"&\+[A-Za-z]+", source))
    phon_fragments += len(re.findall(r"\b[A-Za-z]+-\s", source))
    paralinguistic = len(
        re.findall(r"&=(?:laugh|cough|sigh|noise|breath|sniff|cry|whisper)\b", lower_source)
    )
    paralinguistic += len(re.findall(r"\[=![^\]]+\]", source, flags=re.IGNORECASE))
    retracings = len(re.findall(r"\[\s*//\s*\]|\[\s*/\s*\]", source))
    unintelligible = len(re.findall(r"\b(?:xxx|yyy|www)\b", lower_source))
    pauses = len(re.findall(r"\((?:\s*\.+\s*|\s*\d+(?:\.\d+)?\s*)\)", source))

    return {
        "n_filled_pauses": int(filled_pauses),
        "n_phon_fragments": int(phon_fragments),
        "n_paralinguistic": int(paralinguistic),
        "n_retracings": int(retracings),
        "n_unintelligible": int(unintelligible),
        "n_pauses": int(pauses),
        "entryage": _extract_age_from_metadata(cha_text),
        "sex": _extract_sex_from_metadata(cha_text),
        "educ": _extract_education_from_metadata(cha_text),
    }


def build_feature_vector_from_cha(features_dict: dict) -> list:
    """Build the model feature vector in the exact training order."""
    normalized: dict[str, int | float] = {}
    missing = []

    for key in CHA_FEATURE_ORDER:
        value = features_dict.get(key)
        if key == "sex":
            parsed_sex = _parse_sex_value(value)
            if parsed_sex is None:
                missing.append(key)
            else:
                normalized[key] = parsed_sex
            continue

        if key in {"entryage", "educ"}:
            if value is None or value == "":
                missing.append(key)
                continue
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                missing.append(key)
            continue

        if value is None or value == "":
            normalized[key] = 0
            continue

        try:
            normalized[key] = int(float(value))
        except (TypeError, ValueError):
            missing.append(key)

    if missing:
        readable = ", ".join(CHA_FEATURE_LABELS.get(key, key) for key in sorted(set(missing)))
        raise ValueError(f"Missing required speech information: {readable}")

    features_dict.update(normalized)
    return [normalized[key] for key in CHA_FEATURE_ORDER]


def speech_word_count(transcript: str) -> int:
    return len(re.findall(r"\b[\w']+\b", transcript or ""))


def speech_feature_display_value(key: str, value) -> str:
    if value is None or value == "":
        return "Missing"
    if key == "sex":
        parsed = _parse_sex_value(value)
        if parsed is None:
            return "Missing"
        return "Female" if parsed == 0 else "Male"
    if key in {"entryage", "educ"}:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.1f}"
    return str(int(value))


def dict_to_markdown(data):
    return "\n".join(f"- **{key}:** {value}" for key, value in data.items())


def speech_features_to_markdown(analysis_features: dict) -> str:
    return "\n".join(
        f"- **{FRIENDLY_SPEECH_FEATURES.get(key, key)}:** {value}"
        for key, value in (analysis_features or {}).items()
    )


def observed_linguistic_signs(analysis_features):
    feature_meanings = {
        "Filled pauses": "hesitation markers in the speech sample",
        "Phonological fragments": "partial or interrupted word productions",
        "Paralinguistic markers": "non-word vocal or speech-related markers",
        "Repetitions / corrections": "repeated words or self-corrections",
        "Unintelligible words": "words that could not be clearly understood",
        "Pauses": "silent breaks or interruptions in speech flow",
        "Age": "patient age used as a clinical feature",
        "Sex": "patient sex used as a clinical feature",
        "Years of education": "education level used as a clinical feature",
    }
    return "\n".join(
        (
            f"- **{FRIENDLY_SPEECH_FEATURES.get(name, name)}:** {value}. "
            f"This represents {feature_meanings.get(name, 'a recorded feature')}."
        )
        for name, value in analysis_features.items()
    )


def sanitize_llm_explanation(text: str, fallback: str) -> str:
    if not text:
        return fallback

    forbidden_labels = [
        "SIMPLE_EXPLANATION:",
        "SIMPLE_EXPLANATION",
        "OBSERVED_LINGUISTIC_SIGNS:",
        "OBSERVED_LINGUISTIC_SIGNS",
        "OBSERVE_LINGUISTIC_SIGNS:",
        "OBSERVE_LINGUISTIC_SIGNS",
        "MODEL_OUTPUT:",
        "MODEL_OUTPUT",
    ]
    forbidden_content_terms = [
        "cookie",
        "cookies",
        "boy",
        "girl",
        "mother",
        "water",
        "sink",
        "laughter",
        "running water",
    ]
    forbidden_persona_patterns = [
        r"\bmy name is\b",
        r"\bi am\b",
        r"\bi'm\b",
        r"\bi was\b",
        r"\bi have\b",
        r"\bdr\.?\s+[A-Z][A-Za-z]+",
        r"\bdoctor\s+[A-Z][A-Za-z]+",
    ]

    clean_text = text.strip()
    for label in forbidden_labels:
        clean_text = clean_text.replace(label, "").strip()

    clean_text = "\n".join(
        line.strip()
        for line in clean_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    if any(term in clean_text.lower() for term in forbidden_content_terms):
        return fallback
    if any(re.search(pattern, clean_text, flags=re.IGNORECASE) for pattern in forbidden_persona_patterns):
        return fallback

    return clean_text or fallback


def _speech_pattern_count(analysis_features: dict, *names: str) -> int:
    for name in names:
        if name in analysis_features:
            try:
                return int(float(analysis_features.get(name) or 0))
            except (TypeError, ValueError):
                return 0
    return 0


def _count_phrase(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return f"one {singular}"
    return f"{count} {plural or singular + 's'}"


def speech_patterns_summary_sentence(analysis_features: dict) -> str:
    pattern_specs = [
        ("hesitation word", "hesitation words", ("Filled pauses", "Hesitation words", "n_filled_pauses")),
        ("interrupted word", "interrupted words", ("Phonological fragments", "Interrupted words", "n_phon_fragments")),
        ("non-verbal speech marker", "non-verbal speech markers", ("Paralinguistic markers", "Non-verbal speech markers", "n_paralinguistic")),
        ("self-correction", "self-corrections", ("Repetitions / corrections", "Self-corrections", "n_retracings")),
        ("unclear word", "unclear words", ("Unintelligible words", "Unclear words", "n_unintelligible")),
        ("pause", "pauses", ("Pauses", "n_pauses")),
    ]

    present = []
    absent = []
    for singular, plural, names in pattern_specs:
        count = _speech_pattern_count(analysis_features, *names)
        if count > 0:
            present.append(_count_phrase(count, singular, plural))
        else:
            absent.append(plural)

    if present and absent:
        return (
            f"{', '.join(present)}, with no detected "
            f"{', '.join(absent[:-1])}{', or ' if len(absent) > 1 else ''}{absent[-1]}"
        )
    if present:
        return ", ".join(present)
    return (
        "no detected hesitation words, interrupted words, non-verbal speech "
        "markers, self-corrections, unclear words, or pauses"
    )


def neutral_speech_simple_explanation(result: dict, analysis_features: dict) -> str:
    prediction = prediction_display(result.get("prediction", ""))
    confidence_text = percent(result.get("confidence", 0.0))
    patterns = speech_patterns_summary_sentence(analysis_features)
    return (
        "This speech and language analysis is based on the uploaded picture "
        f"description transcript. The analysis result is {prediction} with a "
        f"confidence score of {confidence_text}. The extracted speech patterns "
        f"show {patterns}. These observations can support clinical review, but "
        "they do not represent a medical diagnosis."
    )


def build_nlp_llm_explanation(result, analysis_features):
    return neutral_speech_simple_explanation(result, analysis_features)


def build_chat_answer(question, transcript, prediction, confidence):
    context, sources = retrieve_rag_context(
        question=question,
        transcript=transcript,
        prediction=prediction,
        k=4,
    )
    source_links = sources_markdown(sources)

    prompt = f"""
You are a patient-friendly medical explanation assistant.

Rules:
- Answer in English.
- Do not provide a final medical diagnosis.
- Use only the retrieved medical context.
- Do not invent information.
- Do not mention FAISS, embeddings, chunks, local file paths, or page numbers.
- The transcript may come from a picture-description task. The image content
  is only the task used to collect speech.
- Do not interpret picture content as a medical symptom.
- Do not invent psychological or emotional interpretations.
- Focus only on language signs such as filled pauses, hesitations,
  repetitions, corrections, phonological fragments, unintelligible words,
  pauses, reduced information content, discourse organization, and
  word-finding difficulties.
- Do not combine this speech result with any brain scan result.

Patient transcript:
{transcript}

Result:
{prediction_display(prediction)}

Confidence score:
{percent(confidence)}

User question:
{question}

Retrieved medical context:
{context}

Answer clearly and simply.
"""
    answer = safe_generate_with_llm(prompt)
    answer = patient_friendly_text(answer)
    return (
        f"{answer}\n\n*This is not a medical diagnosis. Please review the result "
        f"with a healthcare professional.*\n\n**Information Sources**\n{source_links}"
    )


def generate_patient_report(
    result,
    patient_info,
    analysis_features,
    llm_explanation,
    transcript: str = "",
):
    prediction = result["prediction"]
    confidence = result["confidence"]
    source_links = sources_markdown(result.get("sources", []))
    simple_explanation = sanitize_llm_explanation(
        llm_explanation,
        required_nlp_explanation(prediction, confidence),
    )

    title = "# Patient Report"
    patient_name = patient_info.get("Patient Name", "Not provided")
    if patient_name != "Not provided":
        title += f" - {patient_name}"
    uploaded_file = Path(str(patient_info.get("Uploaded File", "Not provided"))).name
    report_patient_info = {
        "Patient Case ID": patient_info.get("Patient Case ID", "Not provided"),
        "Analysis ID": patient_info.get("Analysis ID", "Not provided"),
        "Patient Name": patient_info.get("Patient Name", "Not provided"),
        "Study Date": patient_info.get("Study Date", "Not provided"),
        "Responsible Clinician": patient_info.get("Responsible Clinician", "Not provided"),
        "Clinical Notes": sanitize_clinical_notes(patient_info.get("Clinical Notes", "")),
    }

    report_text = f"""{title}

## Patient Information

{dict_to_markdown(report_patient_info)}

## Uploaded Speech File

{uploaded_file}

## Extracted Speech Text

{transcript.strip() or "No extracted speech text was available."}

## Automatically Extracted Speech Information

{speech_features_to_markdown(analysis_features)}

## Speech Analysis Result

- **Result:** {prediction_display(prediction)}
- **Confidence:** {percent(confidence)}

## Simple Explanation

{simple_explanation}

## Medical Note

The transcript was automatically extracted and cleaned from the uploaded speech transcript file. This result is not a medical diagnosis and must be reviewed by a healthcare professional.
"""
    return clean_report_text(report_text)


def generate_cnn_report(
    result: dict,
    patient_id: str,
    gradcam_orientation: str | None = None,
    gradcam_slice_count: int = 0,
    gradcam_paths: list[str | Path] | None = None,
    mri_preview_path: str | Path | None = None,
    mri_source_name: str | None = None,
    mri_rag_explanation: str | None = None,
    mri_rag_sources: list | None = None,
    analysis_id: str = "",
    preprocessing_note: str | None = None,
) -> str:
    prediction = str(result.get("prediction", "Not provided"))
    prediction_meaning = mri_prediction_meaning(prediction)
    confidence = float(result.get("confidence", 0.0))
    prob_cn = float(result.get("prob_cn", 0.0))
    prob_ad = float(result.get("prob_ad", 0.0))
    orientation = gradcam_orientation or "Not generated"
    slice_count = gradcam_slice_count if gradcam_slice_count else "Not generated"
    display_mode = visual_explanation_display_mode_label(
        gradcam_paths or [],
        "Not generated",
    )
    uploaded_scan_name = Path(str(mri_source_name or "Not provided")).name
    preprocessing_note = (
        str(preprocessing_note).strip()
        if str(preprocessing_note or "").strip()
        else "The uploaded scan was used as already prepared for analysis."
    )
    preview_section = ""
    if mri_preview_path and Path(mri_preview_path).exists():
        preview_section = (
            "\n## MRI Preview Image\n\n"
            f"![MRI Preview]({report_image_reference(mri_preview_path)})\n"
        )

    gradcam_section = ""
    valid_items = []
    for item in gradcam_paths or []:
        image_path = gradcam_item_path(item)
        if image_path and image_path.exists():
            valid_items.append((item, image_path))

    if valid_items:
        lines = ["\n## Visual Explanation Images\n"]
        for index, (item, path) in enumerate(valid_items, start=1):
            lines.append(f"### {gradcam_item_caption(item, index, orientation)}")
            lines.append(f"![Slice {index}]({report_image_reference(path)})")
        gradcam_section = "\n".join(lines) + "\n"
    elif gradcam_paths:
        gradcam_section = (
            "\n## Visual Explanation Images\n\n"
            "Visual explanation image was not available for this report.\n"
        )

    mri_rag_text = (
        str(mri_rag_explanation).strip()
        if str(mri_rag_explanation or "").strip()
        else "Brain scan explanation was not generated."
    )
    mri_source_links = mri_sources_markdown(mri_rag_sources)

    report_text = f"""# Brain Scan Report - {patient_identifier(patient_id)}

## Patient Information

- **Patient Case ID:** {patient_identifier(patient_id)}
- **Analysis ID:** {analysis_id or "Not available"}

## Analysis Details

- **Date and Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Analysis Type:** Brain Scan Analysis

## Uploaded Brain Scan File

{uploaded_scan_name}

## Scan Preparation

{preprocessing_note}

## Analysis Result

The result is: **{prediction_meaning}**.

## Confidence Score

The confidence score is **{percent(confidence)}**.

## Probability Details

- **Probability Cognitively Normal profile:** {percent(prob_cn)}
- **Probability Alzheimer's Disease compatible profile:** {percent(prob_ad)}

## Visual Explanation Details

- **Visual explanation views:** {orientation}
- **Visual display mode:** {display_mode}
- **Displayed image slices:** {slice_count}

## Simple Explanation

{cnn_explanation(prediction)}

## Visual Explanation

The colored areas show which parts of the brain scan influenced the result. They do not mean that these areas are diseased.

- **Blue / dark:** lower influence
- **Yellow / red:** stronger influence
- **Highlighted areas:** do not confirm disease
{preview_section}{gradcam_section}

## Simple Brain Scan Explanation

{mri_rag_text}

## Information Sources

{mri_source_links}

## Important Medical Note

This visual explanation must be reviewed by a healthcare professional. It does not confirm disease.
"""
    return clean_report_text(report_text)


FRIENDLY_SPEECH_FEATURES = {
    "n_filled_pauses": "Hesitation words",
    "n_phon_fragments": "Interrupted words",
    "n_paralinguistic": "Non-verbal speech markers",
    "n_retracings": "Self-corrections",
    "n_unintelligible": "Unclear words",
    "n_pauses": "Pauses",
    "entryage": "Age",
    "sex": "Sex",
    "educ": "Years of education",
    "Filled pauses": "Hesitation words",
    "Phonological fragments": "Interrupted words",
    "Paralinguistic markers": "Non-verbal speech markers",
    "Repetitions / corrections": "Self-corrections",
    "Unintelligible words": "Unclear words",
    "Pauses": "Pauses",
    "Age": "Age",
    "Sex": "Sex",
    "Years of education": "Years of education",
}


def report_basename(kind: str, patient_case_id: str, analysis_id: str) -> str:
    prefix = "brain_scan_report" if kind == "brain" else "speech_language_report"
    case = str(patient_case_id or "").strip() or "PAT-UNKNOWN"
    analysis = str(analysis_id or "").strip() or generate_analysis_id()
    return f"{prefix}_{case}_{analysis}"


def _image_data_uri(image_path: str | Path) -> str | None:
    try:
        raw = Path(image_path).read_bytes()
        suffix = Path(image_path).suffix.lower().lstrip(".") or "png"
        if suffix == "jpg":
            suffix = "jpeg"
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:image/{suffix};base64,{encoded}"
    except Exception:
        return None


def _report_html_shell(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg1: #F8F4EC; --bg2: #E7F6EF; --bg3: #F3E8FF;
    --card: #FFFDF8;
    --primary: #0F9D7A; --deep: #096B5A;
    --violet: #8B5CF6; --coral: #F97362; --gold: #C99A2E;
    --text: #202124; --muted: #6B7280; --line: #ECE7DB;
    --warn-bg: #FFF7E2; --warn-border: #C99A2E;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 30px 16px 64px;
    background:
      radial-gradient(900px 480px at 88% -6%, rgba(139,92,246,0.10), transparent 60%),
      radial-gradient(820px 460px at -6% 2%, rgba(15,157,122,0.10), transparent 58%),
      linear-gradient(135deg, var(--bg1), var(--bg2) 55%, var(--bg3));
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    line-height: 1.62;
  }}
  .page {{ max-width: 880px; margin: 0 auto; }}
  .card {{
    background: rgba(255,253,248,0.92);
    backdrop-filter: blur(6px);
    border: 1px solid var(--line);
    border-radius: 20px; padding: 24px 28px; margin: 16px 0;
    box-shadow: 0 18px 44px -30px rgba(20,30,26,0.55);
  }}
  .header {{
    background: linear-gradient(135deg, #ffffff 0%, #eefaf4 55%, #f3ecff 100%);
    border-left: 6px solid var(--primary);
  }}
  .eyebrow {{
    font-size: 12px; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--deep); margin-bottom: 6px;
  }}
  h1 {{ font-size: 27px; margin: 0 0 4px; }}
  h2 {{ font-size: 17px; margin: 0 0 14px; color: var(--text); }}
  p {{ margin: 0 0 10px; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; margin-top: 16px; }}
  .meta-item {{ background: rgba(255,255,255,0.7); border: 1px solid var(--line); border-radius: 13px; padding: 11px 14px; }}
  .meta-label {{ font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
  .meta-value {{ font-size: 15px; font-weight: 600; margin-top: 3px; word-break: break-word; }}
  .badge {{
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 15px; font-weight: 700; padding: 9px 17px; border-radius: 999px; color: #fff;
  }}
  .badge-good {{ background: linear-gradient(135deg, var(--primary), var(--deep)); }}
  .badge-alert {{ background: linear-gradient(135deg, #f6925f, var(--coral)); }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin-top: 16px; }}
  .stat {{ background: linear-gradient(180deg, #ffffff, #f4faf7); border: 1px solid #d9eee5; border-radius: 15px; padding: 15px; text-align: center; }}
  .stat .v {{ font-size: 21px; font-weight: 800; color: var(--deep); }}
  .stat .l {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .img-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; margin-top: 12px; }}
  .img-cell img {{ width: 100%; border-radius: 13px; border: 1px solid var(--line); display: block; }}
  .img-cap {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
  .legend {{ display: grid; gap: 10px; margin-top: 12px; }}
  .legend-row {{ display: flex; align-items: center; gap: 10px; font-size: 14px; }}
  .dot {{ width: 16px; height: 16px; border-radius: 50%; flex: 0 0 auto; }}
  .dot-low {{ background: linear-gradient(135deg,#1d2f8f,#10131f); }}
  .dot-mid {{ background: linear-gradient(135deg,#2faf70,#ffd965); }}
  .dot-high {{ background: linear-gradient(135deg,#ffef8a,#d6452f); }}
  .feature-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  .feature-table td {{ padding: 9px 12px; border-bottom: 1px solid var(--line); font-size: 14px; }}
  .feature-table td:first-child {{ color: var(--muted); width: 55%; }}
  .feature-table td:last-child {{ font-weight: 600; text-align: right; }}
  .transcript {{ background: rgba(255,255,255,0.7); border: 1px solid var(--line); border-radius: 13px; padding: 14px 16px; font-size: 14px; color: var(--text); white-space: pre-wrap; }}
  .note {{ background: var(--warn-bg); border: 1px solid var(--warn-border); border-left: 6px solid var(--warn-border); border-radius: 15px; padding: 16px 18px; color: #6f5311; }}
  .note strong {{ color: #5a430d; }}
  .footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 22px; }}
  @media (max-width: 620px) {{ .meta-grid, .stat-grid, .img-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="page">
{body}
<div class="footer">Generated on {datetime.now().strftime("%d/%m/%Y %H:%M")} &middot; Decision-support output</div>
</div>
</body>
</html>"""


def _meta_item(label: str, value: str) -> str:
    return (
        '<div class="meta-item">'
        f'<div class="meta-label">{html.escape(label)}</div>'
        f'<div class="meta-value">{html.escape(str(value))}</div>'
        "</div>"
    )


def brain_scan_result_label(prediction: str) -> str:
    if prediction == "CN":
        return "Cognitively Normal Profile"
    if prediction == "AD":
        return "Alzheimer's Disease-Compatible Profile"
    return f"{prediction} — Result for professional review"


def brain_scan_simple_explanation(prediction: str) -> str:
    if prediction == "CN":
        return (
            "The brain scan analysis classified this scan as a Cognitively Normal "
            "profile. This means that the system did not detect a brain scan pattern "
            "strongly associated with Alzheimer's disease in this scan."
        )
    if prediction == "AD":
        return (
            "The brain scan analysis classified this scan as compatible with an "
            "Alzheimer's disease profile. This does not mean that the patient has "
            "Alzheimer's disease. It means that the system detected a brain scan "
            "pattern that may be associated with this condition."
        )
    return (
        "The brain scan analysis produced a decision-support result. This is not a "
        "diagnosis and must be reviewed by a healthcare professional."
    )


def generate_brain_scan_report_html(
    result: dict,
    patient_case_id: str,
    analysis_id: str,
    gradcam_paths: list | None = None,
    gradcam_orientation: str | None = None,
    mri_source_name: str | None = None,
    preprocessing_note: str | None = None,
) -> str:
    prediction = str(result.get("prediction", "Unknown"))
    badge_class = "badge-good" if prediction == "CN" else "badge-alert"
    uploaded_scan_name = Path(str(mri_source_name or "Not provided")).name
    preprocessing_note = (
        str(preprocessing_note).strip()
        if str(preprocessing_note or "").strip()
        else "The uploaded scan was used as already prepared for analysis."
    )

    header = (
        '<div class="card header">'
        '<div class="eyebrow">Brain Scan Analysis</div>'
        "<h1>Brain Scan Report</h1>"
        '<div class="meta-grid">'
        + _meta_item("Patient Case ID", patient_case_id)
        + _meta_item("Analysis ID", analysis_id)
        + _meta_item("Date", datetime.now().strftime("%Y-%m-%d %H:%M"))
        + _meta_item("Analysis Type", "Brain Scan Analysis")
        + "</div></div>"
    )

    result_summary = (
        '<div class="card">'
        "<h2>Result Summary</h2>"
        f'<span class="badge {badge_class}">{html.escape(brain_scan_result_label(prediction))}</span>'
        '<div class="stat-grid">'
        f'<div class="stat"><div class="v">{percent(result.get("confidence", 0.0))}</div>'
        '<div class="l">Confidence</div></div>'
        f'<div class="stat"><div class="v">{percent(result.get("prob_cn", 0.0))}</div>'
        '<div class="l">Probability of Cognitively Normal profile</div></div>'
        f'<div class="stat"><div class="v">{percent(result.get("prob_ad", 0.0))}</div>'
        '<div class="l">Probability of Alzheimer\'s disease-compatible profile</div></div>'
        "</div></div>"
    )

    uploaded_scan = (
        '<div class="card">'
        "<h2>Uploaded Brain Scan File</h2>"
        f"<p>{html.escape(uploaded_scan_name)}</p>"
        "</div>"
    )

    preprocessing = (
        '<div class="card">'
        "<h2>Scan Preparation</h2>"
        f"<p>{html.escape(preprocessing_note)}</p>"
        "</div>"
    )

    simple = (
        '<div class="card">'
        "<h2>Simple Explanation</h2>"
        f"<p>{html.escape(brain_scan_simple_explanation(prediction))}</p>"
        "</div>"
    )

    visual = ""
    cells = []
    for index, item in enumerate(valid_gradcam_items(gradcam_paths or []), start=1):
        image_path = gradcam_item_path(item)
        data_uri = _image_data_uri(image_path) if image_path else None
        if not data_uri:
            continue
        caption = gradcam_item_caption(item, index, gradcam_orientation)
        cells.append(
            '<div class="img-cell">'
            f'<img src="{data_uri}" alt="Visual explanation slice {index}">'
            f'<div class="img-cap">{html.escape(caption)}</div>'
            "</div>"
        )
    if cells:
        visual = (
            '<div class="card">'
            "<h2>Visual Explanation</h2>"
            "<p>The highlighted colors show the areas of the image that influenced "
            "the result. They do not confirm disease.</p>"
            '<div class="img-grid">' + "".join(cells) + "</div></div>"
        )

    color_guide = (
        '<div class="card">'
        "<h2>Color Guide</h2>"
        '<div class="legend">'
        '<div class="legend-row"><span class="dot dot-low"></span>'
        "<span><strong>Blue / dark areas:</strong> lower influence on the result</span></div>"
        '<div class="legend-row"><span class="dot dot-mid"></span>'
        "<span><strong>Green / yellow areas:</strong> moderate influence on the result</span></div>"
        '<div class="legend-row"><span class="dot dot-high"></span>'
        "<span><strong>Yellow / red areas:</strong> stronger influence, not confirmed disease</span></div>"
        "</div></div>"
    )

    note_card = (
        '<div class="card note">'
        f"<strong>Medical Note.</strong> {html.escape(MEDICAL_NOTE)}"
        "</div>"
    )

    body = header + uploaded_scan + preprocessing + result_summary + simple + visual + color_guide + note_card
    return _report_html_shell("Brain Scan Report", body)


def speech_result_label(prediction: str) -> str:
    if prediction == "ProbableAD":
        return "ProbableAD"
    if prediction == "Control":
        return "Control"
    return str(prediction)


def generate_speech_language_report_html(
    result: dict,
    patient_case_id: str,
    analysis_id: str,
    transcript: str,
    analysis_features: dict,
    simple_explanation: str,
    patient_info: dict | None = None,
) -> str:
    prediction = str(result.get("prediction", "Unknown"))
    confidence = float(result.get("confidence", 0.0))
    badge_class = "badge-alert" if prediction == "ProbableAD" else "badge-good"
    uploaded_file = Path(str((patient_info or {}).get("Uploaded File", "Not provided"))).name

    header = (
        '<div class="card header">'
        '<div class="eyebrow">Speech and Language Analysis</div>'
        "<h1>Speech and Language Report</h1>"
        '<div class="meta-grid">'
        + _meta_item("Patient Case ID", patient_case_id)
        + _meta_item("Analysis ID", analysis_id)
        + _meta_item("Date", datetime.now().strftime("%Y-%m-%d %H:%M"))
        + _meta_item("Analysis Type", "Speech and Language Analysis")
        + "</div></div>"
    )

    result_summary = (
        '<div class="card">'
        "<h2>Speech Analysis Result</h2>"
        f'<span class="badge {badge_class}">{html.escape(speech_result_label(prediction))}</span>'
        '<div class="stat-grid">'
        f'<div class="stat"><div class="v">{percent(confidence)}</div>'
        '<div class="l">Confidence score</div></div>'
        "</div></div>"
    )

    patient_rows = []
    for key in (
        "Patient Case ID",
        "Patient Name",
        "Study Date",
        "Responsible Clinician",
        "Clinical Notes",
    ):
        value = (patient_info or {}).get(key, "Not provided")
        if key == "Clinical Notes":
            value = sanitize_clinical_notes(value)
        patient_rows.append(
            f"<tr><td>{html.escape(str(key))}</td>"
            f"<td>{html.escape(str(value))}</td></tr>"
        )
    patient_section = (
        '<div class="card">'
        "<h2>Patient Information</h2>"
        '<table class="feature-table">' + "".join(patient_rows) + "</table>"
        "</div>"
        if patient_rows else ""
    )
    uploaded_section = (
        '<div class="card">'
        "<h2>Uploaded Speech File</h2>"
        f"<p>{html.escape(uploaded_file)}</p>"
        "</div>"
    )

    feature_rows = []
    for key, value in (analysis_features or {}).items():
        friendly = FRIENDLY_SPEECH_FEATURES.get(key, key)
        feature_rows.append(
            f"<tr><td>{html.escape(friendly)}</td>"
            f"<td>{html.escape(str(value))}</td></tr>"
        )
    transcript_text = str(transcript or "").strip() or "No transcript was provided."
    input_summary = (
        '<div class="card">'
        "<h2>Extracted Speech Text</h2>"
        f'<div class="transcript">{html.escape(transcript_text)}</div>'
        "</div>"
        '<div class="card">'
        "<h2>Automatically Extracted Speech Information</h2>"
        '<table class="feature-table">' + "".join(feature_rows) + "</table>"
        "</div>"
    )

    simple = (
        '<div class="card">'
        "<h2>Simple Explanation</h2>"
        f"<p>{html.escape(str(simple_explanation).strip())}</p>"
        "<p>This result is a decision-support output and does not mean that the "
        "patient has Alzheimer's disease.</p>"
        "</div>"
    )

    note_card = (
        '<div class="card note">'
        "<strong>Medical Note.</strong> "
        "The transcript was automatically extracted and cleaned from the uploaded "
        "speech transcript file. This result is not a medical diagnosis and must "
        "be reviewed by a healthcare professional."
        "</div>"
    )

    body = (
        header + patient_section + uploaded_section + input_summary
        + result_summary + simple + note_card
    )
    return _report_html_shell("Speech and Language Report", body)


def save_report_markdown(report_text: str, report_type: str, patient_id: str) -> Path:
    ensure_output_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_patient_id = safe_filename_part(patient_id, fallback="patient")
    report_dir = CNN_REPORTS_DIR if report_type == "cnn" else REPORTS_DIR
    report_path = report_dir / f"{report_type}_report_{safe_patient_id}_{timestamp}.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def save_patient_report(report_text: str, patient_id: str = "patient") -> Path:
    return save_report_markdown(report_text, "nlp", patient_id)


def save_cnn_report(
    result: dict,
    patient_id: str,
    gradcam_orientation: str | None = None,
    gradcam_slice_count: int = 0,
    gradcam_paths: list[str | Path] | None = None,
    mri_preview_path: str | Path | None = None,
    mri_rag_explanation: str | None = None,
    mri_rag_sources: list | None = None,
    preprocessing_note: str | None = None,
) -> Path:
    report_text = generate_cnn_report(
        result=result,
        patient_id=patient_id,
        gradcam_orientation=gradcam_orientation,
        gradcam_slice_count=gradcam_slice_count,
        gradcam_paths=gradcam_paths,
        mri_preview_path=mri_preview_path,
        mri_source_name=st.session_state.get("mri_source_name"),
        mri_rag_explanation=mri_rag_explanation,
        mri_rag_sources=mri_rag_sources,
        preprocessing_note=preprocessing_note or st.session_state.get("mri_preprocessing_note"),
    )
    return save_report_markdown(report_text, "cnn", patient_id)


def save_brain_scan_reports(
    result: dict,
    patient_case_id: str,
    analysis_id: str,
    gradcam_paths: list | None = None,
    gradcam_orientation: str | None = None,
    mri_preview_path: str | Path | None = None,
    mri_source_name: str | None = None,
    mri_rag_explanation: str | None = None,
    mri_rag_sources: list | None = None,
    preprocessing_note: str | None = None,
) -> dict:
    """
    Write the Brain Scan report as .md (+ PDF) and a self-contained patient HTML
    file, using the global Patient Case ID and the analysis-specific Analysis ID.
    """
    ensure_output_dirs()
    base = report_basename("brain", patient_case_id, analysis_id)
    md_path = CNN_REPORTS_DIR / f"{base}.md"
    html_path = CNN_REPORTS_DIR / f"{base}.html"

    valid_paths = valid_gradcam_items(gradcam_paths or [])
    md_text = generate_cnn_report(
        result=result,
        patient_id=patient_case_id,
        analysis_id=analysis_id,
        gradcam_orientation=gradcam_orientation,
        gradcam_slice_count=visual_explanation_slice_count(valid_paths),
        gradcam_paths=gradcam_paths,
        mri_preview_path=mri_preview_path,
        mri_source_name=mri_source_name,
        mri_rag_explanation=mri_rag_explanation,
        mri_rag_sources=mri_rag_sources,
        preprocessing_note=preprocessing_note,
    )
    md_path.write_text(md_text, encoding="utf-8")

    html_text = generate_brain_scan_report_html(
        result=result,
        patient_case_id=patient_case_id,
        analysis_id=analysis_id,
        gradcam_paths=gradcam_paths,
        gradcam_orientation=gradcam_orientation,
        mri_source_name=mri_source_name,
        preprocessing_note=preprocessing_note,
    )
    html_path.write_text(html_text, encoding="utf-8")

    try:
        stale_pdf = md_path.with_suffix(".pdf")
        if stale_pdf.exists():
            stale_pdf.unlink()
        pdf_path = save_patient_report_pdf(md_text, md_path)
    except Exception:
        pdf_path = None

    return {"md": md_path, "html": html_path, "pdf": pdf_path}


def save_speech_language_reports(
    result: dict,
    patient_case_id: str,
    analysis_id: str,
    transcript: str,
    analysis_features: dict,
    patient_info: dict,
    llm_explanation: str,
) -> dict:
    """
    Write the Speech and Language report as .md (+ PDF) and a self-contained
    patient HTML file, using the Patient Case ID and the Analysis ID.
    """
    ensure_output_dirs()
    base = report_basename("speech", patient_case_id, analysis_id)
    md_path = REPORTS_DIR / f"{base}.md"
    html_path = REPORTS_DIR / f"{base}.html"

    md_text = generate_patient_report(
        result=result,
        patient_info=patient_info,
        analysis_features=analysis_features,
        llm_explanation=llm_explanation,
        transcript=transcript,
    )
    md_path.write_text(md_text, encoding="utf-8")

    simple_explanation = sanitize_llm_explanation(
        llm_explanation,
        required_nlp_explanation(result["prediction"], result["confidence"]),
    )
    html_text = generate_speech_language_report_html(
        result=result,
        patient_case_id=patient_case_id,
        analysis_id=analysis_id,
        transcript=transcript,
        analysis_features=analysis_features,
        simple_explanation=simple_explanation,
        patient_info=patient_info,
    )
    html_path.write_text(html_text, encoding="utf-8")

    try:
        stale_pdf = md_path.with_suffix(".pdf")
        if stale_pdf.exists():
            stale_pdf.unlink()
        pdf_path = save_patient_report_pdf(md_text, md_path)
    except Exception:
        pdf_path = None

    return {"md": md_path, "html": html_path, "pdf": pdf_path}


def markdown_to_pdf_blocks(report_text: str):
    blocks = []

    for raw_line in report_text.splitlines():
        line = raw_line.strip()

        if not line:
            blocks.append(("space", ""))
            continue

        if line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
        elif line.startswith("## "):
            blocks.append(("heading", line[3:].strip()))
        elif line.startswith("- "):
            blocks.append(("bullet", line[2:].strip()))
        else:
            blocks.append(("text", line))

    return blocks


def markdown_inline_to_reportlab(text: str) -> str:
    link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    parts = []
    cursor = 0

    for match in link_pattern.finditer(text):
        parts.append(html.escape(text[cursor:match.start()]))
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        parts.append(f'<link href="{url}"><font color="#0a795a">{label}</font></link>')
        cursor = match.end()

    parts.append(html.escape(text[cursor:]))
    escaped = "".join(parts)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def parse_report_sections(report_text: str) -> tuple[str, dict[str, list[str]]]:
    title = "Patient Report"
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("# "):
            title = line[2:].strip()
            continue

        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
            continue

        if current_section:
            sections[current_section].append(line)

    return title, sections


def report_bullet_rows(lines: list[str]) -> list[tuple[str, str]]:
    rows = []
    for line in lines:
        if not line.startswith("- "):
            continue

        text = line[2:].strip()
        match = re.match(r"\*\*(.+?):\*\*\s*(.*)", text)
        if match:
            rows.append((match.group(1).strip(), match.group(2).strip()))
        else:
            rows.append(("", text))
    return rows


def section_plain_text(lines: list[str]) -> str:
    cleaned = []
    for line in lines:
        if line.startswith("- "):
            cleaned.append(line[2:].strip())
        elif line.startswith("!["):
            continue
        elif line.startswith("### "):
            continue
        else:
            cleaned.append(line)
    return " ".join(cleaned).strip()


def markdown_image(line: str) -> tuple[str, Path] | None:
    match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
    if not match:
        return None

    label = match.group(1).strip() or "Image"
    raw_path = match.group(2).strip()
    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path
    return label, image_path


def report_images_from_lines(lines: list[str]) -> list[tuple[str, Path]]:
    images = []
    pending_label: str | None = None
    for line in lines:
        if line.startswith("### "):
            pending_label = line[4:].strip()
            continue

        image = markdown_image(line)
        if image:
            label, image_path = image
            images.append((pending_label or label, image_path))
            pending_label = None
    return images


def add_section_title(story, title: str, Paragraph, Spacer, styles) -> None:
    story.append(Spacer(1, 10))
    story.append(Paragraph(markdown_inline_to_reportlab(title), styles["ReportSectionTitle"]))
    story.append(Spacer(1, 6))


def build_pdf_with_reportlab(report_text: str, pdf_path: Path) -> None:
    colors = importlib.import_module("reportlab.lib.colors")
    pagesizes = importlib.import_module("reportlab.lib.pagesizes")
    styles_module = importlib.import_module("reportlab.lib.styles")
    units = importlib.import_module("reportlab.lib.units")
    platypus = importlib.import_module("reportlab.platypus")

    A4 = pagesizes.A4
    getSampleStyleSheet = styles_module.getSampleStyleSheet
    cm = units.cm
    Paragraph = platypus.Paragraph
    ReportImage = platypus.Image
    Table = platypus.Table
    TableStyle = platypus.TableStyle
    SimpleDocTemplate = platypus.SimpleDocTemplate
    Spacer = platypus.Spacer
    KeepTogether = platypus.KeepTogether
    PageBreak = platypus.PageBreak

    PRIMARY = colors.HexColor("#0F9D7A")
    DEEP = colors.HexColor("#096B5A")
    VIOLET = colors.HexColor("#8B5CF6")
    CORAL = colors.HexColor("#F97362")
    GOLD = colors.HexColor("#C99A2E")
    BG_LIGHT = colors.HexColor("#F8F4EC")
    CARD_BG = colors.HexColor("#FFFDF8")
    LINE = colors.HexColor("#ECE7DB")
    TEXT = colors.HexColor("#202124")
    MUTED = colors.HexColor("#6B7280")
    WARNING_BG = colors.HexColor("#FFF7E2")

    styles = getSampleStyleSheet()
    title_style = styles["Title"].clone("ReportTitle")
    title_style.fontName = "Helvetica-Bold"
    title_style.fontSize = 20
    title_style.leading = 24
    title_style.textColor = colors.HexColor("#16342d")
    title_style.spaceAfter = 4

    eyebrow_style = styles["BodyText"].clone("ReportEyebrow")
    eyebrow_style.fontName = "Helvetica-Bold"
    eyebrow_style.fontSize = 8
    eyebrow_style.leading = 10
    eyebrow_style.textColor = DEEP

    subtitle_style = styles["BodyText"].clone("ReportSubtitle")
    subtitle_style.fontName = "Helvetica"
    subtitle_style.fontSize = 9
    subtitle_style.leading = 12
    subtitle_style.textColor = MUTED

    section_style = styles["Heading2"].clone("ReportSectionTitle")
    section_style.fontName = "Helvetica-Bold"
    section_style.fontSize = 12
    section_style.leading = 15
    section_style.textColor = DEEP
    styles.add(section_style)

    body_style = styles["BodyText"].clone("ReportBody")
    body_style.fontName = "Helvetica"
    body_style.fontSize = 9.6
    body_style.leading = 14
    body_style.textColor = colors.HexColor("#1f2937")

    small_style = styles["BodyText"].clone("ReportSmall")
    small_style.fontName = "Helvetica"
    small_style.fontSize = 8.5
    small_style.leading = 11
    small_style.textColor = MUTED

    label_style = styles["BodyText"].clone("ReportLabel")
    label_style.fontName = "Helvetica-Bold"
    label_style.fontSize = 8.3
    label_style.leading = 10
    label_style.textColor = MUTED

    value_style = styles["BodyText"].clone("ReportValue")
    value_style.fontName = "Helvetica-Bold"
    value_style.fontSize = 12
    value_style.leading = 15
    value_style.textColor = colors.HexColor("#16342d")

    badge_style = styles["BodyText"].clone("ReportBadge")
    badge_style.fontName = "Helvetica-Bold"
    badge_style.fontSize = 11
    badge_style.leading = 13
    badge_style.textColor = colors.white

    stat_value_style = styles["BodyText"].clone("ReportStatValue")
    stat_value_style.fontName = "Helvetica-Bold"
    stat_value_style.fontSize = 16
    stat_value_style.leading = 19
    stat_value_style.textColor = DEEP

    bullet_style = body_style.clone("ReportBullet")
    bullet_style.leftIndent = 14
    bullet_style.firstLineIndent = -8

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.7 * cm,
        title="Patient Report",
    )
    page_width = A4[0] - doc.leftMargin - doc.rightMargin
    title, sections = parse_report_sections(report_text)
    is_speech_report = (
        "Uploaded Speech File" in sections
        or "Extracted Speech Text" in sections
        or "Automatically Extracted Speech Information" in sections
    )

    def make_table(data, style_commands, col_widths=None):
        table = Table(data, colWidths=col_widths, hAlign="LEFT")
        table.setStyle(TableStyle(style_commands))
        return table

    def paragraph(text: str, style=body_style):
        return Paragraph(markdown_inline_to_reportlab(text), style)

    def text_card(text: str, background=CARD_BG, border=LINE):
        return make_table(
            [[paragraph(text or "Not provided", body_style)]],
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ],
            [page_width],
        )

    def info_table(rows: list[tuple[str, str]], label_width: float = 0.36):
        return make_table(
            [
                [paragraph(key, label_style), paragraph(value or "Not provided", body_style)]
                for key, value in rows
            ],
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F2EA")),
                ("BACKGROUND", (1, 0), (1, -1), CARD_BG),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ],
            [page_width * label_width, page_width * (1 - label_width)],
        )

    def warning_box(text: str):
        return make_table(
            [[paragraph(text, body_style)]],
            [
                ("BACKGROUND", (0, 0), (-1, -1), WARNING_BG),
                ("BOX", (0, 0), (-1, -1), 0.8, GOLD),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, GOLD),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ],
            [page_width],
        )

    def result_is_alert(text: str) -> bool:
        lowered = str(text or "").lower()
        return "probable" in lowered or "alzheimer" in lowered or "compatible" in lowered or "ad" == lowered.strip()

    def result_summary_card(result_text: str, confidence_text: str):
        alert = result_is_alert(result_text)
        badge_color = CORAL if alert else PRIMARY
        card_bg = colors.HexColor("#FFF4EF") if alert else colors.HexColor("#EFFAF5")
        badge = make_table(
            [[paragraph(result_text or "Not provided", badge_style)]],
            [
                ("BACKGROUND", (0, 0), (-1, -1), badge_color),
                ("BOX", (0, 0), (-1, -1), 0.0, badge_color),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ],
        )
        confidence = make_table(
            [
                [paragraph("Confidence score", label_style)],
                [paragraph(confidence_text or "Not provided", stat_value_style)],
            ],
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D9EEE5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ],
            [page_width * 0.35],
        )
        return make_table(
            [[badge, confidence]],
            [
                ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D4EDE2")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ],
            [page_width * 0.58, page_width * 0.42],
        )

    def make_report_image(image_path: Path, max_width: float, max_height: float):
        try:
            pil_module = importlib.import_module("PIL.Image")
            with pil_module.open(image_path) as pil_img:
                width_px, height_px = pil_img.size
            if width_px <= 0 or height_px <= 0:
                return None

            ratio = height_px / width_px
            image_width = max_width
            image_height = image_width * ratio
            if image_height > max_height:
                image_height = max_height
                image_width = image_height / ratio

            return ReportImage(str(image_path), width=image_width, height=image_height)
        except Exception:
            return None

    def append_report_image(story, image_path: Path, caption: str, max_width: float, max_height: float) -> bool:
        image = make_report_image(image_path, max_width, max_height)
        if image is None:
            return False
        story.append(image)
        story.append(paragraph(caption, small_style))
        story.append(Spacer(1, 6))
        return True

    def footer(canvas, document):
        canvas.saveState()
        width, _height = A4
        canvas.setFillColor(BG_LIGHT)
        canvas.rect(0, 0, width, _height, fill=1, stroke=0)
        y = 0.9 * cm
        canvas.setStrokeColor(LINE)
        canvas.line(document.leftMargin, y + 12, width - document.rightMargin, y + 12)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(document.leftMargin, y, "Patient report - decision-support output")
        canvas.drawRightString(width - document.rightMargin, y, f"Page {document.page}")
        canvas.restoreState()

    header_patient_rows = report_bullet_rows(sections.get("Patient Information", []))
    header_patient_map = {key.lower(): value for key, value in header_patient_rows}
    header_detail_rows = report_bullet_rows(sections.get("Analysis Details", []))
    header_detail_map = {key.lower(): value for key, value in header_detail_rows}
    header_meta = [
        ("Patient Case ID", header_patient_map.get("patient case id", "Not provided")),
        ("Analysis ID", header_patient_map.get("analysis id", header_detail_map.get("analysis id", "Not provided"))),
        ("Date", datetime.now().strftime("%d/%m/%Y %H:%M")),
        ("Analysis Type", header_detail_map.get("analysis type", "Speech and Language Analysis" if is_speech_report else "Brain Scan Analysis")),
    ]
    display_title = "Speech and Language Report" if is_speech_report else "Brain Scan Report"
    eyebrow = "Speech and Language Analysis" if is_speech_report else "Brain Scan Analysis"
    header_inner_width = page_width - 28
    meta_cells = [
        make_table(
            [[paragraph(label, label_style)], [paragraph(value or "Not provided", body_style)]],
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9EEE5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ],
            [header_inner_width / 4.0 - 8],
        )
        for label, value in header_meta
    ]
    header_grid = make_table(
        [meta_cells],
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ],
        [
            header_inner_width / 4.0,
            header_inner_width / 4.0,
            header_inner_width / 4.0,
            header_inner_width / 4.0,
        ],
    )
    story = [
        make_table(
            [
                [paragraph(eyebrow, eyebrow_style)],
                [paragraph(display_title, title_style)],
                [header_grid],
            ],
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F6EF")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B9DFCF")),
                ("LINEBEFORE", (0, 0), (0, -1), 4.0, PRIMARY),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ],
            [page_width],
        ),
        Spacer(1, 12),
    ]

    patient_rows = report_bullet_rows(sections.get("Patient Information", []))
    if is_speech_report:
        patient_rows = [
            (key, sanitize_clinical_notes(value) if key == "Clinical Notes" else value)
            for key, value in patient_rows
            if key not in {"Patient Case ID", "Analysis ID"}
        ]
    if patient_rows:
        add_section_title(story, "Patient Information", Paragraph, Spacer, styles)
        story.append(info_table(patient_rows))

    detail_rows = report_bullet_rows(sections.get("Analysis Details", []))
    if detail_rows:
        add_section_title(story, "Analysis Details", Paragraph, Spacer, styles)
        story.append(info_table(detail_rows))

    uploaded_file_title = (
        "Uploaded Speech File"
        if sections.get("Uploaded Speech File", [])
        else "Uploaded Brain Scan File"
    )
    uploaded_file_lines = (
        sections.get("Uploaded Speech File", [])
        or sections.get("Uploaded Brain Scan File", [])
    )
    if uploaded_file_lines:
        add_section_title(story, uploaded_file_title, Paragraph, Spacer, styles)
        story.append(
            make_table(
                [[paragraph(section_plain_text(uploaded_file_lines), value_style)]],
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfaf7")),
                    ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#e7e3d8")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ],
                [page_width],
            )
        )

    if is_speech_report:
        extracted_speech_lines = sections.get("Extracted Speech Text", [])
        if extracted_speech_lines:
            add_section_title(story, "Extracted Speech Text", Paragraph, Spacer, styles)
            story.append(text_card(section_plain_text(extracted_speech_lines)))

        extracted_info_lines = sections.get("Automatically Extracted Speech Information", [])
        if extracted_info_lines:
            add_section_title(
                story,
                "Automatically Extracted Speech Information",
                Paragraph,
                Spacer,
                styles,
            )
            feature_rows = report_bullet_rows(extracted_info_lines)
            if feature_rows:
                story.append(info_table(feature_rows, label_width=0.42))

    speech_result_rows = report_bullet_rows(sections.get("Speech Analysis Result", []))
    if speech_result_rows:
        speech_result_map = {key.lower(): value for key, value in speech_result_rows}
        result_text = speech_result_map.get("result", "")
        confidence_text = speech_result_map.get("confidence", "")
        result_section_title = "Speech Analysis Result"
    else:
        result_text = section_plain_text(sections.get("Analysis Result", []))
        confidence_text = section_plain_text(sections.get("Confidence Score", []))
        result_section_title = "Brain Scan Result"
    if result_text or confidence_text:
        add_section_title(story, result_section_title, Paragraph, Spacer, styles)
        story.append(result_summary_card(result_text, confidence_text))

    for extra_title in ("Probability Details", "Visual Explanation Details"):
        extra_lines = sections.get(extra_title, [])
        if extra_lines:
            add_section_title(story, extra_title, Paragraph, Spacer, styles)
            extra_rows = report_bullet_rows(extra_lines)
            if extra_rows:
                story.append(info_table(extra_rows, label_width=0.48))
            else:
                story.append(text_card(section_plain_text(extra_lines)))

    simple_lines = sections.get("Simple Explanation", [])
    if simple_lines:
        add_section_title(story, "Simple Explanation", Paragraph, Spacer, styles)
        story.append(text_card(section_plain_text(simple_lines)))

    gradcam_lines = sections.get("Visual Explanation", [])
    if gradcam_lines:
        add_section_title(story, "Visual Explanation", Paragraph, Spacer, styles)
        legend_rows = report_bullet_rows(gradcam_lines)
        visual_text = section_plain_text([line for line in gradcam_lines if not line.startswith("- ")])
        if visual_text:
            story.append(text_card(visual_text, background=colors.HexColor("#F9F7F0")))
        if legend_rows:
            legend_data = []
            legend_colors = [colors.HexColor("#1E2A36"), colors.HexColor("#F3B33D"), colors.HexColor("#E66B3D")]
            for index, (label, value) in enumerate(legend_rows[:3]):
                chip = make_table(
                    [[""]],
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), legend_colors[index]),
                        ("BOX", (0, 0), (-1, -1), 0.3, colors.HexColor("#D9D1C3")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ],
                    [0.35 * cm],
                )
                legend_data.append(
                    [
                        chip,
                        paragraph(f"**{label}:** {value}", body_style),
                    ]
                )
            story.append(
                make_table(
                    legend_data,
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
                        ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("LEFTPADDING", (0, 0), (0, -1), 4),
                        ("RIGHTPADDING", (0, 0), (0, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ],
                    [0.8 * cm, page_width - 0.8 * cm],
                )
            )

    preview_images = report_images_from_lines(sections.get("MRI Preview Image", []))
    if preview_images:
        add_section_title(story, "MRI Preview Image", Paragraph, Spacer, styles)
        for label, image_path in preview_images:
            if image_path.exists():
                append_report_image(story, image_path, label, page_width * 0.55, 7.5 * cm)

    gradcam_images = report_images_from_lines(sections.get("Visual Explanation Images", []))
    if gradcam_images:
        add_section_title(story, "Visual Explanation Images", Paragraph, Spacer, styles)
        available_images = [(label, image_path) for label, image_path in gradcam_images if image_path.exists()]
        if not available_images:
            story.append(paragraph("Visual explanation image was not available for this report.", body_style))
        elif len(available_images) == 1:
            label, image_path = available_images[0]
            if not append_report_image(story, image_path, label, page_width, 12.5 * cm):
                story.append(paragraph("Visual explanation image was not available for this report.", body_style))
        else:
            image_cells = []
            cell_width = page_width / 4.0
            for label, image_path in available_images:
                image = make_report_image(image_path, cell_width - 8, 3.6 * cm)
                if image is not None:
                    image_cells.append([image, paragraph(label, small_style)])

            if not image_cells:
                story.append(paragraph("Visual explanation image was not available for this report.", body_style))
                image_cells = []

            for row_start in range(0, len(image_cells), 4):
                row = image_cells[row_start:row_start + 4]
                while len(row) < 4:
                    row.append("")
                grid = make_table(
                    [row],
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ],
                    [cell_width, cell_width, cell_width, cell_width],
                )
                story.append(grid)
    elif "Visual Explanation Images" in sections:
        add_section_title(story, "Visual Explanation Images", Paragraph, Spacer, styles)
        story.append(
            paragraph(
                section_plain_text(sections.get("Visual Explanation Images", []))
                or "Visual explanation image was not available for this report.",
                body_style,
            )
        )

    mri_explanation_lines = sections.get("Simple Brain Scan Explanation", [])
    if mri_explanation_lines:
        add_section_title(story, "Simple Brain Scan Explanation", Paragraph, Spacer, styles)
        story.append(text_card(section_plain_text(mri_explanation_lines)))

    extracted_speech_lines = sections.get("Extracted Speech Text", [])
    if extracted_speech_lines and not is_speech_report:
        add_section_title(story, "Extracted Speech Text", Paragraph, Spacer, styles)
        story.append(
            make_table(
                [[paragraph(section_plain_text(extracted_speech_lines), body_style)]],
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfaf7")),
                    ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#e7e3d8")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ],
                [page_width],
            )
        )

    extracted_info_lines = sections.get("Automatically Extracted Speech Information", [])
    if extracted_info_lines and not is_speech_report:
        add_section_title(
            story,
            "Automatically Extracted Speech Information",
            Paragraph,
            Spacer,
            styles,
        )
        feature_rows = report_bullet_rows(extracted_info_lines)
        if feature_rows:
            feature_data = [
                [paragraph(key, label_style), paragraph(value or "Not provided", body_style)]
                for key, value in feature_rows
            ]
            story.append(
                make_table(
                    feature_data,
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f5f4ef")),
                        ("BACKGROUND", (1, 0), (1, -1), colors.white),
                        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e7e3d8")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e7e3d8")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ],
                    [page_width * 0.42, page_width * 0.58],
                )
            )
        else:
            for line in extracted_info_lines:
                story.append(paragraph(line, body_style))

    sign_lines = (
        sections.get("Speech Patterns Observed", [])
        or sections.get("Observed Linguistic Signs", [])
    )
    if sign_lines:
        add_section_title(story, "Speech Patterns Observed", Paragraph, Spacer, styles)
        sign_flow = []
        for line in sign_lines:
            if line.startswith("- "):
                sign_flow.append(paragraph("- " + line[2:].strip(), bullet_style))
            else:
                sign_flow.append(paragraph(line, body_style))
        story.append(KeepTogether(sign_flow[:3]))
        for item in sign_flow[3:]:
            story.append(item)

    info_source_lines = (
        sections.get("Information Sources", [])
        or sections.get("MRI Medical Sources Used", [])
        or sections.get("Medical Sources Used", [])
    )
    if info_source_lines:
        add_section_title(story, "Information Sources", Paragraph, Spacer, styles)
        for line in info_source_lines:
            if line.startswith("- "):
                story.append(paragraph("- " + line[2:].strip(), bullet_style))
            else:
                story.append(paragraph(line, body_style))

    note_lines = sections.get("Medical Note", []) or sections.get("Important Medical Note", [])
    if note_lines:
        note_title = "Medical Note" if sections.get("Medical Note", []) else "Important Medical Note"
        add_section_title(story, note_title, Paragraph, Spacer, styles)
        story.append(warning_box(section_plain_text(note_lines)))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def pdf_escape(text: str) -> str:
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_pdf_text(text: str, max_chars: int = 92) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def build_minimal_pdf(report_text: str, pdf_path: Path) -> None:
    lines = []
    for block_type, text in markdown_to_pdf_blocks(report_text):
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        if block_type == "space":
            lines.append("")
        elif block_type == "title":
            lines.extend(wrap_pdf_text(clean.upper(), 70))
            lines.append("")
        elif block_type == "heading":
            lines.append(clean)
            lines.append("-" * min(len(clean), 70))
        elif block_type == "bullet":
            lines.extend(wrap_pdf_text("- " + clean, 88))
        else:
            lines.extend(wrap_pdf_text(clean))

    pages = [lines[index:index + 46] for index in range(0, len(lines), 46)] or [[]]
    objects = []

    def add_object(content: str) -> int:
        objects.append(content)
        return len(objects)

    font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    content_ids = []
    for page_lines in pages:
        stream_lines = ["BT", "/F1 10 Tf", "50 800 Td", "14 TL"]
        for index, line in enumerate(page_lines):
            if index:
                stream_lines.append("T*")
            stream_lines.append(f"({pdf_escape(line)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)
        content_id = add_object(
            f"<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\n"
            f"stream\n{stream}\nendstream"
        )
        content_ids.append(content_id)

    first_page_id = len(objects) + 1
    page_ids = list(range(first_page_id, first_page_id + len(content_ids)))
    pages_id = first_page_id + len(content_ids)
    pages_kids = [f"{page_id} 0 R" for page_id in page_ids]

    for content_id in content_ids:
        add_object(
            "<< /Type /Page "
            f"/Parent {pages_id} 0 R "
            "/MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )

    add_object(
        f"<< /Type /Pages /Kids [{' '.join(pages_kids)}] /Count {len(page_ids)} >>"
    )
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, content in enumerate(objects, start=1):
        offsets.append(pdf.tell())
        pdf.write(f"{object_id} 0 obj\n{content}\nendobj\n".encode("latin-1", errors="replace"))

    xref_offset = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.write(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )

    pdf_path.write_bytes(pdf.getvalue())


def save_patient_report_pdf(report_text: str, markdown_path: Path | None = None) -> Path:
    ensure_output_dirs()
    if markdown_path is not None:
        pdf_path = markdown_path.with_suffix(".pdf")
    else:
        pdf_path = REPORTS_DIR / f"patient_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    build_pdf_with_reportlab(report_text, pdf_path)

    return pdf_path


def ensure_pdf_for_report(report_path: Path) -> Path | None:
    """Create the styled PDF next to a report if it is missing."""
    pdf_path = report_path.with_suffix(".pdf")
    if pdf_path.exists():
        return pdf_path

    md_path = report_path.with_suffix(".md")
    if not md_path.exists():
        return None

    try:
        md_text = md_path.read_text(encoding="utf-8")
        return save_patient_report_pdf(md_text, md_path)
    except Exception:
        return None


def report_sibling_files(report_path: Path) -> dict:
    """Return the available download formats (html / md / pdf) for one report."""
    formats = {}
    for ext in ("html", "md", "pdf"):
        candidate = report_path.with_suffix(f".{ext}")
        if ext == "pdf" and not candidate.exists():
            candidate = ensure_pdf_for_report(report_path) or candidate
        if candidate.exists():
            formats[ext] = candidate
    return formats


def report_has_valid_visual_images(report_path: Path) -> bool:
    md_path = report_path.with_suffix(".md")
    if not md_path.exists():
        return False

    try:
        _title, sections = parse_report_sections(md_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    images = report_images_from_lines(sections.get("Visual Explanation Images", []))
    return any(image_path.exists() for _label, image_path in images)


def can_generate_current_brain_visual_pdf(report_path: Path) -> bool:
    if report_kind(report_path) != "Brain Scan":
        return False
    current_report = st.session_state.get("cnn_report_path")
    if not current_report or Path(current_report).with_suffix(".md") != report_path.with_suffix(".md"):
        return False
    return bool(st.session_state.get("cnn_result") and st.session_state.get("mri_path"))


def generate_current_brain_report_with_visual_images() -> Path | None:
    result = st.session_state.get("cnn_result")
    mri_path = st.session_state.get("mri_path")
    if not result or not mri_path:
        return None

    patient_id = ensure_patient_id()
    analysis_id = st.session_state.get("cnn_analysis_id") or generate_analysis_id()
    st.session_state.cnn_analysis_id = analysis_id

    gradcam_paths = valid_gradcam_items(st.session_state.get("gradcam_paths") or [])
    if not gradcam_paths:
        gradcam_paths = run_gradcam_generation(
            mri_path,
            orientation="multi",
            num_slices=5,
            display_mode="overlay",
        )
        gradcam_paths = valid_gradcam_items(gradcam_paths)
        st.session_state.gradcam_paths = gradcam_paths
        st.session_state.gradcam_generated_orientation = "Axial, sagittal and coronal views"
        st.session_state.gradcam_generated_display_mode = "Overlay view"

    paths = save_brain_scan_reports(
        result=result,
        patient_case_id=patient_id,
        analysis_id=analysis_id,
        gradcam_paths=gradcam_paths,
        gradcam_orientation=visual_explanation_orientation_label(
            gradcam_paths,
            st.session_state.get("gradcam_generated_orientation"),
        ),
        mri_preview_path=st.session_state.get("cnn_preview_path"),
        mri_source_name=st.session_state.get("mri_source_name"),
        mri_rag_explanation=st.session_state.get("mri_rag_explanation"),
        mri_rag_sources=st.session_state.get("mri_rag_sources"),
        preprocessing_note=st.session_state.get("mri_preprocessing_note"),
    )
    st.session_state.cnn_report_path = paths["md"]
    st.session_state.cnn_report_html_path = paths["html"]
    st.session_state.cnn_report_pdf_path = paths["pdf"]
    return paths["pdf"]


def generated_reports() -> list[Path]:
    by_stem: dict[tuple, Path] = {}
    for report_dir in (REPORTS_DIR, CNN_REPORTS_DIR):
        if not report_dir.exists():
            continue
        for pattern in ("*_report_*.html", "*_report_*.md"):
            for path in report_dir.glob(pattern):
                key = (str(report_dir), path.stem)
                # Prefer the HTML version as the representative file for a report.
                if key not in by_stem or path.suffix.lower() == ".html":
                    by_stem[key] = path
    return sorted(
        by_stem.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def save_uploaded_mri(uploaded_file) -> Path:
    ensure_output_dirs()
    original_name = Path(uploaded_file.name).name
    _stem, suffix = nifti_stem_and_suffix(original_name)
    if suffix not in {".nii", ".nii.gz"}:
        raise ValueError("Unsupported MRI file extension. Please upload a .nii or .nii.gz file.")
    output_path = MRI_UPLOAD_DIR / f"mri_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


# ---------------------------------------------------------------------------
# MRI volume preview (lightweight, never blocks the analysis)
# ---------------------------------------------------------------------------
def load_mri_volume_cached(mri_path_str: str):
    if mri_path_str in _MRI_VOLUME_CACHE:
        return _MRI_VOLUME_CACHE[mri_path_str]

    nib = importlib.import_module("nibabel")
    image = nib.load(mri_path_str)
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    _MRI_VOLUME_CACHE[mri_path_str] = data
    return data


def normalize_slice(slice_2d) -> np.ndarray:
    slice_2d = np.asarray(slice_2d, dtype=np.float32)
    low = float(np.percentile(slice_2d, 1))
    high = float(np.percentile(slice_2d, 99))
    if high <= low:
        low = float(slice_2d.min())
        high = float(slice_2d.max())
    if high <= low:
        return np.zeros(slice_2d.shape, dtype=np.uint8)
    clipped = np.clip(slice_2d, low, high)
    norm = (clipped - low) / (high - low)
    return (norm * 255.0).astype(np.uint8)


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        plane = volume[index, :, :]
    elif axis == 1:
        plane = volume[:, index, :]
    else:
        plane = volume[:, :, index]
    return np.rot90(plane)


def save_mri_preview_image(mri_path: str | Path, patient_id: str) -> Path | None:
    try:
        image_module = importlib.import_module("PIL.Image")
        volume = load_mri_volume_cached(str(mri_path))
        if volume is None or volume.ndim != 3 or int(min(volume.shape)) < 2:
            return None

        axis = 2
        index = int(volume.shape[axis]) // 2
        image_array = normalize_slice(take_slice(volume, axis, index))
        image = image_module.fromarray(image_array, mode="L")
        image = image.resize((512, 512))
        asset_dir = CNN_REPORTS_DIR / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        output_path = asset_dir / (
            f"mri_preview_{safe_filename_part(patient_id, 'patient')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        image.save(output_path)
        return output_path
    except Exception:
        return None


def build_3d_preview(volume: np.ndarray):
    go = importlib.import_module("plotly.graph_objects")

    factors = [max(1, dim // 44) for dim in volume.shape]
    small = volume[:: factors[0], :: factors[1], :: factors[2]].astype(np.float32)

    low = float(np.percentile(small, 35))
    high = float(np.percentile(small, 99))
    if high <= low:
        low, high = float(small.min()), float(small.max())
    if high <= low:
        high = low + 1.0

    xs, ys, zs = np.mgrid[
        0 : small.shape[0],
        0 : small.shape[1],
        0 : small.shape[2],
    ]

    figure = go.Figure(
        data=go.Volume(
            x=xs.flatten(),
            y=ys.flatten(),
            z=zs.flatten(),
            value=small.flatten(),
            isomin=low,
            isomax=high,
            opacity=0.08,
            surface_count=14,
            colorscale="Greys",
            showscale=False,
        )
    )
    figure.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return figure


def get_visual_explanation_caption(slice_number: int, orientation: str | None = None) -> str:
    # Fallback only for older sessions that still contain image paths without
    # heatmap metadata. New captions come from cnn_module/src/gradcam_3d.py.
    zones = [
        "near the upper left of this image",
        "near the upper center of this image",
        "near the upper right of this image",
        "on the image left side",
        "near the center of this image",
        "on the image right side",
        "near the lower left of this image",
        "near the lower center of this image",
        "near the lower right of this image",
    ]
    zone = zones[(slice_number - 1) % len(zones)]
    captions = [
        f"A small highlighted area appears {zone}.",
        f"The main colored area is visible {zone}.",
        f"A brighter area appears {zone}.",
        f"A wider colored area is visible {zone}.",
        f"A focused color area appears {zone}.",
        f"The strongest visible color is located {zone}.",
        f"A limited colored area is seen {zone}.",
        f"The color pattern is mostly visible {zone}.",
        f"A compact highlighted area appears {zone}.",
        f"The brightest part of this slice appears {zone}.",
        f"A broad colored pattern is strongest {zone}.",
        f"A small bright focus is visible {zone}.",
    ]
    text = captions[(slice_number - 1) % len(captions)]
    return f"Slice {slice_number} - {text}"


def initialize_state() -> None:
    defaults = {
        "page": "Home",
        "patient_case_id": "",
        "cnn_analysis_id": "",
        "nlp_analysis_id": "",
        "cnn_result": None,
        "gradcam_paths": [],
        "gradcam_orientation": "Axial",
        "gradcam_display_mode": "Overlay view",
        "gradcam_generated_orientation": None,
        "gradcam_generated_display_mode": None,
        "cnn_report_path": None,
        "cnn_report_html_path": None,
        "cnn_report_pdf_path": None,
        "cnn_preview_path": None,
        "mri_path": None,
        "mri_original_path": None,
        "mri_processed_path": None,
        "mri_source_name": None,
        "apply_hd_bet_cpu": False,
        "confirm_skull_stripped_mri": False,
        "hd_bet_applied": False,
        "mri_preprocessing_note": "",
        "show_mri_preview": False,
        "mri_rag_explanation": "",
        "mri_rag_sources": [],
        "brain_messages": [],
        "nlp_result": None,
        "nlp_transcript": "",
        "nlp_uploaded_file_name": "",
        "nlp_cleaned_transcript": "",
        "nlp_extracted_features": {},
        "nlp_feature_vector": [],
        "nlp_confidence": None,
        "nlp_patient_info": {},
        "nlp_analysis_features": {},
        "nlp_llm_explanation": "",
        "speech_messages": [],
        "patient_report": None,
        "patient_report_path": None,
        "patient_report_html_path": None,
        "patient_report_pdf_path": None,
        "combined_report_path": None,
        "combined_report_html_path": None,
        "combined_report_pdf_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Make sure a Patient Case ID always exists for the current session.
    ensure_patient_case_id()


def reset_current_patient_case() -> None:
    """
    Start a brand-new patient case: a new Patient Case ID is generated and the
    current session results are cleared. Saved history and previously generated
    reports on disk are NOT deleted.
    """
    st.session_state["patient_case_id"] = generate_patient_case_id()
    cleared = {
        "cnn_analysis_id": "",
        "nlp_analysis_id": "",
        "cnn_result": None,
        "gradcam_paths": [],
        "gradcam_display_mode": "Overlay view",
        "gradcam_generated_orientation": None,
        "gradcam_generated_display_mode": None,
        "cnn_report_path": None,
        "cnn_report_html_path": None,
        "cnn_report_pdf_path": None,
        "cnn_preview_path": None,
        "mri_path": None,
        "mri_original_path": None,
        "mri_processed_path": None,
        "mri_source_name": None,
        "apply_hd_bet_cpu": False,
        "confirm_skull_stripped_mri": False,
        "hd_bet_applied": False,
        "mri_preprocessing_note": "",
        "show_mri_preview": False,
        "mri_rag_explanation": "",
        "mri_rag_sources": [],
        "brain_messages": [],
        "nlp_result": None,
        "nlp_transcript": "",
        "nlp_uploaded_file_name": "",
        "nlp_cleaned_transcript": "",
        "nlp_extracted_features": {},
        "nlp_feature_vector": [],
        "nlp_confidence": None,
        "nlp_patient_info": {},
        "nlp_analysis_features": {},
        "nlp_llm_explanation": "",
        "speech_messages": [],
        "patient_report": None,
        "patient_report_path": None,
        "patient_report_html_path": None,
        "patient_report_pdf_path": None,
        "combined_report_path": None,
        "combined_report_html_path": None,
        "combined_report_pdf_path": None,
    }
    for key, value in cleared.items():
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Theme (premium, colored, glassmorphism)
# ---------------------------------------------------------------------------
def inject_theme() -> None:  # noqa: C901
    """Premium minimal clinical theme — white + green only, no icons."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
        /* ============================================================
           NEUROMIND — PREMIUM CLINICAL PLATFORM
           Palette: #F7FBF8 #FFFFFF #0F8F75 #064E3B #DDF7ED #EEFDF5
                    #10231E #6B7C75 #DCEFE6
           ============================================================ */
        :root{
            --bg:#F7FBF8; --bg-soft:#EEF8F2; --card:#FFFFFF;
            --green:#0F8F75; --green-d:#0B6F5B; --green-deep:#064E3B;
            --green-l:#DDF7ED; --green-xl:#EEFDF5;
            --txt:#10231E; --txt-2:#2C4A40; --muted:#6B7C75; --faint:#9AACA4;
            --line:#DCEFE6; --line-2:#C7E6D7;
            --ok:#0F8F75; --ok-bg:#EEFDF5; --ok-br:#BFE9D8; --ok-tx:#0B5F4B;
            --warn:#B7791F; --warn-bg:#FBF6E9; --warn-br:#EAD9AD; --warn-tx:#6B5417;
            --info:#0E7C7B; --info-bg:#E9F7F6; --info-br:#BFE3E2; --info-tx:#0B4F4E;
            --err:#B4322B; --err-bg:#FBEEED; --err-br:#EBC9C6; --err-tx:#6E1F1B;
            --r-sm:10px; --r-md:14px; --r-lg:18px;
            --sd-xs:0 1px 2px rgba(6,78,59,0.05);
            --sd-sm:0 2px 10px rgba(15,143,117,0.07);
            --sd-md:0 6px 22px rgba(15,143,117,0.09);
            --sd-lg:0 14px 40px rgba(15,143,117,0.12);
            --ease:cubic-bezier(0.22,1,0.36,1);
        }

        /* ---------- BASE ---------- */
        *,*::before,*::after{ box-sizing:border-box; }
        html,body,[class*="css"],.stApp,[data-testid="stMarkdownContainer"]{
            font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
            -webkit-font-smoothing:antialiased;
        }
        .stApp{
            background:
                radial-gradient(820px 520px at 15% -5%, rgba(15,143,117,0.05), transparent 60%),
                radial-gradient(680px 460px at 92% 4%, rgba(15,143,117,0.04), transparent 58%),
                var(--bg) !important;
            color:var(--txt);
        }
        [data-testid="stAppViewContainer"],[data-testid="stMain"],
        [data-testid="stMainBlockContainer"],.main,.main>div,section.main{ background:transparent !important; }
        [data-testid="stHeader"]{ background:transparent !important; box-shadow:none !important; }
        .block-container{ padding-top:1.1rem !important; padding-bottom:3.5rem !important; max-width:1120px; }
        h1,h2,h3,h4,h5{
            font-family:'Manrope',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
            color:var(--txt) !important; letter-spacing:-0.03em; line-height:1.2;
        }
        p,li{ color:var(--txt-2) !important; }
        label{ color:var(--muted) !important; font-size:0.85rem !important; }
        a{ color:var(--green); }

        /* ---------- ANIMATIONS (subtle) ---------- */
        @keyframes fadeUp{ from{opacity:0;transform:translateY(10px);} to{opacity:1;transform:translateY(0);} }
        @keyframes fadeIn{ from{opacity:0;} to{opacity:1;} }
        @keyframes growBar{ from{width:0;} }
        .fu,.fade-up{ animation:fadeUp 0.42s var(--ease) both; }
        .fi,.fade-in,.afi{ animation:fadeIn 0.34s ease both; }

        /* ---------- BORDERED CONTAINERS (st.container border=True) ---------- */
        [data-testid="stVerticalBlockBorderWrapper"]{
            background:var(--card) !important;
            border:1px solid var(--line) !important;
            border-radius:var(--r-lg) !important;
            box-shadow:var(--sd-sm) !important;
            transition:border-color .2s var(--ease), box-shadow .2s var(--ease), transform .2s var(--ease);
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover{
            border-color:var(--line-2) !important; box-shadow:var(--sd-md) !important;
        }

        /* ---------- METRICS ---------- */
        [data-testid="stMetric"]{
            background:var(--green-xl) !important; border:1px solid var(--line) !important;
            border-radius:var(--r-md) !important; padding:15px 18px !important; box-shadow:none !important;
        }
        [data-testid="stMetricLabel"] p{
            font-size:0.68rem !important; font-weight:700 !important; letter-spacing:0.07em !important;
            text-transform:uppercase !important; color:var(--muted) !important;
        }
        [data-testid="stMetricValue"]{
            font-weight:800 !important; color:var(--green-deep) !important;
            font-size:1.55rem !important; letter-spacing:-0.02em !important;
        }
        [data-testid="stMetricDelta"]{ display:none; }

        /* ---------- INPUTS ---------- */
        [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,
        [data-testid="stDateInput"] input,[data-testid="stNumberInput"] input{
            background:#fff !important; border:1px solid var(--line) !important;
            color:var(--txt) !important; border-radius:var(--r-sm) !important;
            font-size:0.88rem !important; caret-color:var(--green);
        }
        [data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus,
        [data-testid="stNumberInput"] input:focus{
            border-color:var(--green) !important; box-shadow:0 0 0 3px rgba(15,143,117,0.12) !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"]>div:first-child{
            background:#fff !important; border-color:var(--line) !important;
            color:var(--txt) !important; border-radius:var(--r-sm) !important;
        }
        ::placeholder{ color:var(--faint) !important; }
        [data-testid="stFileUploader"]{
            background:var(--green-xl) !important; border:2px dashed var(--line-2) !important;
            border-radius:var(--r-md) !important; padding:12px !important; transition:all .2s var(--ease);
        }
        [data-testid="stFileUploader"]:hover,[data-testid="stFileUploader"]:focus-within{
            border-color:var(--green) !important; background:var(--green-l) !important;
        }
        [data-testid="stFileUploader"] section>span,[data-testid="stFileUploader"] section small{ color:var(--muted) !important; }
        [data-testid="stExpander"]{
            background:var(--card) !important; border:1px solid var(--line) !important; border-radius:var(--r-md) !important;
        }
        [data-testid="stExpander"] summary,[data-testid="stExpander"] summary>span{ color:var(--txt-2) !important; }
        [data-testid="stRadio"] [aria-checked="true"]>div:first-child{ border-color:var(--green) !important; background:var(--green) !important; }
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{ background:var(--green) !important; border-color:var(--green) !important; }
        [data-testid="stImageCaption"]{ color:var(--faint) !important; font-size:0.76rem !important; }
        [data-testid="stNumberInput"] button{ background:var(--green-xl) !important; color:var(--muted) !important; border:1px solid var(--line) !important; }
        .stSpinner>div>div{ border-top-color:var(--green) !important; }

        /* ---------- BUTTONS (green, modern) ---------- */
        .stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{
            background:var(--green) !important; color:#fff !important; border:1px solid var(--green) !important;
            border-radius:var(--r-sm) !important; font-weight:600 !important; font-size:0.87rem !important;
            letter-spacing:0.01em !important; padding:0.55rem 1.15rem !important;
            box-shadow:0 2px 10px rgba(15,143,117,0.18) !important; transition:all .18s var(--ease) !important;
        }
        .stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{
            background:var(--green-d) !important; border-color:var(--green-d) !important;
            transform:translateY(-1px) !important; box-shadow:0 6px 18px rgba(15,143,117,0.26) !important; color:#fff !important;
        }
        .stButton>button:active{ transform:translateY(0) !important; }
        .stButton>button:disabled{
            background:var(--green-xl) !important; color:var(--faint) !important;
            border-color:var(--line) !important; box-shadow:none !important; transform:none !important;
        }

        /* ---------- SIDEBAR (white / pale green, no icons) ---------- */
        [data-testid="stSidebar"]{
            background:linear-gradient(180deg,#FFFFFF 0%, var(--green-xl) 100%) !important;
            border-right:1px solid var(--line) !important; box-shadow:none !important;
            min-width:248px !important; max-width:268px !important;
        }
        [data-testid="stSidebar"] > div{ background:transparent !important; }
        /* sidebar nav buttons: ghost by default, green when active (type=primary) */
        [data-testid="stSidebar"] .stButton>button{
            background:transparent !important; color:var(--txt-2) !important;
            border:1px solid transparent !important; box-shadow:none !important;
            text-align:left !important; padding:9px 13px !important; font-weight:500 !important;
        }
        [data-testid="stSidebar"] .stButton>button:hover{
            background:#fff !important; color:var(--green-deep) !important;
            border-color:var(--line) !important; transform:none !important; box-shadow:var(--sd-xs) !important;
        }
        [data-testid="stSidebar"] .stButton>button[kind="primary"],
        [data-testid="stSidebar"] .stButton>button[data-testid="baseButton-primary"]{
            background:var(--green) !important; color:#fff !important; border-color:var(--green) !important;
            font-weight:600 !important; box-shadow:0 2px 10px rgba(15,143,117,0.18) !important;
        }
        [data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{
            background:var(--green-d) !important; border-color:var(--green-d) !important;
        }
        .sb-brand{ padding:16px 6px 14px; border-bottom:1px solid var(--line); margin-bottom:12px; }
        .sb-brand-name{ font-weight:800; font-size:1.12rem; color:var(--green-deep); letter-spacing:-0.02em; }
        .sb-brand-tag{ font-size:0.64rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted); margin-top:3px; }
        .sb-case{ background:#fff; border:1px solid var(--line); border-radius:var(--r-md); padding:11px 13px; margin-bottom:12px; box-shadow:var(--sd-xs); }
        .sb-case-lbl{ font-size:0.6rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-bottom:5px; }
        .sb-case-id{ font-family:'JetBrains Mono','SF Mono',monospace; font-size:0.74rem; font-weight:600; color:var(--green-deep);
            background:var(--green-l); border:1px solid var(--line-2); border-radius:8px; padding:6px 9px; word-break:break-all; letter-spacing:0.02em; }
        .sb-nav-label{ font-size:0.6rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:var(--muted); margin:14px 4px 6px; }
        .sb-note{ font-size:0.68rem; color:var(--muted); line-height:1.55; padding:12px 4px 4px; border-top:1px solid var(--line); margin-top:12px; }

        /* ---------- PAGE HEADER ---------- */
        .page-head{ margin:2px 0 16px; }
        .page-eyebrow{ font-size:0.64rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:var(--green); margin-bottom:7px; }
        .page-title{ font-weight:800; font-size:1.7rem; color:var(--txt); letter-spacing:-0.03em; line-height:1.15; }
        .page-sub{ font-size:0.92rem; color:var(--muted); margin-top:6px; }
        .block-gap{ height:14px; }

        /* ---------- ACTIVE CASE PANEL ---------- */
        .case-panel{
            background:linear-gradient(135deg,#fff 0%, var(--green-xl) 100%);
            border:1px solid var(--line); border-left:4px solid var(--green);
            border-radius:var(--r-lg); padding:18px 22px; box-shadow:var(--sd-sm);
        }
        .case-panel-lbl{ font-size:0.62rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); }
        .case-panel-id{ font-family:'JetBrains Mono','SF Mono',monospace; font-size:0.98rem; font-weight:600; color:var(--green-deep); margin-top:6px; letter-spacing:0.02em; }
        .case-panel-note{ font-size:0.8rem; color:var(--muted); margin-top:9px; line-height:1.55; }

        /* ---------- DASHBOARD MODULES ---------- */
        .module{ padding:4px 2px 2px; }
        .module-title{ font-weight:800; font-size:1.04rem; color:var(--txt); letter-spacing:-0.01em; }
        .module-desc{ color:var(--muted); font-size:0.85rem; line-height:1.55; margin-top:7px; min-height:42px; }
        .module-status{ display:inline-block; font-size:0.62rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase;
            padding:3px 10px; border-radius:999px; margin-top:12px; }
        .module-status.is-done{ background:var(--ok-bg); color:var(--ok-tx); border:1px solid var(--ok-br); }
        .module-status.is-idle{ background:var(--bg-soft); color:var(--muted); border:1px solid var(--line); }

        /* ---------- CLINICAL PATHWAY ---------- */
        .path-strip{ background:var(--card); border:1px solid var(--line); border-radius:var(--r-lg); padding:18px 22px; box-shadow:var(--sd-xs); }
        .path-title{ font-size:0.66rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted); margin-bottom:14px; }
        .path-row{ display:flex; align-items:flex-start; }
        .path-step{ flex:1; display:flex; flex-direction:column; align-items:center; gap:7px; position:relative; }
        .path-step + .path-step::before{ content:''; position:absolute; right:100%; top:13px; width:calc(100% - 28px); height:1.5px; background:var(--line); transform:translateX(50%); }
        .path-step.has-done + .path-step::before{ background:var(--line-2); }
        .path-dot{ width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:800; z-index:1; }
        .path-dot.done{ background:var(--green); color:#fff; box-shadow:0 2px 8px rgba(15,143,117,0.28); }
        .path-dot.idle{ background:#fff; color:var(--faint); border:1.5px solid var(--line); }
        .path-lbl{ font-size:0.64rem; color:var(--muted); text-align:center; line-height:1.3; max-width:84px; }
        .path-lbl.done{ color:var(--txt-2); font-weight:600; }

        /* ---------- REPORTS: SUMMARY + DOCS + TIMELINE ---------- */
        .summary-panel{ background:var(--card); border:1px solid var(--line); border-radius:var(--r-lg); padding:18px 20px; box-shadow:var(--sd-xs); }
        .summary-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
        .summary-item{ }
        .summary-lbl{ font-size:0.6rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); margin-bottom:5px; }
        .summary-val{ font-weight:800; font-size:1.18rem; color:var(--green-deep); letter-spacing:-0.01em; }
        .summary-val.mono{ font-family:'JetBrains Mono','SF Mono',monospace; font-size:0.84rem; font-weight:600; word-break:break-all; }
        .doc-group-label{ font-size:0.66rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--green); margin:16px 0 8px; }
        .doc-card{ background:var(--card); border:1px solid var(--line); border-left:3px solid var(--green); border-radius:var(--r-md); padding:14px 18px; box-shadow:var(--sd-xs); margin-bottom:8px; transition:border-color .18s var(--ease), box-shadow .18s var(--ease); }
        .doc-card:hover{ border-color:var(--line-2); box-shadow:var(--sd-sm); }
        .doc-kind{ font-size:0.62rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--green); }
        .doc-title{ font-weight:600; font-size:0.9rem; color:var(--txt); margin-top:3px; }
        .doc-date{ font-size:0.76rem; color:var(--muted); margin-top:3px; }
        .tl-row{ display:flex; gap:14px; padding:2px 0 14px; }
        .tl-line{ position:relative; width:14px; flex:0 0 14px; display:flex; justify-content:center; }
        .tl-line::before{ content:''; position:absolute; top:6px; bottom:-14px; width:2px; background:var(--line-2); }
        .tl-row:last-child .tl-line::before{ display:none; }
        .tl-node{ width:11px; height:11px; border-radius:50%; background:var(--green); border:2px solid #fff; box-shadow:0 0 0 2px var(--green-l); margin-top:3px; z-index:1; }
        .tl-body{ flex:1; }
        .tl-kind{ font-size:0.62rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; color:var(--green); }
        .tl-main{ font-weight:600; font-size:0.9rem; color:var(--txt); margin-top:2px; }
        .tl-date{ font-size:0.76rem; color:var(--muted); margin-top:2px; }

        /* ---------- SHARED COMPONENTS (used by Brain / Speech / About) ---------- */
        .section-hd,.section-title{ font-weight:700; font-size:1.1rem; color:var(--txt); margin:4px 0 3px; display:flex; align-items:center; gap:10px; }
        .section-hd::before,.section-title::before{ content:''; width:4px; height:18px; border-radius:2px; background:var(--green); display:inline-block; flex-shrink:0; }
        .section-sub{ color:var(--muted); font-size:0.84rem; margin:2px 0 6px 14px; line-height:1.5; }

        .note{ border-radius:var(--r-sm); padding:11px 16px; margin:8px 0; font-size:0.87rem; border:1px solid var(--line); background:var(--bg-soft); color:var(--txt-2); border-left:4px solid var(--muted); line-height:1.55; }
        .note-info{ background:var(--info-bg); border-color:var(--info-br); border-left-color:var(--info); color:var(--info-tx); }
        .note-success{ background:var(--ok-bg); border-color:var(--ok-br); border-left-color:var(--ok); color:var(--ok-tx); }
        .note-warning{ background:var(--warn-bg); border-color:var(--warn-br); border-left-color:var(--warn); color:var(--warn-tx); }
        .note-error{ background:var(--err-bg); border-color:var(--err-br); border-left-color:var(--err); color:var(--err-tx); }

        .step-card{ display:flex; gap:14px; align-items:flex-start; background:var(--card); border:1px solid var(--line); border-radius:var(--r-md); padding:14px 18px; box-shadow:var(--sd-xs); margin:6px 0; }
        .step-num{ flex-shrink:0; width:32px; height:32px; border-radius:9px; background:var(--green); color:#fff; font-weight:800; font-size:0.92rem; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 8px rgba(15,143,117,0.24); }
        .step-t{ font-weight:700; font-size:0.94rem; color:var(--txt); margin-bottom:3px; }
        .step-x{ color:var(--muted); font-size:0.85rem; line-height:1.5; }

        .result-card{ border-radius:var(--r-lg); padding:22px 24px; margin:6px 0 14px; border:1px solid var(--line); background:var(--card); box-shadow:var(--sd-sm); position:relative; overflow:hidden; }
        .result-card::before{ content:''; position:absolute; left:0; top:0; bottom:0; width:5px; }
        .result-good{ background:linear-gradient(135deg,var(--green-xl),var(--card)); }
        .result-good::before{ background:var(--green); }
        .result-alert{ background:linear-gradient(135deg,var(--warn-bg),var(--card)); border-color:var(--warn-br); }
        .result-alert::before{ background:var(--warn); }
        .rc-top{ display:flex; align-items:center; justify-content:space-between; gap:14px; }
        .rc-label{ font-size:0.67rem; font-weight:700; letter-spacing:0.13em; text-transform:uppercase; color:var(--muted); margin-bottom:5px; }
        .rc-value{ font-weight:800; font-size:1.55rem; line-height:1.1; letter-spacing:-0.02em; color:var(--green-deep); }
        .result-alert .rc-value{ color:#9A6212; }
        .rc-chip{ width:54px; height:54px; border-radius:15px; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.95rem; color:#fff; background:var(--green); }
        .result-alert .rc-chip{ background:var(--warn); }
        .rc-bar-wrap{ margin-top:14px; }
        .rc-bar-labels{ display:flex; justify-content:space-between; margin-bottom:6px; }
        .rc-bar-lbl{ font-size:0.72rem; font-weight:500; color:var(--muted); }
        .rc-bar-val{ font-size:0.84rem; font-weight:700; color:var(--green); }
        .result-alert .rc-bar-val{ color:#9A6212; }
        .rc-bar{ height:7px; background:var(--green-l); border-radius:999px; overflow:hidden; }
        .rc-fill{ height:100%; border-radius:999px; background:var(--green); transition:width .9s var(--ease); animation:growBar .9s var(--ease); }
        .result-alert .rc-fill{ background:var(--warn); }
        .rc-exp{ margin-top:14px; color:var(--txt-2); font-size:0.88rem; line-height:1.62; padding-top:12px; border-top:1px solid var(--line); }
        /* legacy confidence aliases */
        .rc-conf{ margin-top:8px; font-size:0.84rem; color:var(--muted); }
        .rc-conf b{ color:var(--green); }
        .confidence-track{ height:7px; background:var(--green-l); border-radius:999px; overflow:hidden; }
        .confidence-fill{ height:100%; border-radius:999px; background:var(--green); }
        .confidence-lbl{ color:var(--muted); } .confidence-pct{ color:var(--green); font-weight:700; }

        .explain-card{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--green); border-radius:var(--r-md); padding:16px 18px; margin:10px 0; box-shadow:var(--sd-xs); }
        .explain-card.explain-warning{ background:var(--warn-bg); border-left-color:var(--warn); }
        .explain-title{ font-weight:800; color:var(--txt); font-size:0.94rem; margin-bottom:6px; }
        .explain-body{ color:var(--muted); font-size:0.88rem; line-height:1.6; }
        .explain-list{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:10px; }
        .explain-item{ background:var(--green-xl); border:1px solid var(--line); border-radius:8px; padding:9px 11px; color:var(--muted); font-size:0.83rem; line-height:1.42; }
        .explain-item strong{ color:var(--green-deep); }

        .legend-wrap,.color-legend{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:10px; }
        .legend-item,.legend-row{ display:flex; align-items:center; gap:9px; background:var(--green-xl); border:1px solid var(--line); border-radius:8px; padding:10px; color:var(--muted); font-size:0.83rem; }
        .legend-dot,.l-dot{ width:13px; height:13px; border-radius:50%; flex-shrink:0; }
        .l-low,.legend-low{ background:var(--green-deep); }
        .l-mid,.legend-mid{ background:var(--green); }
        .l-high,.legend-high{ background:var(--green-l); border:1px solid var(--line-2); }

        /* Report cards (About / combined section legacy) */
        .report-card,.rpt-card{ background:var(--card); border:1px solid var(--line); border-radius:var(--r-md); padding:16px 18px; box-shadow:var(--sd-xs); margin-bottom:10px; }
        .report-meta,.rpt-meta{ color:var(--muted); font-size:0.84rem; line-height:1.55; }
        .report-meta b{ color:var(--txt); }

        /* Combined-report assessment status (Étape 2 keeps its icons; styled green here) */
        .assess-status-grid,.assess-grid{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }
        .assess-status-card,.assess-card{ background:var(--card); border:1px solid var(--line); border-radius:var(--r-md); padding:18px 20px; box-shadow:var(--sd-xs); display:flex; align-items:center; gap:14px; }
        .assess-icon{ width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.2rem; flex-shrink:0; }
        .assess-icon-ok{ background:var(--ok-bg); border:1px solid var(--ok-br); }
        .assess-icon-miss{ background:var(--bg-soft); border:1px solid var(--line); }
        .assess-lbl{ font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin-bottom:3px; }
        .assess-val{ font-weight:700; font-size:0.94rem; }
        .assess-val-ok{ color:var(--ok); } .assess-val-miss{ color:var(--muted); }
        .assess-badge-ok{ display:inline-flex; font-size:0.59rem; font-weight:800; text-transform:uppercase; padding:2px 8px; border-radius:999px; background:var(--ok-bg); color:var(--ok); border:1px solid var(--ok-br); margin-top:4px; }
        .assess-badge-miss{ display:inline-flex; font-size:0.59rem; font-weight:800; text-transform:uppercase; padding:2px 8px; border-radius:999px; background:var(--bg-soft); color:var(--muted); border:1px solid var(--line); margin-top:4px; }

        /* Clinical insight panels (Brain / Speech Q&A — no chat-bubble chrome) */
        .clinical-panel-wrap{ background:var(--green-xl); border:1px solid var(--line); border-top:3px solid var(--green); border-radius:var(--r-lg); margin:10px 0; overflow:hidden; }
        .clinical-panel-header{ display:flex; align-items:center; gap:12px; padding:16px 20px 14px; background:var(--green-l); border-bottom:1px solid var(--line); }
        .cp-icon{ width:40px; height:40px; border-radius:11px; background:var(--green); display:flex; align-items:center; justify-content:center; color:#fff; flex-shrink:0; }
        .cp-icon:empty{ display:none; }            /* hide empty icon box after de-iconifying */
        .cp-title{ font-weight:700; font-size:0.94rem; color:var(--green-deep); }
        .cp-sub{ font-size:0.76rem; color:var(--muted); margin-top:2px; line-height:1.4; }
        .cp-badge{ margin-left:auto; display:inline-flex; align-items:center; gap:5px; font-size:0.58rem; font-weight:800; text-transform:uppercase; letter-spacing:0.08em; padding:3px 9px; border-radius:999px; background:var(--green-l); color:var(--green); border:1px solid var(--line-2); }
        .cp-badge-dot{ width:5px; height:5px; border-radius:50%; background:var(--green); }
        [data-testid="stChatMessageContent"]{ background:#fff !important; border:1px solid var(--line) !important; border-radius:12px !important; color:var(--txt-2) !important; }
        [data-testid="stChatInput"] textarea{ background:#fff !important; color:var(--txt) !important; }
        [data-testid="stChatInput"] textarea:focus{ border-color:var(--green) !important; box-shadow:0 0 0 3px rgba(15,143,117,0.12) !important; }

        /* Patient case card (de-iconified) */
        .case-card{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--green); border-radius:var(--r-md); padding:16px 18px; box-shadow:var(--sd-xs); }
        .case-lbl{ font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:var(--muted); margin-bottom:4px; }
        .case-id-val{ font-family:'JetBrains Mono','SF Mono',monospace; font-size:0.84rem; font-weight:600; color:var(--green-deep); background:var(--green-l); border:1px solid var(--line-2); border-radius:7px; padding:5px 10px; letter-spacing:0.02em; display:inline-block; }
        .case-note{ font-size:0.76rem; color:var(--muted); margin-top:10px; line-height:1.5; }

        /* Hero (kept minimal; only used by unused helper) */
        .hero{ background:linear-gradient(135deg,var(--green-deep),var(--green)); border-radius:var(--r-lg); padding:34px 40px; color:#fff; }
        .hero-title{ color:#fff !important; } .hero-sub{ color:rgba(255,255,255,0.8) !important; }

        /* Kill any floating chat affordance */
        .fc-toggle,.fc-btn,.fc-panel{ display:none !important; pointer-events:none !important; }

        /* ============================================================
           COMMAND CENTER — clinical dashboard (white + green, animated)
           ============================================================ */
        .cmd-head{ display:flex; align-items:center; justify-content:space-between; gap:16px;
            background:var(--card); border:1px solid var(--line); border-radius:var(--r-lg);
            padding:16px 22px; box-shadow:var(--sd-sm); }
        .cmd-left{ display:flex; align-items:center; gap:13px; min-width:0; }
        .cmd-mark{ width:42px; height:42px; border-radius:11px; background:var(--green-l);
            color:var(--green-deep); display:flex; align-items:center; justify-content:center;
            font-weight:800; font-size:1.15rem; flex:0 0 auto; }
        .cmd-name{ font-weight:800; font-size:1.1rem; color:var(--green-deep); letter-spacing:-0.02em; line-height:1.15; }
        .cmd-tag{ font-size:0.66rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted); margin-top:2px; }
        .cmd-right{ display:flex; align-items:center; gap:16px; flex:0 0 auto; }
        .cmd-live{ display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:20px;
            background:var(--green-xl); border:1px solid var(--line-2); font-size:0.74rem; color:var(--green-deep); }
        .cmd-live .dot{ width:6px; height:6px; border-radius:50%; background:var(--green); animation:nmBlink 2.2s ease-in-out infinite; }
        .cmd-caseline{ display:flex; align-items:center; gap:10px; margin:12px 4px 0; }
        .cmd-caseline .lbl{ font-size:0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); }
        .cmd-caseline .id{ font-family:'JetBrains Mono','SF Mono',monospace; font-size:0.8rem; color:var(--txt-2); letter-spacing:0.02em; }
        @keyframes nmBlink{ 0%,100%{ opacity:1; } 50%{ opacity:.35; } }

        .nm-eeg-line{ stroke:#0F8F75; stroke-width:2; fill:none; stroke-linecap:round; stroke-linejoin:round;
            stroke-dasharray:6 6; animation:nmEeg 3.2s linear infinite; }
        @keyframes nmEeg{ to{ stroke-dashoffset:-260; } }

        .nm-ring-arc{ animation:nmRingDraw 1.3s cubic-bezier(.22,1,.36,1) both; }
        @keyframes nmRingDraw{ from{ stroke-dashoffset:339.29; } }
        .nm-ring-num{ font-weight:800; font-size:30px; fill:var(--green-deep); }
        .nm-ring-sub{ font-size:11px; fill:var(--muted); }
        .nm-gauge-arc{ animation:nmGaugeDraw 1.3s cubic-bezier(.22,1,.36,1) both; }
        @keyframes nmGaugeDraw{ from{ stroke-dashoffset:257.6; } }
        .nm-gauge-num{ font-weight:800; font-size:30px; fill:var(--green-deep); }

        .ring-card{ display:flex; align-items:center; gap:18px; }
        .ring-list{ display:flex; flex-direction:column; gap:10px; font-size:0.86rem; }
        .ring-li{ display:flex; align-items:center; gap:9px; color:var(--txt); }
        .ring-li .pip{ width:9px; height:9px; border-radius:50%; flex:0 0 auto; }
        .ring-li.done .pip{ background:var(--green); }
        .ring-li.idle{ color:var(--muted); } .ring-li.idle .pip{ background:var(--faint); }

        .snap-grid{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }
        .snap{ background:var(--green-xl); border:1px solid var(--line); border-radius:var(--r-md); padding:13px 15px; }
        .snap-lbl{ font-size:0.6rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; color:var(--muted); }
        .snap-val{ font-weight:800; font-size:1.4rem; color:var(--green-deep); margin-top:4px; letter-spacing:-0.02em; }

        .nm-tile{ position:relative; overflow:hidden; padding:2px 2px 2px 8px; }
        .nm-tile-acc{ position:absolute; left:0; top:2px; width:3px; height:30px; background:var(--green); transition:height .28s var(--ease); }
        [data-testid="stVerticalBlockBorderWrapper"]:hover .nm-tile-acc{ height:calc(100% - 4px); }
        .nm-tile-title{ font-weight:800; font-size:1.04rem; color:var(--txt); letter-spacing:-0.01em; }
        .nm-tile-desc{ color:var(--muted); font-size:0.84rem; line-height:1.5; margin-top:6px; min-height:38px; }
        .nm-tile-status{ display:inline-block; font-size:0.62rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; padding:3px 10px; border-radius:999px; margin-top:11px; }
        .nm-tile-status.is-done{ background:var(--ok-bg); color:var(--ok-tx); border:1px solid var(--ok-br); }
        .nm-tile-status.is-idle{ background:var(--bg-soft); color:var(--muted); border:1px solid var(--line); }

        .pw-row{ display:flex; align-items:flex-start; }
        .pw-step{ flex:1; display:flex; flex-direction:column; align-items:center; gap:7px; }
        .pw-seg{ flex:0 0 auto; height:2px; width:30px; margin-top:14px; background:var(--line); }
        .pw-seg.on{ background:var(--green); }
        .pw-dot{ position:relative; width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.78rem; font-weight:800; }
        .pw-dot.done{ background:var(--green); color:#fff; box-shadow:0 2px 8px rgba(15,143,117,0.28); }
        .pw-dot.cur{ background:var(--green-l); color:var(--green-deep); border:1.5px solid var(--green); }
        .pw-dot.cur::after{ content:''; position:absolute; inset:-4px; border-radius:50%; border:1.5px solid var(--green); animation:nmPulse 2s ease-in-out infinite; }
        .pw-dot.idle{ background:#fff; color:var(--faint); border:1.5px solid var(--line); }
        .pw-lbl{ font-size:0.64rem; color:var(--muted); text-align:center; line-height:1.3; max-width:84px; }
        .pw-lbl.on{ color:var(--txt-2); font-weight:600; }
        @keyframes nmPulse{ 0%,100%{ transform:scale(1); opacity:.5; } 50%{ transform:scale(1.35); opacity:0; } }

        /* result card with radial gauge */
        .rc-grid{ display:flex; align-items:center; gap:22px; }
        .rc-gauge{ flex:0 0 auto; }
        .rc-main{ flex:1; min-width:0; }
        .result-alert{ background:linear-gradient(135deg,var(--green-xl),var(--card)) !important; border-color:var(--line-2) !important; }
        .result-alert::before{ background:var(--green-deep) !important; }
        .result-alert .rc-value{ color:var(--green-deep) !important; }

        @media (max-width:860px){
            .summary-grid,.explain-list,.legend-wrap,.color-legend,
            .assess-status-grid,.assess-grid,.snap-grid{ grid-template-columns:1fr; }
            .rc-grid,.ring-card{ flex-direction:column; align-items:flex-start; }
        }

        /* ── PAGE HEADER IMPROVEMENTS ─────────────────────────────── */
        .page-eyebrow{ letter-spacing:0.2em; }
        .page-title{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:900; font-size:1.9rem; letter-spacing:-0.04em;
        }
        .section-hd,.section-title{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:800; font-size:1.12rem; letter-spacing:-0.02em;
        }
        .cmd-name{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:900; letter-spacing:-0.03em;
        }
        .nm-tile-title{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:800;
        }

        /* ── SIDEBAR POLISH ───────────────────────────────────────── */
        .sb-brand{
            padding:20px 8px 16px;
            border-bottom:1px solid var(--line);
            margin-bottom:14px;
            background:linear-gradient(135deg,var(--green-xl) 0%,#fff 100%);
            border-radius:0 0 var(--r-md) var(--r-md);
            margin-top:-8px;
        }
        .sb-brand-name{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:900; font-size:1.22rem; color:var(--green-deep);
            letter-spacing:-0.03em;
        }
        .sb-brand-tag{
            font-size:0.58rem; font-weight:800; letter-spacing:0.2em;
            text-transform:uppercase; color:var(--green); margin-top:3px;
        }
        .sb-case{ transition:box-shadow .18s var(--ease); }
        .sb-case:hover{ box-shadow:var(--sd-sm) !important; }

        /* ── RESULT CARD POLISH ───────────────────────────────────── */
        .rc-label{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:700; font-size:0.7rem; letter-spacing:0.14em;
        }
        .rc-value{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:900; font-size:1.65rem; letter-spacing:-0.03em;
        }

        /* ── SECONDARY (GHOST) BUTTONS ────────────────────────────── */
        .stButton>button[kind="secondary"],[data-testid="baseButton-secondary"]{
            background:var(--card) !important;
            color:var(--green-deep) !important;
            border:1px solid var(--line-2) !important;
            box-shadow:none !important;
        }
        .stButton>button[kind="secondary"]:hover,[data-testid="baseButton-secondary"]:hover{
            background:var(--green-xl) !important;
            border-color:var(--green) !important;
            color:var(--green-deep) !important;
            transform:translateY(-1px) !important;
            box-shadow:0 3px 10px rgba(15,143,117,0.12) !important;
        }
        /* Keep sidebar nav secondary buttons styled correctly */
        [data-testid="stSidebar"] .stButton>button[kind="secondary"]{
            background:transparent !important;
            color:var(--txt-2) !important;
            border-color:transparent !important;
            box-shadow:none !important;
        }
        [data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover{
            background:#fff !important;
            color:var(--green-deep) !important;
            border-color:var(--line) !important;
            transform:none !important;
            box-shadow:var(--sd-xs) !important;
        }

        /* ── CLINICAL INSIGHT PANEL (no chat bubbles) ─────────────── */
        .cip-header{
            background:linear-gradient(140deg,var(--card) 0%,var(--green-xl) 100%);
            border:1px solid var(--line); border-top:4px solid var(--green);
            border-radius:var(--r-lg); padding:22px 26px 18px;
            display:flex; align-items:flex-start;
            justify-content:space-between; gap:14px;
            box-shadow:0 4px 20px rgba(15,143,117,0.08),0 1px 4px rgba(0,0,0,0.03);
            margin:14px 0 12px;
        }
        .cip-left{ flex:1; min-width:0; }
        .cip-eyebrow{
            font-size:0.58rem; font-weight:800; letter-spacing:0.22em;
            text-transform:uppercase; color:var(--green); margin-bottom:6px;
        }
        .cip-title{
            font-family:'Manrope',-apple-system,sans-serif;
            font-weight:900; font-size:1.12rem; color:var(--green-deep);
            letter-spacing:-0.03em; line-height:1.2;
        }
        .cip-sub{
            font-size:0.84rem; color:var(--muted); margin-top:6px;
            line-height:1.52; max-width:560px;
        }
        .cip-status{
            display:inline-flex; align-items:center; gap:6px;
            padding:5px 13px; border-radius:999px;
            background:var(--green-xl); border:1px solid var(--line-2);
            font-size:0.6rem; font-weight:800; letter-spacing:0.1em;
            text-transform:uppercase; color:var(--green);
            white-space:nowrap; flex-shrink:0; margin-top:3px;
        }
        .cip-status-dot{
            width:6px; height:6px; border-radius:50%; background:var(--green);
            animation:nmBlink 2.2s ease-in-out infinite;
        }
        .cip-sugg-hd{
            font-size:0.62rem; font-weight:800; letter-spacing:0.16em;
            text-transform:uppercase; color:var(--muted); margin:12px 2px 6px;
        }
        /* Q&A response card question line */
        .cip-q-line{
            display:flex; align-items:flex-start; gap:10px;
            margin-bottom:12px; padding-bottom:12px;
            border-bottom:1px solid var(--line);
        }
        .cip-q-badge{
            flex-shrink:0; width:24px; height:24px; border-radius:8px;
            background:linear-gradient(135deg,var(--green),var(--green-d));
            color:#fff; font-size:0.65rem; font-weight:800;
            display:inline-flex; align-items:center; justify-content:center;
            margin-top:1px; box-shadow:0 2px 6px rgba(15,143,117,0.25);
        }
        .cip-q-text{
            font-family:'Manrope',-apple-system,sans-serif;
            font-weight:700; font-size:0.94rem; color:var(--txt); line-height:1.4;
        }

        /* ── STEP CARD POLISH ─────────────────────────────────────── */
        .step-card{ transition:box-shadow .2s var(--ease),transform .2s var(--ease); }
        .step-card:hover{ box-shadow:var(--sd-md) !important; transform:translateY(-1px); }
        .step-num{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:900;
        }
        .step-t{
            font-family:'Manrope',-apple-system,sans-serif !important;
            font-weight:800;
        }

        /* ── NM TILE POLISH ───────────────────────────────────────── */
        [data-testid="stVerticalBlockBorderWrapper"]:has(.nm-tile){
            transition:box-shadow .22s var(--ease),transform .22s var(--ease) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.nm-tile):hover{
            box-shadow:var(--sd-lg) !important;
            transform:translateY(-2px) !important;
        }

        /* ── COMMAND HEAD POLISH ──────────────────────────────────── */
        .cmd-head{
            border:1.5px solid var(--line-2) !important;
            transition:box-shadow .2s var(--ease);
        }
        .cmd-head:hover{ box-shadow:var(--sd-md) !important; }

        /* ── DOC CARD POLISH ──────────────────────────────────────── */
        .doc-card{
            transition:box-shadow .2s var(--ease),transform .18s var(--ease),border-color .18s var(--ease);
        }
        .doc-card:hover{ transform:translateY(-1px); box-shadow:var(--sd-sm) !important; }

        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------


def note(text: str, kind: str = "info") -> None:
    st.markdown(
        f'<div class="note note-{kind}">{html.escape(patient_friendly_text(text))}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    sub = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="fu"><div class="section-hd">{html.escape(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def show_error(message: str, exc: Exception | None = None) -> None:
    _ = exc
    note(message, "error")


def hero_card(title: str, subtitle: str, reassure: str) -> None:
    case_id = ensure_patient_case_id()
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-glow"></div>
          <div class="hero-eyebrow">Clinical Decision-Support Platform</div>
          <div class="hero-title">{html.escape(title)}</div>
          <div class="hero-sub">{html.escape(subtitle)}</div>
          <div class="hero-case">
            <div>
              <div class="hero-case-label">Active Patient Case</div>
              <div class="hero-case-id">{html.escape(case_id)}</div>
            </div>
            <div class="hero-case-sep"></div>
            <div class="hero-case-status">
              <span class="dot"></span> Session Active
            </div>
          </div>
          <div class="hero-chips">
            <span class="chip chip-b"><span class="d"></span>Brain Imaging Analysis</span>
            <span class="chip chip-c"><span class="d"></span>Speech &amp; Language Assessment</span>
            <span class="chip chip-w"><span class="d"></span>Clinical Reports</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, text: str, tone: str = "emerald") -> None:
    tone_map = {"emerald": "b", "blue": "b", "violet": "c", "cyan": "c",
                "coral": "i", "indigo": "i", "gold": "s", "teal": "s"}
    ic_cls = f"ic-{tone_map.get(tone, 'b')}"
    st.markdown(
        f"""
        <div class="feat-card fu">
          <div class="feat-ic {ic_cls}">{icon}</div>
          <div class="feat-title">{html.escape(title)}</div>
          <div class="feat-desc">{html.escape(text)}</div>
          <div class="feat-arrow">Open &rarr;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_card(step: int, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="step-card fade-in">
          <div class="step-num">{step}</div>
          <div>
            <div class="step-t">{html.escape(title)}</div>
            <div class="step-x">{html.escape(text)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(title: str, result: str, confidence: str, explanation: str, tone: str, chip: str) -> None:
    try:
        raw = confidence.strip().rstrip("%")
        pct = float(raw)
        if pct <= 1.0:
            pct *= 100
        pct = max(0.0, min(100.0, pct))
    except (ValueError, AttributeError):
        pct = 0.0
    st.markdown(
        f"""
        <div class="result-card result-{html.escape(tone)} fu">
          <div class="rc-grid">
            <div class="rc-gauge">{_gauge_svg(pct, tone)}</div>
            <div class="rc-main">
              <div class="rc-label">{html.escape(title)}</div>
              <div class="rc-value">{html.escape(result)}</div>
              <div class="rc-exp">{html.escape(patient_friendly_text(explanation))}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def medical_note_card(text: str) -> None:
    note(text, "warning")


def patient_case_card(patient_case_id: str) -> None:
    st.markdown(
        f"""
        <div class="case-panel fu">
          <div class="case-panel-lbl">Active Patient Case</div>
          <div class="case-panel-id">{html.escape(patient_case_id)}</div>
          <div class="case-panel-note">This identifier links all assessments, visual explanations, and clinical reports in the current session.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5) REPORT CARD  — replaces report_card()  (text only, no icons)

def explanation_card(title: str, body: str, items: list[tuple[str, str]] | None = None, kind: str = "info") -> None:
    items_html = ""
    if items:
        parts = [
            f'<div class="explain-item"><strong>{html.escape(l)}:</strong> {html.escape(t)}</div>'
            for l, t in items
        ]
        items_html = '<div class="explain-list">' + "".join(parts) + "</div>"
    st.markdown(
        (
            f'<div class="explain-card explain-{html.escape(kind)}">'
            f'<div class="explain-title">{html.escape(title)}</div>'
            f'<div class="explain-body">{html.escape(patient_friendly_text(body))}</div>'
            f"{items_html}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_color_legend() -> None:
    st.markdown(
        """
        <div class="explain-card">
          <div class="explain-title">Visual Explanation — Color Guide</div>
          <div class="explain-body">Colors indicate the degree of influence each brain region had on the result. They do not confirm the presence of disease.</div>
          <div class="legend-wrap">
            <div class="legend-item"><span class="legend-dot l-low"></span><span><strong>Blue / dark:</strong>&nbsp; lower influence on result</span></div>
            <div class="legend-item"><span class="legend-dot l-mid"></span><span><strong>Green / yellow:</strong>&nbsp; moderate influence</span></div>
            <div class="legend-item"><span class="legend-dot l-high"></span><span><strong>Yellow / red:</strong>&nbsp; stronger influence — not confirmed disease</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def floating_chat_button() -> None:
    # Replaced by inline Clinical Insight Panels — no floating chat icon
    pass


def render_gradcam_grid(paths: list, orientation: str | None = None) -> None:
    paths = valid_gradcam_items(paths)
    if not paths:
        note("No visual explanation images are available yet.", "info")
        return

    if len(paths) == 1:
        item = paths[0]
        image_path = gradcam_item_path(item)
        if not image_path:
            return
        try:
            st.image(
                image_path.read_bytes(),
                caption=gradcam_item_caption(item, 1, orientation),
                use_container_width=True,
            )
        except Exception:
            note("This visual explanation image is not available.", "info")
        return

    for row_start in range(0, len(paths), 4):
        columns = st.columns(4)
        for column_index, column in enumerate(columns):
            image_index = row_start + column_index
            if image_index >= len(paths):
                continue
            item = paths[image_index]
            image_path = gradcam_item_path(item)
            if not image_path:
                continue
            with column:
                try:
                    st.image(
                        image_path.read_bytes(),
                        caption=gradcam_item_caption(item, image_index + 1, orientation),
                        use_container_width=True,
                    )
                except Exception:
                    note("This image slice is not available.", "info")


def render_mri_preview(mri_path: str | Path) -> None:
    explanation_card(
        "How to read this brain scan preview",
        (
            "This preview lets you scroll through the brain scan slice by slice. "
            "It is only a visual preview and does not affect the analysis result."
        ),
        [
            ("Axial", "view from top to bottom"),
            ("Coronal", "view from front to back"),
            ("Sagittal", "view from left to right"),
        ],
    )

    try:
        volume = load_mri_volume_cached(str(mri_path))
    except Exception:
        note("A preview is not available for this file, but the analysis can still be performed.", "info")
        return

    if volume is None or volume.ndim != 3 or int(min(volume.shape)) < 2:
        note("A preview is not available for this file, but the analysis can still be performed.", "info")
        return

    with st.container(border=True):
        view = st.radio(
            "View orientation",
            ["Axial", "Coronal", "Sagittal"],
            horizontal=True,
            key="mri_view",
        )
        axis = {"Axial": 2, "Coronal": 1, "Sagittal": 0}[view]
        max_index = int(volume.shape[axis]) - 1
        index = st.slider(
            "Slice position",
            min_value=0,
            max_value=max_index,
            value=max_index // 2,
            key=f"mri_slice_{view}",
        )
        try:
            image = normalize_slice(take_slice(volume, axis, index))
            left, middle, right = st.columns([1, 2, 1])
            with middle:
                st.image(
                    image,
                    caption=f"{view} view - slice {index + 1} of {max_index + 1}",
                    use_container_width=True,
                    clamp=True,
                )
        except Exception:
            note("This slice could not be displayed, but the analysis can still be performed.", "info")

        with st.expander("Optional 3D volume preview"):
            st.caption("Optional and may take a few seconds to build. The slice viewer above is faster.")
            if st.button("Generate 3D Preview", key="generate_3d_preview"):
                try:
                    with st.spinner("Building 3D preview..."):
                        figure = build_3d_preview(volume)
                    st.plotly_chart(figure, use_container_width=True)
                except Exception:
                    note("3D preview is not available for this file, but the analysis can still be performed.", "info")


def status_pill(active: bool, active_label: str, idle_label: str) -> str:
    if active:
        return (
            f'<span class="sb-pill sb-active">'
            f'<span class="dot"></span>{html.escape(active_label)}</span>'
        )
    return (
        f'<span class="sb-pill sb-idle">'
        f'<span class="dot"></span>{html.escape(idle_label)}</span>'
    )


def timeline_card(date: str, analysis_type: str, result: str, confidence: str) -> None:
    kind_lbl = html.escape(friendly_analysis_type(analysis_type))
    st.markdown(
        f"""
        <div class="tl-row fu">
          <div class="tl-line"><span class="tl-node"></span></div>
          <div class="tl-body">
            <div class="tl-kind">{kind_lbl}</div>
            <div class="tl-main">{html.escape(str(result))} &middot; {html.escape(str(confidence))}</div>
            <div class="tl-date">{html.escape(str(date))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 7) INTEGRATED SUMMARY  — NEW function (add it; reuses unchanged logic)

def report_card(report_info: dict) -> None:
    kind = report_info.get("kind", "")
    if kind == "Brain Scan":
        kind_lbl = "Brain Imaging Analysis"
    elif kind == "Speech and Language":
        kind_lbl = "Speech & Language Assessment"
    elif "Combined" in kind or "Multimodal" in kind:
        kind_lbl = "Integrated Patient Summary"
    else:
        kind_lbl = kind or "Clinical Report"
    st.markdown(
        f"""
        <div class="doc-card fu">
          <div class="doc-kind">{html.escape(kind_lbl)}</div>
          <div class="doc-title">{html.escape(report_info.get("title", "Clinical Report"))}</div>
          <div class="doc-date">{html.escape(report_info.get("date", ""))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 6) TIMELINE CARD  — replaces timeline_card()  (no emoji)

def chat_panel(title: str, context_type: str) -> None:
    """
    Premium Clinical Insight Panel — no chat bubbles, structured clinical cards.
    context_type is "brain" or "speech".
    """
    if context_type == "brain":
        messages_key = "brain_messages"
        panel_title  = "Clinical Interpretation Panel"
        panel_sub    = "Select or enter a clinical review question about this brain imaging result. Responses are decision-support outputs only and are not a medical diagnosis."
        placeholder  = "e.g. What are the recommended next clinical steps for this result?"
        suggestions  = [
            "What does this imaging profile mean clinically?",
            "Do highlighted areas confirm Alzheimer's disease?",
            "How reliable is this confidence score?",
            "What are the recommended next clinical steps?",
        ]
    else:
        messages_key = "speech_messages"
        panel_title  = "Language Assessment Insight"
        panel_sub    = "Submit a clinical question about this language assessment. Answers are for decision-support review only and are not a medical diagnosis."
        placeholder  = "e.g. What do these speech patterns indicate clinically?"
        suggestions  = [
            "What does this language assessment profile mean?",
            "What are the key speech markers observed?",
            "Is this assessment a definitive medical diagnosis?",
            "How should these findings guide clinical review?",
        ]

    # ── Premium panel header ──────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="cip-header fu">
          <div class="cip-left">
            <div class="cip-eyebrow">Clinical Decision Support</div>
            <div class="cip-title">{html.escape(panel_title)}</div>
            <div class="cip-sub">{html.escape(panel_sub)}</div>
          </div>
          <div class="cip-status">
            <span class="cip-status-dot"></span>Active
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Suggested question chips ──────────────────────────────────────────────
    st.markdown('<div class="cip-sugg-hd">Suggested clinical questions</div>', unsafe_allow_html=True)
    sugg_cols = st.columns(2)
    clicked_suggestion: str | None = None
    for sugg_i, sugg_text in enumerate(suggestions):
        if sugg_cols[sugg_i % 2].button(
            sugg_text,
            key=f"cip_s_{context_type}_{sugg_i}",
            use_container_width=True,
        ):
            clicked_suggestion = sugg_text

    # ── Custom question input ─────────────────────────────────────────────────
    st.markdown(
        '<div class="cip-sugg-hd" style="margin-top:14px;">Or type your own clinical question</div>',
        unsafe_allow_html=True,
    )
    user_input = st.text_input(
        "Clinical question",
        placeholder=placeholder,
        key=f"cip_input_{context_type}",
        label_visibility="collapsed",
    )
    ask_col, clear_col = st.columns([4, 1])
    ask_clicked = ask_col.button(
        "Get Clinical Insight",
        key=f"cip_ask_{context_type}",
        type="primary",
        use_container_width=True,
    )
    if clear_col.button("Clear", key=f"cip_clear_{context_type}", use_container_width=True):
        st.session_state[messages_key] = []
        rerun_app()

    # ── Determine which question to process ───────────────────────────────────
    question_to_process: str | None = None
    if clicked_suggestion:
        question_to_process = clicked_suggestion
    elif ask_clicked and user_input.strip():
        question_to_process = user_input.strip()

    if question_to_process:
        st.session_state[messages_key].append({"role": "user", "content": question_to_process})
        answer_sources = []
        with st.spinner("Preparing clinical insight..."):
            try:
                if context_type == "brain":
                    result = st.session_state.get("cnn_result") or {}
                    rag = run_mri_rag_explanation(
                        result,
                        gradcam_info=current_gradcam_info(),
                        question=question_to_process,
                    )
                    answer, answer_sources = build_brain_panel_answer(
                        question_to_process,
                        result,
                        rag,
                    )
                else:
                    answer = build_chat_answer(
                        question=question_to_process,
                        transcript=st.session_state.get("nlp_transcript", ""),
                        prediction=(st.session_state.get("nlp_result") or {}).get("prediction", ""),
                        confidence=(st.session_state.get("nlp_result") or {}).get("confidence", 0.0),
                    )
            except Exception:
                answer = (
                    "This answer could not be generated right now. Please try a "
                    "different question or review the result with a healthcare professional."
                )
        assistant_message = {"role": "assistant", "content": answer}
        if context_type == "brain":
            assistant_message["sources"] = answer_sources
        st.session_state[messages_key].append(assistant_message)

    # ── Render Q&A as structured clinical cards (no chat bubbles) ─────────────
    messages = st.session_state.get(messages_key, [])
    qa_pairs: list[tuple[str, str | None, list]] = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            q = messages[i]["content"]
            assistant_message = (
                messages[i + 1]
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant"
                else {}
            )
            a = assistant_message.get("content") if assistant_message else None
            sources = assistant_message.get("sources", []) if assistant_message else []
            qa_pairs.append((q, a, sources))
            i += 2
        else:
            i += 1

    for q, a, sources in reversed(qa_pairs):
        with st.container(border=True):
            st.markdown(
                f'<div class="cip-q-line">'
                f'<span class="cip-q-badge">Q</span>'
                f'<span class="cip-q-text">{html.escape(q)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if a:
                st.markdown(patient_friendly_text(a))
                if context_type == "brain":
                    st.markdown("**Reference sources**")
                    if has_retrieved_sources(sources):
                        st.markdown(mri_sources_markdown(sources))
                    else:
                        st.info(
                            "This response was generated from the current assessment result. "
                            "No external reference document was retrieved for this question."
                        )


def render_sidebar() -> None:
    """Premium clinical navigation rail with Manrope brand mark."""
    current_page = st.session_state.get("page", "Home")
    case_id = ensure_patient_case_id()
    now_str = datetime.now().strftime("%H:%M")

    with st.sidebar:
        # ── Brand mark ─────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="sb-brand">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
                <div style="width:36px;height:36px;border-radius:10px;
                            background:linear-gradient(135deg,var(--green),var(--green-deep));
                            display:flex;align-items:center;justify-content:center;
                            color:#fff;font-family:'Manrope',sans-serif;font-weight:900;
                            font-size:1.05rem;letter-spacing:-0.02em;flex-shrink:0;
                            box-shadow:0 3px 10px rgba(15,143,117,0.3);">N</div>
                <div>
                  <div class="sb-brand-name">NeuroMind</div>
                  <div class="sb-brand-tag">Clinical Decision Support</div>
                </div>
              </div>
              <div style="display:flex;align-items:center;gap:6px;margin-top:10px;
                          padding-top:10px;border-top:1px solid var(--line);">
                <span style="width:6px;height:6px;border-radius:50%;background:var(--green);
                             animation:nmBlink 2.2s ease-in-out infinite;flex-shrink:0;"></span>
                <span style="font-size:0.65rem;color:var(--muted);font-weight:500;">
                  Session active &middot; {html.escape(now_str)}
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Patient case card ───────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="sb-case">
              <div class="sb-case-lbl">Active Patient Case</div>
              <div class="sb-case-id">{html.escape(case_id)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("New Patient Case", key="start_new_patient_case", use_container_width=True):
            reset_current_patient_case()
            rerun_app()

        # ── Navigation ─────────────────────────────────────────────────────
        st.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)
        nav_items = [
            ("Dashboard",          "Home"),
            ("Brain Imaging",      "Brain Scan"),
            ("Speech & Language",  "Speech & Language"),
            ("Integrated Summary", "Integrated Summary"),
            ("Patient Reports",    "My Reports"),
            ("About",              "About"),
        ]
        for label, page_key in nav_items:
            is_active = current_page == page_key
            if st.button(
                label,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = page_key
                rerun_app()

        st.markdown(
            f'<div class="sb-note">{html.escape(MEDICAL_NOTE)}</div>',
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 3) DASHBOARD  — replaces render_home()

def _ring_svg(pct: float, center: str, sub: str) -> str:
    """Animated radial completeness ring (white + green, no icons)."""
    try:
        pct = max(0.0, min(100.0, float(pct)))
    except (TypeError, ValueError):
        pct = 0.0
    circumference = 339.292  # 2*pi*54
    offset = circumference * (1.0 - pct / 100.0)
    return (
        '<svg width="128" height="128" viewBox="0 0 132 132" role="img">'
        '<circle cx="66" cy="66" r="54" fill="none" stroke="#DCEFE6" stroke-width="11"/>'
        '<circle class="nm-ring-arc" cx="66" cy="66" r="54" fill="none" stroke="#0F8F75" '
        'stroke-width="11" stroke-linecap="round" transform="rotate(-90 66 66)" '
        f'stroke-dasharray="{circumference:.2f}" style="stroke-dashoffset:{offset:.2f}"/>'
        f'<text x="66" y="62" text-anchor="middle" class="nm-ring-num">{html.escape(center)}</text>'
        f'<text x="66" y="82" text-anchor="middle" class="nm-ring-sub">{html.escape(sub)}</text>'
        '</svg>'
    )


def _gauge_svg(pct: float, tone: str = "good") -> str:
    """Animated semicircular confidence gauge (white + green, no icons)."""
    try:
        pct = max(0.0, min(100.0, float(pct)))
    except (TypeError, ValueError):
        pct = 0.0
    length = 257.6  # pi*82  (semicircle)
    offset = length * (1.0 - pct / 100.0)
    color = "#064E3B" if tone == "alert" else "#0F8F75"
    return (
        '<svg width="172" height="100" viewBox="0 0 200 112" role="img">'
        '<path d="M18,100 A82,82 0 0 1 182,100" fill="none" stroke="#DCEFE6" '
        'stroke-width="12" stroke-linecap="round"/>'
        '<path class="nm-gauge-arc" d="M18,100 A82,82 0 0 1 182,100" fill="none" '
        f'stroke="{color}" stroke-width="12" stroke-linecap="round" '
        f'stroke-dasharray="{length:.1f}" style="stroke-dashoffset:{offset:.1f}"/>'
        f'<text x="100" y="92" text-anchor="middle" class="nm-gauge-num">{pct:.1f}%</text>'
        '</svg>'
    )


def render_home() -> None:
    """Clinical command center dashboard (white + green, animated, Manrope)."""
    case_id        = ensure_patient_case_id()
    cnn_active     = st.session_state.get("cnn_result") is not None
    nlp_active     = st.session_state.get("nlp_result") is not None
    report_count   = len(generated_reports())
    reports_active = report_count > 0
    summary_ready  = cnn_active and nlp_active

    done_count   = sum(bool(x) for x in (cnn_active, nlp_active, summary_ready))
    completeness = int(round(done_count / 3.0 * 100))
    now_str      = datetime.now().strftime("%A, %B %d  ·  %H:%M")

    # ── Page header ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="page-head fu">
          <div class="page-eyebrow">Clinical Command Center</div>
          <div class="page-title">Patient Dashboard</div>
          <div class="page-sub">
            Independent brain imaging and language assessments — decision-support only,
            not a medical diagnosis.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Command header: brand · live EEG motif · session ──────────────
    st.markdown(
        f"""
        <div class="cmd-head">
          <div class="cmd-left">
            <div class="cmd-mark">N</div>
            <div>
              <div class="cmd-name">NeuroMind</div>
              <div class="cmd-tag">Alzheimer clinical decision support</div>
            </div>
          </div>
          <div class="cmd-right">
            <svg width="132" height="34" viewBox="0 0 132 34" role="img">
              <polyline class="nm-eeg-line" points="0,18 14,18 20,18 26,6 32,30 38,18 56,18 64,18 70,9 76,27 82,18 100,18 108,18 114,13 120,22 132,18"/>
            </svg>
            <span class="cmd-live"><span class="dot"></span>{html.escape(now_str)}</span>
          </div>
        </div>
        <div class="cmd-caseline">
          <span class="lbl">Active patient case</span>
          <span class="id">{html.escape(case_id)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)

    # ── Completeness ring + case snapshot ─────────────────────────────
    def _ring_li(is_done: bool, label: str) -> str:
        cls = "done" if is_done else "idle"
        return f'<div class="ring-li {cls}"><span class="pip"></span>{label}</div>'

    col_ring, col_snap = st.columns([1, 1], gap="medium")
    with col_ring:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="ring-card fu">
                  {_ring_svg(completeness, f"{completeness}%", "case ready")}
                  <div class="ring-list">
                    {_ring_li(cnn_active, "Brain imaging")}
                    {_ring_li(nlp_active, "Speech &amp; language")}
                    {_ring_li(summary_ready, "Integrated summary")}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with col_snap:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="fu">
                  <div class="snap-grid">
                    <div class="snap"><div class="snap-lbl">Modules complete</div><div class="snap-val">{done_count} / 3</div></div>
                    <div class="snap"><div class="snap-lbl">Reports ready</div><div class="snap-val">{report_count}</div></div>
                    <div class="snap"><div class="snap-lbl">Brain imaging</div><div class="snap-val">{"Done" if cnn_active else "&mdash;"}</div></div>
                    <div class="snap"><div class="snap-lbl">Speech</div><div class="snap-val">{"Done" if nlp_active else "&mdash;"}</div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)

    # ── Module tiles (animated accent, no icons) ──────────────────────
    modules = [
        ("Brain Imaging Analysis", "Structural MRI review with confidence score and visual explanation.",
         "Complete" if cnn_active else "Not started", cnn_active, "home_brain", "Brain Scan"),
        ("Speech & Language Assessment", "Transcript-based language markers and assessment profile.",
         "Complete" if nlp_active else "Not started", nlp_active, "home_speech", "Speech & Language"),
        ("Integrated Patient Summary", "Side-by-side review of imaging and language outputs.",
         "Available" if (cnn_active or nlp_active) else "Awaiting analyses",
         (cnn_active or nlp_active), "home_combined", "Integrated Summary"),
        ("Patient Reports", "Assessment history and downloadable clinical documents.",
         f"{report_count} ready" if reports_active else "None yet", reports_active, "home_reports", "My Reports"),
    ]
    for row in (modules[:2], modules[2:]):
        cols = st.columns(2, gap="medium")
        for col, (title, desc, status, is_done, key, page_key) in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="nm-tile fu">
                          <div class="nm-tile-acc"></div>
                          <div class="nm-tile-title">{html.escape(title)}</div>
                          <div class="nm-tile-desc">{html.escape(desc)}</div>
                          <div class="nm-tile-status {'is-done' if is_done else 'is-idle'}">{html.escape(status)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button("Open", key=key, use_container_width=True):
                        st.session_state.page = page_key
                        rerun_app()

    # ── Connected clinical pathway ────────────────────────────────────
    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)
    steps = [
        ("Patient case", True),
        ("Brain imaging", cnn_active),
        ("Speech & language", nlp_active),
        ("Integrated summary", summary_ready),
        ("Reports", reports_active),
    ]
    parts = []
    for i, (label, is_done) in enumerate(steps):
        if i > 0:
            seg_on = steps[i - 1][1]
            parts.append(f'<div class="pw-seg {"on" if seg_on else ""}"></div>')
        if is_done:
            state = "done"
        elif i == 0 or steps[i - 1][1]:
            state = "cur"
        else:
            state = "idle"
        lbl_cls = "on" if state in ("done", "cur") else ""
        parts.append(
            f'<div class="pw-step">'
            f'<div class="pw-dot {state}">{i + 1}</div>'
            f'<div class="pw-lbl {lbl_cls}">{html.escape(label)}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="path-strip fu"><div class="path-title">Clinical pathway</div>'
        f'<div class="pw-row">{"".join(parts)}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)
    medical_note_card(
        "This platform provides clinical decision-support outputs only. "
        "Results must be reviewed by a qualified healthcare professional. "
        "This tool does not provide a medical diagnosis."
    )


# -----------------------------------------------------------------------------
# 4) PATIENT CASE CARD  — replaces patient_case_card()

def render_brain_scan() -> None:
    st.markdown(
        """
        <div class="page-head fu">
          <div class="page-eyebrow">Clinical Assessment Module</div>
          <div class="page-title">Brain Imaging Analysis</div>
          <div class="page-sub">
            Upload a structural MRI brain scan to receive an imaging profile
            with confidence score and visual explanation — decision-support output only.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    medical_note_card(MEDICAL_NOTE)

    step_card(1, "Step 1 — Upload Brain Imaging Scan", "Select an anonymized MRI scan file to begin the assessment.")
    with st.container(border=True):
        st.write("Supported formats: `.nii` and `.nii.gz` brain scans.")
        uploaded_mri = st.file_uploader(
            "Brain scan file",
            type=["nii", "gz"],
            key="mri_upload",
            label_visibility="collapsed",
        )
        if uploaded_mri is not None:
            if st.session_state.get("mri_source_name") != uploaded_mri.name:
                st.session_state.mri_path = None
                st.session_state.mri_original_path = None
                st.session_state.mri_processed_path = None
                st.session_state.mri_source_name = uploaded_mri.name
                st.session_state.cnn_result = None
                st.session_state.gradcam_paths = []
                st.session_state.gradcam_generated_orientation = None
                st.session_state.gradcam_generated_display_mode = None
                st.session_state.cnn_report_path = None
                st.session_state.cnn_report_html_path = None
                st.session_state.cnn_report_pdf_path = None
                st.session_state.cnn_analysis_id = ""
                st.session_state.cnn_preview_path = None
                st.session_state.hd_bet_applied = False
                st.session_state.mri_preprocessing_note = ""
                st.session_state.confirm_skull_stripped_mri = False
                st.session_state.show_mri_preview = False
                st.session_state.mri_rag_explanation = ""
                st.session_state.mri_rag_sources = []
                st.session_state.brain_messages = []
            note("Scan selected successfully. Continue to Step 2 to analyze it.", "success")

        st.checkbox(
            "Prepare scan automatically before analysis",
            key="apply_hd_bet_cpu",
            help="Applies automated scan preparation before analysis. This may take a few minutes.",
        )
        if st.session_state.get("apply_hd_bet_cpu"):
            note(
                "The scan will be prepared automatically before analysis. "
                "This may take a few minutes.",
                "info",
            )
        else:
            st.checkbox(
                "I confirm that this scan has already been prepared for analysis.",
                key="confirm_skull_stripped_mri",
            )
            if uploaded_mri is not None and not st.session_state.get("confirm_skull_stripped_mri"):
                note(
                    "Please confirm that this scan has already been prepared for analysis, "
                    "or enable automatic scan preparation above.",
                    "warning",
                )

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    step_card(2, "Step 2 — Analyze Brain Imaging", "Run the brain imaging assessment on the selected scan.")
    with st.container(border=True):
        can_analyze_mri = bool(
            uploaded_mri is not None
            and (
                st.session_state.get("apply_hd_bet_cpu")
                or st.session_state.get("confirm_skull_stripped_mri")
            )
        )
        analyze_clicked = st.button(
            "Analyze Brain Imaging",
            disabled=not can_analyze_mri,
            key="analyze_mri_button",
            use_container_width=True,
        )

    if analyze_clicked and uploaded_mri is not None:
        try:
            patient_id = ensure_patient_case_id()
            analysis_id = generate_analysis_id()
            st.session_state.cnn_analysis_id = analysis_id
            saved_path = save_uploaded_mri(uploaded_mri)
            st.session_state.mri_original_path = str(saved_path)
            st.session_state.mri_source_name = uploaded_mri.name
            analysis_mri_path = saved_path

            if st.session_state.get("apply_hd_bet_cpu"):
                try:
                    with st.spinner("Preparing scan for analysis..."):
                        analysis_mri_path = Path(
                            run_hd_bet_cpu(
                                str(saved_path),
                                str(hd_bet_output_path(analysis_id)),
                            )
                        )
                    st.session_state.hd_bet_applied = True
                    st.session_state.mri_preprocessing_note = (
                        "The scan was automatically prepared before analysis."
                    )
                except Exception as exc:
                    st.session_state.hd_bet_applied = False
                    st.session_state.mri_preprocessing_note = ""
                    show_error(
                        "Scan preparation could not be completed. Please upload a pre-processed brain MRI scan or try again.",
                        exc,
                    )
                    return
            else:
                st.session_state.hd_bet_applied = False
                st.session_state.mri_preprocessing_note = (
                    "The uploaded scan was used as already prepared for analysis."
                )

            st.session_state.mri_path = str(analysis_mri_path)
            st.session_state.mri_processed_path = str(analysis_mri_path)

            with st.spinner("Analyzing the brain scan..."):
                cnn_result = run_cnn_prediction(analysis_mri_path)

            st.session_state.cnn_result = cnn_result
            st.session_state.gradcam_paths = []
            st.session_state.gradcam_generated_orientation = None
            st.session_state.gradcam_generated_display_mode = None
            st.session_state.show_mri_preview = False
            st.session_state.mri_rag_explanation = ""
            st.session_state.mri_rag_sources = []
            st.session_state.brain_messages = []
            st.session_state.cnn_preview_path = save_mri_preview_image(analysis_mri_path, patient_id)

            report_paths = {}
            try:
                report_paths = save_brain_scan_reports(
                    result=cnn_result,
                    patient_case_id=patient_id,
                    analysis_id=analysis_id,
                    mri_preview_path=st.session_state.cnn_preview_path,
                    mri_source_name=uploaded_mri.name,
                    preprocessing_note=st.session_state.get("mri_preprocessing_note"),
                )
                st.session_state.cnn_report_path = report_paths["md"]
                st.session_state.cnn_report_html_path = report_paths["html"]
                st.session_state.cnn_report_pdf_path = report_paths["pdf"]
            except Exception:
                st.session_state.cnn_report_path = None
                st.session_state.cnn_report_html_path = None
                st.session_state.cnn_report_pdf_path = None
                note(
                    "Brain scan analysis completed, but the report could not be generated.",
                    "warning",
                )

            try:
                append_history(
                    analysis_type="Brain Scan Analysis",
                    patient_case_id=patient_id,
                    analysis_id=analysis_id,
                    result=cnn_result["prediction"],
                    confidence=cnn_result["confidence"],
                    report_path=report_paths.get("html"),
                    notes="Brain scan report generated successfully.",
                )
            except Exception:
                pass

            try:
                db_saved = save_brain_result_to_database(
                    patient_case_id=patient_id,
                    analysis_id=analysis_id,
                    result=cnn_result,
                    uploaded_file_name=uploaded_mri.name,
                    mri_file_path=analysis_mri_path,
                    report_paths=report_paths,
                )
                warn_database_not_saved(db_saved)
            except Exception:
                note("Patient history is currently unavailable. Results were generated but not saved.", "warning")
            note("Brain scan analysis completed.", "success")
        except Exception as exc:
            log_brain_analysis_error(exc)
            show_error("The analysis could not be completed. Please check the input and try again.", exc)

    if st.session_state.get("cnn_result"):
        result = st.session_state.cnn_result
        prediction = result["prediction"]

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        section("Brain Imaging Profile")

        if prediction == "CN":
            result_card("Brain Imaging Profile", "Cognitively Normal Profile",
                        percent(result["confidence"]),
                        "The imaging assessment did not detect a brain scan pattern strongly "
                        "associated with Alzheimer's disease in this scan. This is a "
                        "decision-support output and must be reviewed by a healthcare professional.",
                        "good", "CN")
        elif prediction == "AD":
            result_card("Brain Imaging Profile", "Alzheimer's Disease-Compatible Profile",
                        percent(result["confidence"]),
                        "The imaging assessment detected a brain scan pattern that may be "
                        "associated with Alzheimer's disease. This is not a diagnosis and "
                        "must be reviewed by a qualified specialist.", "alert", "AD")
        else:
            result_card("Brain Imaging Profile", str(prediction),
                        percent(result["confidence"]),
                        "This result is a clinical decision-support output and must be reviewed "
                        "by a qualified healthcare professional.", "good", "—")

        with st.container(border=True):
            metric_cols = st.columns(3)
            metric_cols[0].metric("Confidence Score",                        percent(result["confidence"]))
            metric_cols[1].metric("Cognitively Normal Profile Score",        percent(result["prob_cn"]))
            metric_cols[2].metric("Alzheimer's Disease-Compatible Score",    percent(result["prob_ad"]))

        if st.session_state.get("mri_preprocessing_note"):
            note(st.session_state.mri_preprocessing_note, "info")

        medical_note_card(MEDICAL_NOTE)

        if st.session_state.get("mri_path"):
            with st.expander("Show brain scan preview"):
                try:
                    render_mri_preview(st.session_state.mri_path)
                except Exception:
                    note("A preview is not available for this file, but the analysis is complete.", "info")

        st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
        section("Visual Explanation",
                "The colored areas show which parts of the image influenced the result. "
                "They do not mean that these areas are diseased.")

        with st.container(border=True):
            explanation_card(
                "How to read this visual explanation",
                (
                    "The colored areas show which parts of the brain scan influenced "
                    "the result. They do not mean that these areas are diseased."
                ),
                [
                    ("Axial view", "top to bottom"),
                    ("Sagittal view", "left to right"),
                    ("Coronal view", "front to back"),
                ],
            )
            render_color_legend()

            selected_display_mode = st.radio(
                "Visual display mode",
                ["Overlay view", "Heatmap only"],
                horizontal=True,
                key="gradcam_display_mode",
            )
            backend_display_mode = (
                "heatmap" if selected_display_mode == "Heatmap only" else "overlay"
            )
            if (
                st.session_state.get("gradcam_generated_display_mode")
                and st.session_state.gradcam_generated_display_mode != selected_display_mode
            ):
                st.session_state.gradcam_paths = []
                st.session_state.gradcam_generated_orientation = None
                st.session_state.gradcam_generated_display_mode = None

            if st.button("Show Visual Explanation", key="generate_gradcam_button"):
                st.session_state.gradcam_paths = []
                st.session_state.gradcam_generated_orientation = None
                st.session_state.gradcam_generated_display_mode = None
                try:
                    with st.spinner("Generating visual explanation..."):
                        gradcam_paths = run_gradcam_generation(
                            result["mri_path"],
                            orientation="multi",
                            num_slices=5,
                            display_mode=backend_display_mode,
                        )
                    st.session_state.gradcam_paths = gradcam_paths
                    st.session_state.gradcam_generated_orientation = (
                        "Axial, sagittal and coronal views"
                    )
                    st.session_state.gradcam_generated_display_mode = selected_display_mode
                    write_updated_cnn_report(result, ensure_patient_id())
                    note("Visual explanation generated successfully.", "success")
                except Exception as exc:
                    st.session_state.gradcam_paths = []
                    st.session_state.gradcam_generated_orientation = None
                    st.session_state.gradcam_generated_display_mode = None
                    show_error("Visual explanation could not be generated for this file.", exc)

            st.session_state.gradcam_paths = valid_gradcam_items(
                st.session_state.get("gradcam_paths") or []
            )
            if st.session_state.gradcam_paths:
                render_gradcam_grid(
                    st.session_state.gradcam_paths,
                    st.session_state.get("gradcam_generated_orientation"),
                )

        st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
        section("Clinical Interpretation")
        with st.container(border=True):
            explanation_card("What this result means", brain_scan_simple_explanation(prediction))
            if st.button("Generate clinical interpretation", key="generate_mri_rag_explanation"):
                try:
                    with st.spinner("Preparing clinical interpretation..."):
                        rag_result = run_mri_rag_explanation(result, gradcam_info=current_gradcam_info())
                    raw = str(rag_result.get("answer", "")).strip()
                    if not raw or raw == "MRI explanation is currently unavailable.":
                        note("A clinical interpretation is currently unavailable.", "warning")
                    else:
                        st.session_state.mri_rag_explanation = patient_friendly_text(
                            strip_chatbot_preamble(raw)
                        )
                        st.session_state.mri_rag_sources = rag_result.get("sources", [])
                        write_updated_cnn_report(result, ensure_patient_id())
                        note("Clinical interpretation generated.", "success")
                except Exception as exc:
                    show_error("A clinical interpretation is currently unavailable.", exc)

            if st.session_state.get("mri_rag_explanation"):
                with st.container(border=True):
                    st.markdown("##### Clinical interpretation")
                    st.write(st.session_state.mri_rag_explanation)
                render_mri_sources(st.session_state.get("mri_rag_sources", []))

        if st.session_state.get("cnn_report_path"):
            st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
            section("Brain Imaging Analysis Report")
            render_report_downloads(Path(st.session_state.cnn_report_path))

        st.markdown("<div style='height: 22px'></div>", unsafe_allow_html=True)
        chat_panel("Clinical Imaging Insight", "brain")


def render_report_downloads(report_path: Path, key_prefix: str = "main") -> None:
    """Show Download HTML / PDF / Markdown buttons for one report, no file paths shown."""
    formats = report_sibling_files(report_path)
    pdf_error_key = f"pdf_error_{key_prefix}_{report_path.stem}"
    if not formats:
        note("No downloadable report is available yet.", "info")
        return

    with st.container(border=True):
        cols = st.columns(3)
        order = [("html", "Download HTML", "text/html"),
                 ("pdf", "Download PDF", "application/pdf"),
                 ("md", "Download Markdown", "text/markdown")]
        for col, (ext, label, mime) in zip(cols, order):
            path = formats.get(ext)
            needs_current_brain_visual_pdf = (
                ext == "pdf"
                and can_generate_current_brain_visual_pdf(report_path)
                and not report_has_valid_visual_images(report_path)
            )
            if needs_current_brain_visual_pdf:
                if col.button(
                    "Generate PDF + Images",
                    key=f"generate_visual_pdf_{key_prefix}_{report_path.stem}",
                    use_container_width=True,
                ):
                    try:
                        pdf_path = generate_current_brain_report_with_visual_images()
                        if not pdf_path or not Path(pdf_path).exists():
                            raise RuntimeError("PDF file was not created.")
                        st.session_state[pdf_error_key] = ""
                        warn_database_not_saved(save_existing_report_to_database(pdf_path))
                        note("PDF report with visual explanation generated successfully.", "success")
                        rerun_app()
                    except ModuleNotFoundError:
                        st.session_state[pdf_error_key] = (
                            "PDF generation requires ReportLab and Pillow. "
                            "Please install the project requirements, then try again."
                        )
                    except Exception:
                        st.session_state[pdf_error_key] = (
                            "PDF with visual explanation could not be generated. "
                            "Please try Show Visual Explanation first, then generate the PDF again."
                        )
            elif path and Path(path).exists():
                with Path(path).open("rb") as handle:
                    col.download_button(
                        label,
                        data=handle.read(),
                        file_name=report_download_filename(Path(path)),
                        mime=mime,
                        key=f"dl_{key_prefix}_{ext}_{Path(path).name}",
                        use_container_width=True,
                    )
            elif ext == "pdf":
                if col.button(
                    "Generate PDF",
                    key=f"generate_{key_prefix}_{ext}_{report_path.stem}",
                    use_container_width=True,
                ):
                    try:
                        md_path = report_path.with_suffix(".md")
                        if not md_path.exists():
                            raise FileNotFoundError("The report source file is missing.")
                        pdf_path = save_patient_report_pdf(
                            md_path.read_text(encoding="utf-8"),
                            md_path,
                        )
                        st.session_state[pdf_error_key] = ""
                        warn_database_not_saved(save_existing_report_to_database(pdf_path))
                        note("PDF report generated successfully. The download button is ready.", "success")
                        rerun_app()
                    except ModuleNotFoundError:
                        st.session_state[pdf_error_key] = (
                            "PDF generation requires ReportLab and Pillow. "
                            "Please install the project requirements, then try again."
                        )
                    except Exception:
                        st.session_state[pdf_error_key] = (
                            "PDF report could not be generated. Please regenerate the report "
                            "and try again."
                        )
            else:
                col.button(
                    label,
                    key=f"dl_{key_prefix}_{ext}_missing",
                    use_container_width=True,
                    disabled=True,
                )

    if st.session_state.get(pdf_error_key):
        note(st.session_state[pdf_error_key], "error")


def render_speech_legacy_manual() -> None:
    section("Speech and Language Analysis",
            "Enter a speech transcript and speech information to receive a clear explanation.")
    medical_note_card(MEDICAL_NOTE)

    with st.form("speech_analysis_form"):
        step_card(1, "Step 1 — Enter speech transcript", "Type or paste the patient's speech transcript.")
        with st.container(border=True):
            transcript = st.text_area(
                "Transcript",
                height=200,
                placeholder="Enter the patient's English picture-description transcript...",
                label_visibility="collapsed",
            )

        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        step_card(2, "Step 2 — Add speech information", "These details help the speech analysis.")
        with st.container(border=True):
            info_left, info_right = st.columns(2)
            with info_left:
                patient_name = st.text_input("Patient Name")
                study_date = st.date_input("Study Date", value=datetime.now().date())
                clinician = st.text_input("Responsible Clinician")
            with info_right:
                clinical_notes = st.text_area(
                    "Clinical Notes",
                    height=132,
                    placeholder=(
                        "Example: The patient completed a picture description task. "
                        "The speech transcript file was uploaded for automatic language analysis."
                    ),
                )

            st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                n_filled_pauses = st.number_input("Hesitation words", min_value=0, value=0)
                n_phon_fragments = st.number_input("Interrupted words", min_value=0, value=0)
                n_paralinguistic = st.number_input("Non-verbal speech markers", min_value=0, value=0)
            with col2:
                n_retracings = st.number_input("Self-corrections", min_value=0, value=0)
                n_unintelligible = st.number_input("Unclear words", min_value=0, value=0)
                n_pauses = st.number_input("Pauses", min_value=0, value=0)
            with col3:
                entryage = st.number_input("Age", min_value=0, max_value=120, value=70)
                sex = st.selectbox("Sex", options=[0, 1],
                                   format_func=lambda value: "Female" if value == 0 else "Male")
                educ = st.number_input("Years of education", min_value=0, max_value=30, value=12)

        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        step_card(3, "Step 3 — Analyze speech", "Run the speech and language analysis.")
        submitted = st.form_submit_button("Analyze speech", use_container_width=True)

    if submitted:
        if not transcript.strip():
            note("Please enter a patient transcript.", "warning")
        else:
            patient_id = ensure_patient_case_id()
            analysis_id = generate_analysis_id()
            st.session_state.nlp_analysis_id = analysis_id
            feature_values = [n_filled_pauses, n_phon_fragments, n_paralinguistic,
                              n_retracings, n_unintelligible, n_pauses, entryage, sex, educ]
            analysis_features = build_analysis_features(
                n_filled_pauses, n_phon_fragments, n_paralinguistic, n_retracings,
                n_unintelligible, n_pauses, entryage, sex, educ)
            patient_info = build_patient_info(
                patient_name=patient_name, patient_id=patient_id, study_date=study_date,
                age=entryage, sex_label="Female" if sex == 0 else "Male",
                education_years=educ, clinician=clinician, clinical_notes=clinical_notes,
                analysis_id=analysis_id)

            try:
                with st.spinner("Analyzing speech transcript..."):
                    nlp_result = run_nlp_prediction(transcript=transcript, feature_values=feature_values)
                with st.spinner("Preparing explanation..."):
                    llm_explanation = build_nlp_llm_explanation(
                        result=nlp_result, analysis_features=analysis_features)

                report_paths = save_speech_language_reports(
                    result=nlp_result,
                    patient_case_id=patient_id,
                    analysis_id=analysis_id,
                    transcript=transcript,
                    analysis_features=analysis_features,
                    patient_info=patient_info,
                    llm_explanation=llm_explanation,
                )

                st.session_state.nlp_result = nlp_result
                st.session_state.nlp_transcript = transcript
                st.session_state.nlp_patient_info = patient_info
                st.session_state.nlp_analysis_features = analysis_features
                st.session_state.nlp_llm_explanation = llm_explanation
                st.session_state.patient_report_path = report_paths["md"]
                st.session_state.patient_report_html_path = report_paths["html"]
                st.session_state.patient_report_pdf_path = report_paths["pdf"]
                st.session_state.patient_report = report_paths["md"].read_text(encoding="utf-8")
                st.session_state.speech_messages = [{
                    "role": "assistant",
                    "content": "Speech analysis completed. You can ask questions about this result.",
                }]
                append_history(
                    analysis_type="Speech and Language Analysis",
                    patient_case_id=patient_id,
                    analysis_id=analysis_id,
                    result=nlp_result["prediction"],
                    confidence=nlp_result["confidence"],
                    report_path=report_paths["html"],
                    notes="Speech and language report generated successfully.",
                )
                note("Speech analysis completed.", "success")
            except Exception as exc:
                print("Speech prediction error:", repr(exc))
                show_error("Speech analysis could not be completed. Please check the "
                           "transcript and information, then try again.", exc)

    if st.session_state.get("nlp_result"):
        nlp_result = st.session_state.nlp_result
        prediction = nlp_result["prediction"]
        confidence = nlp_result["confidence"]

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        section("Speech Analysis Result")

        if prediction == "ProbableAD":
            result_card("Speech Analysis Result", "ProbableAD", percent(confidence),
                        "The speech analysis detected language patterns that may be associated "
                        "with cognitive decline. This is not a diagnosis.", "alert", "AD")
        elif prediction == "Control":
            result_card("Speech Analysis Result", "Control", percent(confidence),
                        "The speech analysis did not detect enough language patterns associated "
                        "with a probable Alzheimer's disease profile.", "good", "CN")
        else:
            result_card("Speech Analysis Result", prediction_display(prediction),
                        percent(confidence),
                        "This result is a decision-support output and must be reviewed by a "
                        "healthcare professional.", "good", "—")

        with st.container(border=True):
            metric_cols = st.columns(2)
            metric_cols[0].metric("Result", prediction_display(prediction))
            metric_cols[1].metric("Confidence Score", percent(confidence))

        section("Simple Explanation")
        with st.container(border=True):
            explanation_card("What this result means",
                             st.session_state.get("nlp_llm_explanation", "")
                             or required_nlp_explanation(prediction, confidence))
            with st.expander("Speech patterns observed"):
                st.markdown(observed_linguistic_signs(st.session_state.nlp_analysis_features))

        medical_note_card(MEDICAL_NOTE)

        if st.session_state.get("patient_report_path"):
            st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
            section("Speech & Language Assessment Report")
            render_report_downloads(Path(st.session_state.patient_report_path), key_prefix="speech")

        st.markdown("<div style='height: 22px'></div>", unsafe_allow_html=True)
        chat_panel("Language Assessment Insight", "speech")


def render_speech() -> None:
    st.markdown(
        """
        <div class="page-head fu">
          <div class="page-eyebrow">Clinical Assessment Module</div>
          <div class="page-title">Speech &amp; Language Assessment</div>
          <div class="page-sub">
            Upload a CHAT transcript file to receive a language assessment profile
            with confidence score — decision-support output only.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    medical_note_card(MEDICAL_NOTE)

    patient_id = ensure_patient_case_id()

    step_card(1, "Step 1 — Patient Information", "Enter the patient and clinician details for this assessment session.")
    with st.container(border=True):
        info_left, info_right = st.columns(2)
        with info_left:
            patient_name = st.text_input("Patient Name", key="speech_patient_name")
            study_date = st.date_input(
                "Study Date",
                value=datetime.now().date(),
                key="speech_study_date",
            )
            clinician = st.text_input("Responsible Clinician", key="speech_clinician")
        with info_right:
            if (
                "speech_notes" in st.session_state
                and is_prompt_like_clinical_note(st.session_state.speech_notes)
            ):
                st.session_state.speech_notes = ""
            clinical_notes = st.text_area(
                "Clinical Notes",
                height=132,
                key="speech_notes",
                placeholder=(
                    "Example: The patient completed a picture description task. "
                    "The speech transcript file was uploaded for automatic language analysis."
                ),
            )

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    step_card(2, "Step 2 — Upload Speech Transcript", "Upload a CHAT transcript file (.cha) for language assessment.")
    with st.container(border=True):
        uploaded_cha = st.file_uploader(
            "Upload speech transcript file",
            type=["cha"],
            help="Upload a CHAT transcript file.",
            key="speech_cha_upload",
        )

    cha_text = ""
    cleaned_transcript = ""
    features_dict: dict = {}
    uploaded_file_name = ""
    parsing_error = None

    if uploaded_cha is not None:
        uploaded_file_name = Path(uploaded_cha.name).name
        if st.session_state.get("nlp_uploaded_file_name") != uploaded_file_name:
            st.session_state.nlp_uploaded_file_name = uploaded_file_name
            st.session_state.nlp_result = None
            st.session_state.nlp_transcript = ""
            st.session_state.nlp_cleaned_transcript = ""
            st.session_state.nlp_extracted_features = {}
            st.session_state.nlp_feature_vector = []
            st.session_state.nlp_confidence = None
            st.session_state.nlp_analysis_features = {}
            st.session_state.nlp_llm_explanation = ""
            st.session_state.patient_report_path = None
            st.session_state.patient_report_html_path = None
            st.session_state.patient_report_pdf_path = None
            st.session_state.patient_report = None

        try:
            cha_text = cha_parser.read_cha_file(uploaded_cha)
            raw_transcript = cha_parser.extract_participant_transcript(cha_text)
            cleaned_transcript = cha_parser.clean_cha_transcript(raw_transcript)
            features_dict = cha_parser.extract_speech_features_from_cha(cha_text)
            st.session_state.nlp_cleaned_transcript = cleaned_transcript
            st.session_state.nlp_extracted_features = features_dict
        except Exception as exc:
            parsing_error = exc
            show_error(
                "The uploaded speech file could not be read. Please check the file and try again.",
                exc,
            )

    if uploaded_cha is None:
        note("Please upload a .cha speech transcript file to continue.", "info")
    elif parsing_error is None:
        if not cleaned_transcript:
            note("The uploaded file does not contain enough patient speech for analysis.", "warning")
        else:
            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            step_card(
                3,
                "Step 3 — Transcript Preview",
                "Review the automatically extracted speech text before running the assessment.",
            )
            with st.container(border=True):
                preview_cols = st.columns(3)
                preview_cols[0].metric("Uploaded file", uploaded_file_name)
                preview_cols[1].metric("Word count", speech_word_count(cleaned_transcript))
                preview_cols[2].metric("Text status", "Extracted")
                note(
                    "This text was automatically extracted from the uploaded speech transcript file.",
                    "info",
                )
                st.text_area(
                    "Extracted Speech Text",
                    value=cleaned_transcript,
                    height=180,
                    disabled=True,
                )

            missing_keys = [
                key for key in ("entryage", "sex", "educ")
                if features_dict.get(key) is None or features_dict.get(key) == ""
            ]
            if missing_keys:
                st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
                section(
                    "Missing information",
                    "Complete only the patient details that were not found in the speech file.",
                )
                with st.container(border=True):
                    missing_cols = st.columns(len(missing_keys))
                    for col, key in zip(missing_cols, missing_keys):
                        with col:
                            if key == "entryage":
                                value = st.text_input("Age", key="missing_entryage")
                                if value.strip():
                                    parsed = _parse_age_value(value)
                                    if parsed is not None:
                                        features_dict[key] = parsed
                                    else:
                                        note("Please enter a valid age.", "warning")
                            elif key == "sex":
                                value = st.selectbox(
                                    "Sex",
                                    options=["Female", "Male"],
                                    index=None,
                                    placeholder="Select sex",
                                    key="missing_sex",
                                )
                                if value:
                                    features_dict[key] = 0 if value == "Female" else 1
                            elif key == "educ":
                                value = st.text_input("Years of education", key="missing_educ")
                                if value.strip():
                                    parsed = _parse_number(value, max_value=40)
                                    if parsed is not None:
                                        features_dict[key] = parsed
                                    else:
                                        note(
                                            "Please enter a valid number of education years.",
                                            "warning",
                                        )

            st.session_state.nlp_extracted_features = features_dict

            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            step_card(
                4,
                "Step 4 — Extracted Speech Markers",
                "These language markers were automatically extracted from the uploaded transcript.",
            )
            with st.container(border=True):
                for row_start in range(0, len(cha_parser.FEATURE_ORDER), 3):
                    cols = st.columns(3)
                    for col, key in zip(cols, cha_parser.FEATURE_ORDER[row_start:row_start + 3]):
                        col.metric(
                            CHA_FEATURE_LABELS.get(key, key),
                            speech_feature_display_value(key, features_dict.get(key)),
                        )

            required_missing = [
                key for key in ("entryage", "sex", "educ")
                if features_dict.get(key) is None or features_dict.get(key) == ""
            ]
            can_analyze = bool(cleaned_transcript.strip()) and not required_missing

            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            step_card(5, "Step 5 — Run Language Assessment", "Analyze the speech transcript and generate the language assessment profile.")
            if required_missing:
                note("Please complete the missing information before analysis.", "warning")

            submitted = st.button(
                "Analyze speech",
                type="primary",
                use_container_width=True,
                disabled=not can_analyze,
            )

            if submitted:
                analysis_id = generate_analysis_id()
                st.session_state.nlp_analysis_id = analysis_id

                try:
                    feature_vector = build_feature_vector_from_cha(features_dict)
                    st.session_state.nlp_feature_vector = feature_vector
                    analysis_features = build_analysis_features(*feature_vector)
                    patient_info = build_patient_info(
                        patient_name=patient_name,
                        patient_id=patient_id,
                        study_date=study_date,
                        age=feature_vector[6],
                        sex_label="Female" if int(feature_vector[7]) == 0 else "Male",
                        education_years=feature_vector[8],
                        clinician=clinician,
                        clinical_notes=clinical_notes,
                        analysis_id=analysis_id,
                    )
                    patient_info["Uploaded File"] = uploaded_file_name

                    with st.spinner("Analyzing speech transcript..."):
                        nlp_result = run_nlp_prediction(
                            transcript=cleaned_transcript,
                            feature_values=feature_vector,
                        )
                    with st.spinner("Preparing explanation..."):
                        llm_explanation = build_nlp_llm_explanation(
                            result=nlp_result,
                            analysis_features=analysis_features,
                        )

                    report_paths = save_speech_language_reports(
                        result=nlp_result,
                        patient_case_id=patient_id,
                        analysis_id=analysis_id,
                        transcript=cleaned_transcript,
                        analysis_features=analysis_features,
                        patient_info=patient_info,
                        llm_explanation=llm_explanation,
                    )

                    st.session_state.nlp_result = nlp_result
                    st.session_state.nlp_confidence = nlp_result["confidence"]
                    st.session_state.nlp_transcript = cleaned_transcript
                    st.session_state.nlp_cleaned_transcript = cleaned_transcript
                    st.session_state.nlp_extracted_features = features_dict
                    st.session_state.nlp_patient_info = patient_info
                    st.session_state.nlp_analysis_features = analysis_features
                    st.session_state.nlp_llm_explanation = llm_explanation
                    st.session_state.patient_report_path = report_paths["md"]
                    st.session_state.patient_report_html_path = report_paths["html"]
                    st.session_state.patient_report_pdf_path = report_paths["pdf"]
                    st.session_state.patient_report = report_paths["md"].read_text(encoding="utf-8")
                    st.session_state.speech_messages = [{
                        "role": "assistant",
                        "content": "Speech analysis completed. You can ask questions about this result.",
                    }]
                    append_history(
                        analysis_type="Speech and Language Analysis",
                        patient_case_id=patient_id,
                        analysis_id=analysis_id,
                        result=nlp_result["prediction"],
                        confidence=nlp_result["confidence"],
                        report_path=report_paths["html"],
                        notes="Speech and language report generated successfully.",
                    )
                    db_saved = save_speech_result_to_database(
                        patient_case_id=patient_id,
                        analysis_id=analysis_id,
                        patient_info=patient_info,
                        uploaded_file_name=uploaded_file_name,
                        cleaned_transcript=cleaned_transcript,
                        extracted_features=features_dict,
                        result=nlp_result,
                        simple_explanation=llm_explanation,
                        report_paths=report_paths,
                    )
                    warn_database_not_saved(db_saved)
                    note("Speech analysis completed.", "success")
                except Exception as exc:
                    print("Speech prediction error:", repr(exc))
                    show_error(
                        "Speech analysis could not be completed. Please check the uploaded file and information, then try again.",
                        exc,
                    )

    if st.session_state.get("nlp_result"):
        nlp_result = st.session_state.nlp_result
        prediction = nlp_result["prediction"]
        confidence = nlp_result["confidence"]

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        section("Language Assessment Profile")

        if prediction == "ProbableAD":
            result_card(
                "Language Assessment Profile",
                "Probable Alzheimer's Disease Profile",
                percent(confidence),
                "The language assessment detected speech patterns that may be associated "
                "with cognitive decline. This is a decision-support output and is not a diagnosis.",
                "alert",
                "AD",
            )
        elif prediction == "Control":
            result_card(
                "Language Assessment Profile",
                "Within Normal Range",
                percent(confidence),
                "The language assessment did not detect enough speech markers associated "
                "with a probable Alzheimer's disease profile.",
                "good",
                "CN",
            )
        else:
            result_card(
                "Language Assessment Profile",
                prediction_display(prediction),
                percent(confidence),
                "This result is a clinical decision-support output and must be reviewed "
                "by a qualified healthcare professional.",
                "good",
                "-",
            )

        with st.container(border=True):
            metric_cols = st.columns(2)
            metric_cols[0].metric("Assessment Profile",  prediction_display(prediction))
            metric_cols[1].metric("Confidence Score",    percent(confidence))

        section("Clinical Assessment Explanation")
        with st.container(border=True):
            explanation_card(
                "What this result means",
                st.session_state.get("nlp_llm_explanation", "")
                or required_nlp_explanation(prediction, confidence),
            )
            with st.expander("Speech markers observed"):
                st.markdown(observed_linguistic_signs(st.session_state.nlp_analysis_features))

        note(
            "The transcript was automatically extracted from the uploaded speech file. "
            "The result is not a medical diagnosis and must be reviewed by a healthcare professional.",
            "warning",
        )

        if st.session_state.get("patient_report_path"):
            st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
            section("Speech & Language Assessment Report")
            render_report_downloads(Path(st.session_state.patient_report_path), key_prefix="speech")

        st.markdown("<div style='height: 22px'></div>", unsafe_allow_html=True)
        chat_panel("Language Assessment Insight", "speech")


def render_integrated_summary() -> None:
    st.markdown(
        """
        <div class="page-head fu">
          <div class="page-eyebrow">Clinical Review</div>
          <div class="page-title">Integrated Patient Summary</div>
          <div class="page-sub">
            Review brain imaging and speech &amp; language assessment outputs
            side-by-side for clinical interpretation.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    patient_id = ensure_patient_case_id()
    db_connected = refresh_database_status()
    log_database_debug(db_connected)
    if db_connected:
        patient_id = render_patient_case_selector(patient_id)
    render_combined_multimodal_report_section(patient_id)


# -----------------------------------------------------------------------------
# 8) PATIENT REPORTS  — replaces render_reports()
# -----------------------------------------------------------------------------
def render_reports() -> None:
    st.markdown(
        """
        <div class="page-head fu">
          <div class="page-eyebrow">Clinical Records</div>
          <div class="page-title">Patient Reports</div>
          <div class="page-sub">
            Assessment history and downloadable clinical documents for this patient case.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    patient_id = ensure_patient_case_id()
    using_database = db_available()
    if using_database:
        patient_id = render_patient_case_selector(patient_id)
    history = database_history(patient_id) if using_database else load_history()
    reports = database_report_paths(patient_id) if using_database else generated_reports()

    analyses_count = 0 if history.empty else len(history)
    last_date = "Not available" if history.empty else str(history.iloc[-1]["date_time"])
    st.markdown(
        f"""
        <div class="summary-panel fu">
          <div class="summary-grid">
            <div class="summary-item"><div class="summary-lbl">Patient Case ID</div><div class="summary-val mono">{html.escape(patient_id)}</div></div>
            <div class="summary-item"><div class="summary-lbl">Analyses</div><div class="summary-val">{analyses_count}</div></div>
            <div class="summary-item"><div class="summary-lbl">Reports</div><div class="summary-val">{len(reports)}</div></div>
            <div class="summary-item"><div class="summary-lbl">Last analysis</div><div class="summary-val">{html.escape(last_date)}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)
    section("Assessment Timeline")
    if history.empty:
        note("No analysis history available yet.", "info")
    else:
        history_view = history.reset_index(drop=True)
        for row_index, row in history_view.iterrows():
            timeline_card(row["date_time"], row["analysis_type"], row["result"], row["confidence"])
            if (not using_database) and st.button("Delete", key=f"delete_history_{row_index}"):
                delete_history_row(row_index)
                rerun_app()

    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)
    section("Clinical Documents")
    if not reports:
        note("No reports generated yet. Complete an assessment to generate the first report.", "info")
        return

    for label, prefixes in (
        ("Brain Imaging Analysis",       ("brain_scan_report_", "cnn_report_")),
        ("Speech & Language Assessment", ("speech_language_report_", "nlp_report_")),
        ("Integrated Patient Summary",   ("combined_multimodal_report_", "combined_multimodal_summary_")),
    ):
        st.markdown(f'<div class="doc-group-label">{html.escape(label)}</div>', unsafe_allow_html=True)
        typed = [p for p in reports if p.name.lower().startswith(prefixes)]
        if not typed:
            note("No reports generated yet.", "info")
            continue
        for index, report_path in enumerate(typed, start=1):
            meta = report_card_metadata(report_path)
            report_card({"kind": report_kind(report_path),
                         "title": f"{report_kind(report_path)} report",
                         "date": meta["date"]})
            render_report_downloads(report_path, key_prefix=f"reports_{label}_{index}")


def render_about() -> None:
    st.markdown(
        """
        <div class="page-head fu">
          <div class="page-eyebrow">Platform Information</div>
          <div class="page-title">About NeuroMind</div>
          <div class="page-sub">
            Alzheimer clinical decision-support platform — independent brain imaging
            and language assessments for healthcare professionals.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="report-card fade-in">
          <div class="rc-label">Clinical Purpose</div>
          <div class="report-meta" style="margin-top:8px">
            This platform provides independent brain imaging and speech-language assessment
            outputs in a structured clinical dashboard. It supports healthcare professionals
            in reviewing decision-support results, visual explanations, and patient case history.
            It does not provide a medical diagnosis.
          </div>
        </div>
        <div class="report-card fade-in">
          <div class="rc-label">Brain Imaging Analysis</div>
          <div class="report-meta" style="margin-top:8px">
            Structural MRI scans are analyzed to produce a brain imaging profile
            (Cognitively Normal or Alzheimer's Disease-Compatible) with a confidence score.
            A visual explanation highlights the image regions that influenced the result.
            These highlighted regions do not confirm disease.
          </div>
        </div>
        <div class="report-card fade-in">
          <div class="rc-label">Speech &amp; Language Assessment</div>
          <div class="report-meta" style="margin-top:8px">
            Clinical speech transcripts are analyzed for language markers such as hesitation words,
            pauses, self-corrections, and unintelligible words. The extracted markers are used to
            produce a language assessment profile with a confidence score. This is a
            decision-support output and is not a neuropsychological evaluation.
          </div>
        </div>
        <div class="report-card fade-in">
          <div class="rc-label">Assessment Independence</div>
          <div class="report-meta" style="margin-top:8px">
            Brain imaging and speech-language assessments are independent and were trained on
            separate datasets. The platform does not automatically combine both outputs into a
            single clinical conclusion. An Integrated Patient Summary can be generated to
            present both results side-by-side for clinical review — it does not merge them
            into a final diagnosis.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    medical_note_card(
        "This platform provides clinical decision-support outputs only. "
        "All results must be reviewed and interpreted by a qualified healthcare professional "
        "before any clinical action is taken. This platform does not provide a medical diagnosis."
    )


# ---------------------------------------------------------------------------
# App entrypoint
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroMind — Alzheimer Clinical Decision-Support",
    page_icon="N",
    layout="wide",
)

try:
    init_db()
    st.session_state.db_available = True
except Exception:
    st.session_state.db_available = False

ensure_output_dirs()
initialize_state()
inject_theme()
render_sidebar()

_PAGE = st.session_state.get("page", "Home")
if _PAGE == "Brain Scan":
    render_brain_scan()
elif _PAGE == "Speech & Language":
    render_speech()
elif _PAGE == "My Reports":
    render_reports()
elif _PAGE == "Integrated Summary":
    render_integrated_summary()
elif _PAGE == "About":
    render_about()
else:
    render_home()

floating_chat_button()
