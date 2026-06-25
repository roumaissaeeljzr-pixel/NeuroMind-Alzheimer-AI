from __future__ import annotations

import contextlib
import csv
import base64
import html
import importlib
import io
import os
import re
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


def run_cnn_prediction(mri_path: str | Path):
    cnn_predictor = importlib.import_module("cnn_predictor")

    return run_quietly(cnn_predictor.predict_mri, mri_path)


def run_gradcam_generation(
    mri_path: str | Path,
    orientation: str = "multi",
    num_slices: int = 5,
    display_mode: str = "overlay",
    output_dir: str | Path | None = None,
):
    gradcam_3d = importlib.import_module("gradcam_3d")

    gradcam_3d = importlib.reload(gradcam_3d)
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
    main_pipeline = importlib.import_module("main_pipeline")

    return run_quietly(
        main_pipeline.run_full_pipeline,
        transcript=transcript,
        feature_values=feature_values,
    )


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


def warn_database_not_saved(saved: bool) -> None:
    if not saved:
        note("Database is not connected. Results were generated but not saved.", "warning")


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
    if not db_available():
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
    if not db_available():
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
    if not db_available():
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
        "Reports are saved in PostgreSQL and can be recovered by selecting a previous Patient Case ID.",
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
            note("No saved patient cases were found in PostgreSQL yet.", "info")

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
    cols = st.columns(2)
    cols[0].metric("Brain Scan Analysis", combined_status_label(brain_row))
    cols[1].metric("Speech and Language Analysis", combined_status_label(speech_row))


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
    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
    section(
        "Combined Multimodal Report",
        "This combined report summarizes the latest brain scan analysis and speech analysis for this patient case.",
    )
    note(
        "Brain scan analysis and speech analysis are independent and are not combined into a final diagnosis.",
        "info",
    )

    if not db_available():
        note("Database is not connected. Combined reports require PostgreSQL history.", "warning")
        return

    brain_row = get_latest_brain_analysis(patient_identifier(patient_id))
    speech_row = get_latest_speech_analysis(patient_identifier(patient_id))
    render_combined_status_cards(brain_row, speech_row)

    if not brain_row and not speech_row:
        note("No brain scan or speech analysis is available for this patient case yet.", "info")
        generate_disabled = True
    else:
        generate_disabled = False
        if not brain_row:
            note("Brain Scan Analysis: Missing. A partial combined report can still be generated.", "warning")
        if not speech_row:
            note("Speech and Language Analysis: Missing. A partial combined report can still be generated.", "warning")

    if st.button(
        "Generate Combined Report",
        key="generate_combined_report",
        type="primary",
        disabled=generate_disabled,
        use_container_width=True,
    ):
        try:
            with st.spinner("Generating combined report..."):
                case_data = build_combined_case_data(patient_identifier(patient_id))
                rag_summary = generate_combined_rag_summary(case_data)
                paths = save_combined_multimodal_report(case_data, rag_summary)
            st.session_state.combined_report_path = paths.get("md")
            st.session_state.combined_report_html_path = paths.get("html")
            st.session_state.combined_report_pdf_path = paths.get("pdf")
            note("Combined report generated successfully.", "success")
        except Exception as exc:
            show_error("Combined report could not be generated. Please try again.", exc)

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
            title = patient_friendly_source_title(item.get("title") or "Medical source")
            url = str(item.get("url") or "").strip()
        else:
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

    return "\n".join(lines) if lines else "- No information sources were retrieved."


def render_mri_sources(sources) -> None:
    if not sources:
        note("No information sources were retrieved.", "info")
        return

    with st.expander("Information sources used"):
        st.markdown(mri_sources_markdown(sources))


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


def _parse_sex_value(value: str) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
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
    missing = [
        key for key in ("entryage", "sex", "educ")
        if features_dict.get(key) is None or features_dict.get(key) == ""
    ]
    if missing:
        readable = ", ".join(CHA_FEATURE_LABELS.get(key, key) for key in missing)
        raise ValueError(f"Missing required speech information: {readable}")

    vector = []
    for key in CHA_FEATURE_ORDER:
        value = features_dict.get(key)
        if key in {"entryage", "educ"}:
            vector.append(float(value))
        else:
            vector.append(int(value or 0))
    return vector


def speech_word_count(transcript: str) -> int:
    return len(re.findall(r"\b[\w']+\b", transcript or ""))


def speech_feature_display_value(key: str, value) -> str:
    if value is None or value == "":
        return "Missing"
    if key == "sex":
        return "Female" if int(value) == 0 else "Male"
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
) -> str:
    prediction = str(result.get("prediction", "Unknown"))
    badge_class = "badge-good" if prediction == "CN" else "badge-alert"
    uploaded_scan_name = Path(str(mri_source_name or "Not provided")).name

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

    body = header + uploaded_scan + result_summary + simple + visual + color_guide + note_card
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
    )
    md_path.write_text(md_text, encoding="utf-8")

    html_text = generate_brain_scan_report_html(
        result=result,
        patient_case_id=patient_case_id,
        analysis_id=analysis_id,
        gradcam_paths=gradcam_paths,
        gradcam_orientation=gradcam_orientation,
        mri_source_name=mri_source_name,
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
    original_name = Path(uploaded_file.name).name.lower()
    suffix = ".nii.gz" if original_name.endswith(".nii.gz") else ".nii"
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
        "mri_source_name": None,
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
        "mri_source_name": None,
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
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg1:#F8F4EC; --bg2:#E7F6EF; --bg3:#F3E8FF;
            --surface:#FFFDF8; --surface-soft:#FBF8F1;
            --ink:#202124; --muted:#6B7280; --line:#ECE7DB;
            --emerald:#0F9D7A; --deep:#096B5A; --emerald-soft:#E7F6EF;
            --violet:#8B5CF6; --violet-soft:#F3E8FF;
            --coral:#F97362; --coral-soft:#FCE8E4;
            --gold:#C99A2E; --gold-soft:#FBF1DA;
            --shadow:0 18px 44px -28px rgba(20,30,26,0.5);
            --shadow-soft:0 10px 26px -20px rgba(20,30,26,0.42);
            --radius:20px;
        }
        html, body, [class*="css"], .stApp, [data-testid="stMarkdownContainer"] {
            font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        }
        .stApp {
            background:
              radial-gradient(1100px 540px at 86% -8%, rgba(139,92,246,0.10), transparent 60%),
              radial-gradient(960px 500px at -6% 4%, rgba(15,157,122,0.10), transparent 56%),
              linear-gradient(135deg, var(--bg1), var(--bg2) 52%, var(--bg3));
            color:var(--ink);
        }
        [data-testid="stHeader"] { background:transparent; }
        .block-container { padding-top:1.4rem; padding-bottom:5rem; max-width:1180px; }
        h1,h2,h3,h4 { font-family:'Manrope',sans-serif; color:var(--ink); letter-spacing:-0.01em; }

        @keyframes riseIn { from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:translateY(0);} }
        @keyframes halo { 0%,100%{box-shadow:0 0 0 0 rgba(15,157,122,0.22);} 50%{box-shadow:0 0 0 12px rgba(15,157,122,0);} }
        @keyframes floatPulse { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-4px);} }
        .fade-in { animation:riseIn 0.55s cubic-bezier(0.21,0.61,0.35,1) both; }

        /* glass card base used by bordered containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background:rgba(255,253,248,0.86);
            backdrop-filter:blur(8px);
            border:1px solid var(--line) !important;
            border-radius:var(--radius) !important;
            box-shadow:var(--shadow-soft);
            transition:transform 0.22s ease, box-shadow 0.22s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform:translateY(-3px);
            box-shadow:0 22px 46px -26px rgba(20,30,26,0.5);
        }

        /* ---------- Hero ---------- */
        .hero {
            position:relative; overflow:hidden;
            background:linear-gradient(135deg,#ffffff 0%,#eefaf4 50%,#f3ecff 100%);
            border:1px solid var(--line); border-radius:26px;
            padding:38px 40px 32px; box-shadow:var(--shadow);
        }
        .hero::before { content:""; position:absolute; left:0; top:0; bottom:0; width:7px;
            background:linear-gradient(180deg,var(--emerald),var(--violet)); }
        .hero::after { content:""; position:absolute; right:-50px; top:-70px; width:250px; height:250px;
            background-image:repeating-linear-gradient(90deg,rgba(15,157,122,0.10) 0 1px,transparent 1px 16px),
                             repeating-linear-gradient(0deg,rgba(139,92,246,0.10) 0 1px,transparent 1px 16px);
            border-radius:50%;
            -webkit-mask:radial-gradient(circle,#000 0 55%,transparent 62%);
                    mask:radial-gradient(circle,#000 0 55%,transparent 62%); opacity:0.75; }
        .hero-eyebrow { font-size:0.74rem; font-weight:700; letter-spacing:0.22em; text-transform:uppercase; color:var(--deep); margin-bottom:8px; }
        .hero-title { font-family:'Manrope',sans-serif; font-weight:800; font-size:2.3rem; line-height:1.08; margin:0; }
        .hero-sub { font-size:1.06rem; color:var(--muted); margin-top:8px; font-weight:500; }
        .hero-reassure { margin-top:14px; font-size:0.94rem; color:var(--ink);
            background:rgba(255,255,255,0.65); border:1px solid var(--line);
            border-radius:14px; padding:11px 14px; }
        .hero-badges { display:flex; flex-wrap:wrap; gap:9px; margin-top:16px; }
        .chip { display:inline-flex; align-items:center; gap:7px; font-size:0.78rem; font-weight:600;
            padding:6px 14px; border-radius:999px; border:1px solid var(--line); background:var(--surface); color:var(--muted); }
        .chip .d { width:7px; height:7px; border-radius:50%; background:currentColor; }
        .chip-emerald { background:var(--emerald-soft); border-color:#cfe7dc; color:var(--deep); }
        .chip-violet { background:var(--violet-soft); border-color:#e2d4fb; color:#6d3fd1; }
        .chip-gold { background:var(--gold-soft); border-color:#ecdcae; color:var(--gold); }

        /* ---------- Feature cards ---------- */
        .feature-card {
            background:linear-gradient(180deg,#ffffff,var(--surface-soft));
            border:1px solid var(--line); border-radius:18px; padding:20px 20px 18px;
            box-shadow:var(--shadow-soft); height:100%;
            transition:transform 0.2s ease, box-shadow 0.2s ease;
        }
        .feature-card:hover { transform:translateY(-4px); box-shadow:var(--shadow); }
        .feature-ic { width:50px; height:50px; border-radius:15px; display:flex; align-items:center;
            justify-content:center; font-size:1.5rem; color:#fff; margin-bottom:12px; }
        .ic-emerald { background:linear-gradient(135deg,var(--emerald),var(--deep)); }
        .ic-violet { background:linear-gradient(135deg,#a07bf0,var(--violet)); }
        .ic-coral { background:linear-gradient(135deg,#fb9a8c,var(--coral)); }
        .feature-t { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.05rem; margin:0 0 5px; }
        .feature-x { color:var(--muted); font-size:0.9rem; line-height:1.55; }

        /* ---------- Step cards ---------- */
        .step-card { display:flex; gap:14px; align-items:flex-start;
            background:linear-gradient(180deg,#ffffff,var(--surface-soft));
            border:1px solid var(--line); border-radius:16px; padding:16px 18px;
            box-shadow:var(--shadow-soft); margin:4px 0 2px; }
        .step-num { flex:0 0 auto; width:38px; height:38px; border-radius:12px;
            background:linear-gradient(135deg,var(--emerald),var(--deep)); color:#fff;
            font-family:'Manrope',sans-serif; font-weight:800; font-size:1.05rem;
            display:flex; align-items:center; justify-content:center; }
        .step-t { font-family:'Manrope',sans-serif; font-weight:700; font-size:1rem; margin:1px 0 3px; }
        .step-x { color:var(--muted); font-size:0.9rem; }

        /* ---------- Result card / banner ---------- */
        .result-card { border-radius:18px; padding:22px 24px; margin:6px 0 14px;
            border:1px solid var(--line); background:var(--surface); box-shadow:var(--shadow-soft); }
        .result-card .rc-top { display:flex; align-items:center; justify-content:space-between; gap:16px; }
        .rc-label { font-size:0.74rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:var(--muted); }
        .rc-value { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.7rem; line-height:1.1; margin-top:4px; }
        .rc-chip { width:58px; height:58px; border-radius:17px; flex-shrink:0; display:flex; align-items:center;
            justify-content:center; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.05rem; color:#fff; }
        .rc-conf { margin-top:14px; font-size:0.9rem; color:var(--muted); }
        .rc-conf b { color:var(--deep); }
        .rc-exp { margin-top:10px; color:var(--ink); font-size:0.94rem; line-height:1.6; }
        .result-good { border-left:6px solid var(--emerald); }
        .result-good .rc-value { color:var(--deep); }
        .result-good .rc-chip { background:linear-gradient(135deg,var(--emerald),var(--deep)); animation:halo 2.6s ease-in-out infinite; }
        .result-alert { border-left:6px solid var(--coral); }
        .result-alert .rc-value { color:var(--coral); }
        .result-alert .rc-chip { background:linear-gradient(135deg,#fb9a8c,var(--coral)); animation:halo 2.6s ease-in-out infinite; }

        /* ---------- Section titles ---------- */
        .section-title { font-family:'Manrope',sans-serif; font-weight:700; font-size:1.18rem; color:var(--ink);
            margin:4px 0 2px; display:flex; align-items:center; gap:10px; }
        .section-title::before { content:""; width:14px; height:14px; border-radius:4px;
            background:linear-gradient(135deg,var(--emerald),var(--violet)); }
        .section-sub { color:var(--muted); font-size:0.88rem; margin:2px 0 6px 24px; }

        /* ---------- Notes / alerts ---------- */
        .note { border-radius:13px; padding:13px 16px; margin:8px 0; font-size:0.9rem;
            border:1px solid var(--line); background:var(--surface-soft); color:var(--ink); border-left:5px solid var(--muted); }
        .note-success { background:var(--emerald-soft); border-left-color:var(--emerald); color:#114c3b; }
        .note-info { background:var(--surface); border-left-color:var(--deep); }
        .note-warning { background:var(--gold-soft); border-left-color:var(--gold); color:#6f5311; }
        .note-error { background:var(--coral-soft); border-left-color:var(--coral); color:#7a2c20; }

        /* ---------- Explanation cards / legend ---------- */
        .explain-card { background:linear-gradient(180deg,#ffffff,var(--surface-soft)); border:1px solid var(--line);
            border-left:5px solid var(--deep); border-radius:14px; padding:15px 17px; margin:10px 0; box-shadow:var(--shadow-soft); }
        .explain-card.explain-warning { background:var(--gold-soft); border-left-color:var(--gold); }
        .explain-title { font-family:'Manrope',sans-serif; font-weight:800; color:var(--ink); font-size:0.98rem; margin-bottom:6px; }
        .explain-body { color:var(--muted); font-size:0.92rem; line-height:1.58; }
        .explain-list { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-top:11px; }
        .explain-item { background:rgba(255,255,255,0.7); border:1px solid var(--line); border-radius:11px; padding:9px 10px; color:var(--muted); font-size:0.86rem; line-height:1.42; }
        .explain-item strong { color:var(--deep); }
        .color-legend { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; margin-top:11px; }
        .legend-row { display:flex; align-items:center; gap:9px; background:rgba(255,255,255,0.7); border:1px solid var(--line);
            border-radius:11px; padding:10px; color:var(--muted); font-size:0.86rem; line-height:1.35; }
        .legend-dot { width:16px; height:16px; border-radius:50%; flex:0 0 auto; box-shadow:inset 0 0 0 1px rgba(0,0,0,0.15); }
        .legend-low { background:linear-gradient(135deg,#1d2f8f,#10131f); }
        .legend-mid { background:linear-gradient(135deg,#2faf70,#ffd965); }
        .legend-high { background:linear-gradient(135deg,#ffef8a,#d6452f); }

        /* ---------- Timeline ---------- */
        .timeline { position:relative; margin-left:8px; padding-left:22px; }
        .timeline::before { content:""; position:absolute; left:6px; top:6px; bottom:6px; width:2px;
            background:linear-gradient(180deg,var(--emerald),var(--violet)); border-radius:2px; }
        .tl-item { position:relative; margin-bottom:6px; }
        .tl-item::before { content:""; position:absolute; left:-21px; top:7px; width:11px; height:11px; border-radius:50%;
            background:var(--emerald); border:2px solid #fff; box-shadow:0 0 0 2px var(--emerald-soft); }
        .tl-mini { color:var(--ink); font-weight:600; font-size:0.92rem; }
        .tl-mini span { color:var(--muted); font-weight:500; }

        /* ---------- Report card ---------- */
        .report-card { background:linear-gradient(180deg,#ffffff,var(--surface-soft)); border:1px solid var(--line);
            border-radius:16px; padding:16px 18px; box-shadow:var(--shadow-soft); }
        .report-head { display:flex; align-items:center; gap:12px; }
        .report-ic { width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center;
            font-size:1.25rem; color:#fff; flex:0 0 auto; }
        .report-meta { color:var(--muted); font-size:0.84rem; }
        .report-meta b { color:var(--ink); }

        /* ---------- Metric ---------- */
        [data-testid="stMetric"] { background:linear-gradient(180deg,#ffffff,var(--surface-soft)); border:1px solid var(--line);
            border-radius:14px; padding:16px 18px; box-shadow:var(--shadow-soft); }
        [data-testid="stMetricLabel"] p { font-size:0.78rem !important; font-weight:600 !important; letter-spacing:0.04em;
            text-transform:uppercase; color:var(--muted) !important; }
        [data-testid="stMetricValue"] { font-family:'Manrope',sans-serif; font-weight:800; color:var(--deep); }

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
            background:linear-gradient(135deg,var(--emerald),var(--deep)); color:#fff; border:none; border-radius:12px;
            padding:0.55rem 1.15rem; font-weight:600; box-shadow:var(--shadow-soft);
            transition:transform 0.18s ease, filter 0.18s ease; }
        .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            filter:brightness(1.06); transform:translateY(-2px); color:#fff; }
        .stButton > button:disabled { background:#d8d4c8; color:#8a8576; box-shadow:none; transform:none; }

        /* ---------- Inputs ---------- */
        [data-testid="stFileUploader"] { background:var(--surface-soft); border:1.5px dashed #cdd6cf; border-radius:14px; padding:10px 14px; }
        [data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:13px !important; background:var(--surface); box-shadow:var(--shadow-soft); }
        [data-testid="stImageCaption"] { color:var(--muted); font-size:0.8rem; font-weight:600; }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#fbfaf6,#eef3ff); border-right:1px solid var(--line); }
        .side-card { background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:15px 16px; margin-bottom:14px; box-shadow:var(--shadow-soft); }
        .side-brand { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.05rem; color:var(--ink); display:flex; align-items:center; gap:8px; }
        .side-brand .logo { width:30px; height:30px; border-radius:10px; background:linear-gradient(135deg,var(--emerald),var(--violet)); display:flex; align-items:center; justify-content:center; font-size:1rem; }
        .side-brand span { color:var(--deep); }
        .side-tag { font-size:0.74rem; color:var(--muted); margin-top:4px; }
        .case-id { font-family:'Manrope',monospace; font-weight:800; font-size:0.92rem; color:var(--deep);
            background:var(--emerald-soft); border:1px solid #cfe7dc; border-radius:10px; padding:8px 11px; word-break:break-all; margin-top:6px; }
        .side-h { font-size:0.72rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted); margin:2px 0 10px; }
        .status-row { display:flex; align-items:center; justify-content:space-between; padding:7px 0; border-bottom:1px dashed var(--line); font-size:0.86rem; }
        .status-row:last-child { border-bottom:none; }
        .status-name { color:var(--ink); font-weight:500; }
        .status-pill { display:inline-flex; align-items:center; gap:6px; font-size:0.74rem; font-weight:600; padding:3px 10px; border-radius:999px; }
        .status-pill .dot { width:7px; height:7px; border-radius:50%; }
        .pill-active { background:var(--emerald-soft); color:var(--deep); }
        .pill-active .dot { background:var(--emerald); }
        .pill-idle { background:#efece3; color:var(--muted); }
        .pill-idle .dot { background:#b6b1a3; }
        .side-note { font-size:0.78rem; color:var(--muted); line-height:1.5; }

        /* ---------- Floating chat ---------- */
        .fc-toggle { position:fixed; opacity:0; pointer-events:none; }
        .fc-btn { position:fixed; right:22px; bottom:26px; z-index:9998; width:58px; height:58px; border-radius:50%;
            background:linear-gradient(135deg,var(--emerald),var(--violet)); color:#fff; font-size:1.5rem;
            display:flex; align-items:center; justify-content:center; cursor:pointer;
            box-shadow:0 14px 30px -10px rgba(15,157,122,0.6); animation:floatPulse 3.4s ease-in-out infinite; transition:transform 0.18s ease; }
        .fc-btn:hover { transform:scale(1.06); }
        .fc-panel { position:fixed; right:22px; bottom:96px; z-index:9997; width:300px; max-height:340px; overflow:auto;
            display:none; background:rgba(255,253,248,0.96); backdrop-filter:blur(10px);
            border:1px solid var(--line); border-radius:18px; padding:16px 17px; box-shadow:var(--shadow); }
        .fc-toggle:checked ~ .fc-panel { display:block; animation:riseIn 0.3s ease both; }
        .fc-h { font-family:'Manrope',sans-serif; font-weight:800; font-size:0.98rem; color:var(--ink); margin-bottom:6px; }
        .fc-msg { background:var(--emerald-soft); border:1px solid #cfe7dc; border-radius:12px; padding:10px 12px; color:#114c3b; font-size:0.88rem; line-height:1.5; }
        .fc-tip { margin-top:10px; font-size:0.82rem; color:var(--muted); line-height:1.5; }
        .fc-note { margin-top:10px; font-size:0.78rem; color:var(--gold); background:var(--gold-soft); border:1px solid #ecdcae; border-radius:10px; padding:8px 10px; }

        @media (max-width:860px) {
            .explain-list, .color-legend { grid-template-columns:1fr; }
            .hero-title { font-size:1.9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------
def note(text: str, kind: str = "info") -> None:
    st.markdown(
        f'<div class="note note-{kind}">{html.escape(patient_friendly_text(text))}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    sub = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="fade-in"><div class="section-title">{html.escape(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def show_error(message: str, exc: Exception | None = None) -> None:
    _ = exc
    note(message, "error")


def hero_card(title: str, subtitle: str, reassure: str) -> None:
    st.markdown(
        f"""
        <div class="hero fade-in">
          <div class="hero-eyebrow">AI Decision-Support Platform</div>
          <div class="hero-title">{html.escape(title)}</div>
          <div class="hero-sub">{html.escape(subtitle)}</div>
          <div class="hero-reassure">{html.escape(reassure)}</div>
          <div class="hero-badges">
            <span class="chip chip-emerald"><span class="d"></span>Brain Scan</span>
            <span class="chip chip-violet"><span class="d"></span>Speech &amp; Language</span>
            <span class="chip chip-gold"><span class="d"></span>Patient-friendly reports</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, text: str, tone: str = "emerald") -> None:
    st.markdown(
        f"""
        <div class="feature-card fade-in">
          <div class="feature-ic ic-{tone}">{icon}</div>
          <div class="feature-t">{html.escape(title)}</div>
          <div class="feature-x">{html.escape(text)}</div>
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
    st.markdown(
        f"""
        <div class="result-card result-{tone} fade-in">
          <div class="rc-top">
            <div>
              <div class="rc-label">{html.escape(title)}</div>
              <div class="rc-value">{html.escape(result)}</div>
            </div>
            <div class="rc-chip">{html.escape(chip)}</div>
          </div>
          <div class="rc-conf">Confidence: <b>{html.escape(confidence)}</b></div>
          <div class="rc-exp">{html.escape(explanation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def medical_note_card(text: str) -> None:
    note(text, "warning")


def patient_case_card(patient_case_id: str) -> None:
    st.markdown(
        f"""
        <div class="report-card fade-in">
          <div class="report-head">
            <div class="report-ic" style="background:linear-gradient(135deg,var(--emerald),var(--violet));">🆔</div>
            <div>
              <div class="rc-label">Patient Case ID</div>
              <div class="case-id" style="margin-top:4px">{html.escape(patient_case_id)}</div>
            </div>
          </div>
          <div class="report-meta" style="margin-top:10px">
            This ID is created automatically and links every analysis and report in this case.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
          <div class="explain-title">Color guide</div>
          <div class="explain-body">The colors show influence on the result, not a diagnosis.</div>
          <div class="color-legend">
            <div class="legend-row"><span class="legend-dot legend-low"></span>
              <span><strong>Blue / dark areas:</strong> lower influence on the result</span></div>
            <div class="legend-row"><span class="legend-dot legend-mid"></span>
              <span><strong>Green / yellow areas:</strong> moderate influence on the result</span></div>
            <div class="legend-row"><span class="legend-dot legend-high"></span>
              <span><strong>Yellow / red areas:</strong> stronger influence, not confirmed disease</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def floating_chat_button() -> None:
    st.markdown(
        """
        <input type="checkbox" id="floatchat" class="fc-toggle">
        <label for="floatchat" class="fc-btn" title="Help">💬</label>
        <div class="fc-panel">
          <div class="fc-h">How can I help you understand this result?</div>
          <div class="fc-msg">Open the <strong>Ask a question</strong> section at the bottom of the
            Brain Scan or Speech &amp; Language page to ask about your result.</div>
          <div class="fc-tip">Example questions:<br>• What does this result mean?<br>
            • Do the colored areas mean disease?<br>• Is this a diagnosis?</div>
          <div class="fc-note">This is not a medical diagnosis. Please review the result with a healthcare professional.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        return f'<span class="status-pill pill-active"><span class="dot"></span>{html.escape(active_label)}</span>'
    return f'<span class="status-pill pill-idle"><span class="dot"></span>{html.escape(idle_label)}</span>'


def timeline_card(date: str, analysis_type: str, result: str, confidence: str) -> None:
    tone = "alert" if str(result) in {"AD", "ProbableAD"} else "good"
    st.markdown(
        f"""
        <div class="report-card fade-in">
          <div class="report-head">
            <div class="report-ic" style="background:linear-gradient(135deg,var(--emerald),var(--violet));">🗂️</div>
            <div>
              <div class="rc-label">{html.escape(friendly_analysis_type(analysis_type))}</div>
              <div class="tl-mini" style="margin-top:2px">{html.escape(str(result))}
                <span>· {html.escape(str(confidence))}</span></div>
            </div>
          </div>
          <div class="report-meta" style="margin-top:10px"><b>Date:</b> {html.escape(str(date))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _ = tone


def report_card(report_info: dict) -> None:
    icon = "🧠" if report_info.get("kind") == "Brain Scan" else "🗣️"
    st.markdown(
        f"""
        <div class="report-card fade-in">
          <div class="report-head">
            <div class="report-ic" style="background:linear-gradient(135deg,var(--emerald),var(--deep));">{icon}</div>
            <div>
              <div class="rc-label">{html.escape(report_info.get("title", "Report"))}</div>
              <div class="report-meta" style="margin-top:3px"><b>Date:</b> {html.escape(report_info.get("date", "Not available"))}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chat_panel(title: str, context_type: str) -> None:
    """
    Render a patient-friendly chat at the bottom of an analysis page.
    context_type is either "brain" or "speech"; both keep the two analyses
    independent and always remind that the result is not a diagnosis.
    """
    section(title, "Answers are simple and are never a medical diagnosis.")
    with st.container(border=True):
        if context_type == "brain":
            messages_key = "brain_messages"
            placeholder = "Ask a question about this brain scan result..."
            examples = (
                "Example questions: What does this result mean? Do the colored "
                "areas mean disease? Is this a diagnosis?"
            )
        else:
            messages_key = "speech_messages"
            placeholder = "Ask a question about this speech result..."
            examples = (
                "Example questions: What does this result mean? What do the speech "
                "patterns show? Is this a diagnosis?"
            )

        st.caption(examples)
        for message in st.session_state.get(messages_key, []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input(placeholder, key=f"chat_input_{context_type}")
        if user_question:
            st.session_state[messages_key].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Preparing answer..."):
                    try:
                        if context_type == "brain":
                            result = st.session_state.get("cnn_result") or {}
                            rag = run_mri_rag_explanation(
                                result,
                                gradcam_info=current_gradcam_info(),
                                question=user_question,
                            )
                            answer = str(rag.get("answer", "")).strip()
                            if not answer or answer == "MRI explanation is currently unavailable.":
                                answer = (
                                    "This answer is currently unavailable. Please review "
                                    "the result with a healthcare professional."
                                )
                            else:
                                src = mri_sources_markdown(rag.get("sources", []))
                                answer = f"{answer}\n\n**Information Sources**\n{src}"
                            answer = (
                                f"{answer}\n\n_This is not a medical diagnosis. Please review "
                                "the result with a healthcare professional._"
                            )
                        else:
                            answer = build_chat_answer(
                                question=user_question,
                                transcript=st.session_state.get("nlp_transcript", ""),
                                prediction=(st.session_state.get("nlp_result") or {}).get("prediction", ""),
                                confidence=(st.session_state.get("nlp_result") or {}).get("confidence", 0.0),
                            )
                    except Exception:
                        answer = (
                            "This answer could not be generated right now. Please try a "
                            "different question. This is not a medical diagnosis."
                        )
                st.markdown(answer)
            st.session_state[messages_key].append({"role": "assistant", "content": answer})


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="side-card">
              <div class="side-brand"><span class="logo">🧠</span>Alzheimer <span>Assistant</span></div>
              <div class="side-tag">Patient decision support</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        case_id = ensure_patient_case_id()
        st.markdown(
            f"""
            <div class="side-card">
              <div class="side-h">Patient Case ID</div>
              <div class="case-id">{html.escape(case_id)}</div>
              <div class="side-note" style="margin-top:6px">
                Created automatically and used for every analysis and report in this case.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start New Patient Case", key="start_new_patient_case", use_container_width=True):
            reset_current_patient_case()
            rerun_app()

        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.markdown('<div class="side-h">Navigation</div>', unsafe_allow_html=True)
        pages = [
            ("Home", "🏠 Home"),
            ("Brain Scan", "🧠 Brain Scan"),
            ("Speech & Language", "🗣️ Speech & Language"),
            ("My Reports", "📄 My Reports"),
            ("About", "ℹ️ About"),
        ]
        for page_key, page_label in pages:
            if st.button(page_label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                rerun_app()
        st.markdown("</div>", unsafe_allow_html=True)

        cnn_active = st.session_state.get("cnn_result") is not None
        nlp_active = st.session_state.get("nlp_result") is not None
        reports_active = bool(generated_reports())

        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.markdown('<div class="side-h">Status</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="status-row"><span class="status-name">Brain scan</span>
              {status_pill(cnn_active, "Completed", "Ready")}</div>
            <div class="status-row"><span class="status-name">Speech analysis</span>
              {status_pill(nlp_active, "Completed", "Ready")}</div>
            <div class="status-row"><span class="status-name">Reports</span>
              {status_pill(reports_active, "Available", "Not yet")}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="side-card"><div class="side-h">Medical Note</div>'
            f'<div class="side-note">{html.escape(MEDICAL_NOTE)}</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_home() -> None:
    hero_card(
        "Alzheimer Multimodal Assistant",
        "A patient-friendly decision-support platform for brain scan and speech analysis.",
        "This platform helps organize AI-assisted results in a clear and understandable "
        "way. It does not provide a medical diagnosis.",
    )

    st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        feature_card("🧠", "Brain Scan Analysis",
                     "Upload a brain scan and receive a simple explanation of the result.", "emerald")
    with cols[1]:
        feature_card("🗣️", "Speech and Language Analysis",
                     "Enter a speech transcript and receive a clear language-based result.", "violet")
    with cols[2]:
        feature_card("📄", "My Reports",
                     "Review and download patient-friendly reports.", "gold")

    st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
    case_col, time_col = st.columns([1, 1])
    with case_col:
        patient_case_card(ensure_patient_case_id())
    with time_col:
        st.markdown(
            """
            <div class="report-card fade-in">
              <div class="rc-label" style="margin-bottom:8px">How it works</div>
              <div class="timeline">
                <div class="tl-item"><div class="tl-mini">Start patient case</div></div>
                <div class="tl-item"><div class="tl-mini">Add information</div></div>
                <div class="tl-item"><div class="tl-mini">Run analysis</div></div>
                <div class="tl-item"><div class="tl-mini">Review explanation</div></div>
                <div class="tl-item"><div class="tl-mini">Download report</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
    medical_note_card(
        "This system is not a medical diagnosis. Results must be reviewed by a "
        "healthcare professional."
    )


def render_brain_scan() -> None:
    section("Brain Scan Analysis",
            "Upload a brain scan and get a clear, patient-friendly result explanation.")
    medical_note_card(MEDICAL_NOTE)

    step_card(1, "Step 1 — Upload brain scan", "Choose a brain scan file to begin.")
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
                st.session_state.show_mri_preview = False
                st.session_state.mri_rag_explanation = ""
                st.session_state.mri_rag_sources = []
                st.session_state.brain_messages = []
            note("Scan selected successfully. Continue to Step 2 to analyze it.", "success")

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    step_card(2, "Step 2 — Analyze brain scan", "Start the analysis of the selected scan.")
    with st.container(border=True):
        analyze_clicked = st.button(
            "Analyze brain scan",
            disabled=uploaded_mri is None,
            key="analyze_mri_button",
            use_container_width=True,
        )

    if analyze_clicked and uploaded_mri is not None:
        try:
            patient_id = ensure_patient_case_id()
            analysis_id = generate_analysis_id()
            st.session_state.cnn_analysis_id = analysis_id
            saved_path = save_uploaded_mri(uploaded_mri)
            st.session_state.mri_path = str(saved_path)
            st.session_state.mri_source_name = uploaded_mri.name

            with st.spinner("Analyzing the brain scan..."):
                cnn_result = run_cnn_prediction(saved_path)

            st.session_state.cnn_result = cnn_result
            st.session_state.gradcam_paths = []
            st.session_state.gradcam_generated_orientation = None
            st.session_state.gradcam_generated_display_mode = None
            st.session_state.show_mri_preview = False
            st.session_state.mri_rag_explanation = ""
            st.session_state.mri_rag_sources = []
            st.session_state.brain_messages = []
            st.session_state.cnn_preview_path = save_mri_preview_image(saved_path, patient_id)

            report_paths = save_brain_scan_reports(
                result=cnn_result,
                patient_case_id=patient_id,
                analysis_id=analysis_id,
                mri_preview_path=st.session_state.cnn_preview_path,
                mri_source_name=uploaded_mri.name,
            )
            st.session_state.cnn_report_path = report_paths["md"]
            st.session_state.cnn_report_html_path = report_paths["html"]
            st.session_state.cnn_report_pdf_path = report_paths["pdf"]

            append_history(
                analysis_type="Brain Scan Analysis",
                patient_case_id=patient_id,
                analysis_id=analysis_id,
                result=cnn_result["prediction"],
                confidence=cnn_result["confidence"],
                report_path=report_paths["html"],
                notes="Brain scan report generated successfully.",
            )
            db_saved = save_brain_result_to_database(
                patient_case_id=patient_id,
                analysis_id=analysis_id,
                result=cnn_result,
                uploaded_file_name=uploaded_mri.name,
                mri_file_path=saved_path,
                report_paths=report_paths,
            )
            warn_database_not_saved(db_saved)
            note("Brain scan analysis completed.", "success")
        except Exception as exc:
            show_error("The analysis could not be completed. Please check the input and try again.", exc)

    if st.session_state.get("cnn_result"):
        result = st.session_state.cnn_result
        prediction = result["prediction"]

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        section("Brain Scan Result")

        if prediction == "CN":
            result_card("Brain Scan Result", "Cognitively Normal Profile",
                        percent(result["confidence"]),
                        "The system did not detect a brain scan pattern strongly associated "
                        "with Alzheimer's disease in this scan.", "good", "CN")
        elif prediction == "AD":
            result_card("Brain Scan Result", "Alzheimer's Disease-Compatible Profile",
                        percent(result["confidence"]),
                        "The system detected a brain scan pattern that may be associated with "
                        "Alzheimer's disease. This is not a diagnosis.", "alert", "AD")
        else:
            result_card("Brain Scan Result", str(prediction),
                        percent(result["confidence"]),
                        "This result is a decision-support output and must be reviewed by a "
                        "healthcare professional.", "good", "—")

        with st.container(border=True):
            metric_cols = st.columns(3)
            metric_cols[0].metric("Confidence", percent(result["confidence"]))
            metric_cols[1].metric("Cognitively Normal", percent(result["prob_cn"]))
            metric_cols[2].metric("Alzheimer's Compatible", percent(result["prob_ad"]))

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
        section("Simple Explanation")
        with st.container(border=True):
            explanation_card("What this result means", brain_scan_simple_explanation(prediction))
            if st.button("Generate detailed explanation", key="generate_mri_rag_explanation"):
                try:
                    with st.spinner("Preparing explanation..."):
                        rag_result = run_mri_rag_explanation(result, gradcam_info=current_gradcam_info())
                    answer = str(rag_result.get("answer", "")).strip()
                    if not answer or answer == "MRI explanation is currently unavailable.":
                        note("A detailed explanation is currently unavailable.", "warning")
                    else:
                        st.session_state.mri_rag_explanation = answer
                        st.session_state.mri_rag_sources = rag_result.get("sources", [])
                        write_updated_cnn_report(result, ensure_patient_id())
                        note("Explanation generated successfully.", "success")
                except Exception as exc:
                    show_error("A detailed explanation is currently unavailable.", exc)

            if st.session_state.get("mri_rag_explanation"):
                with st.container(border=True):
                    st.markdown("##### Detailed explanation")
                    st.write(patient_friendly_text(st.session_state.mri_rag_explanation))
                render_mri_sources(st.session_state.get("mri_rag_sources", []))

        if st.session_state.get("cnn_report_path"):
            st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
            section("Brain Scan Report")
            render_report_downloads(Path(st.session_state.cnn_report_path))

        st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
        chat_panel("Ask a question about this result", "brain")


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
            section("Speech and Language Report")
            render_report_downloads(Path(st.session_state.patient_report_path), key_prefix="speech")

        st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
        chat_panel("Ask a question about this speech result", "speech")


def render_speech() -> None:
    section(
        "Speech and Language Analysis",
        "Upload a speech transcript file and receive a clear language-based result.",
    )
    medical_note_card(MEDICAL_NOTE)

    patient_id = ensure_patient_case_id()

    step_card(1, "Step 1 - Patient Information", "Enter the patient details for this speech analysis.")
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
    step_card(2, "Step 2 - Upload Speech File", "Upload one CHAT transcript file.")
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
                "Step 3 - Transcript Preview",
                "Review the automatically extracted speech text before analysis.",
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
                "Step 4 - Automatically Extracted Speech Information",
                "These values were extracted from the speech file and used for the analysis.",
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
            step_card(5, "Step 5 - Analyze Speech", "Run the speech and language analysis.")
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
                    feature_vector = [
                        features_dict["n_filled_pauses"],
                        features_dict["n_phon_fragments"],
                        features_dict["n_paralinguistic"],
                        features_dict["n_retracings"],
                        features_dict["n_unintelligible"],
                        features_dict["n_pauses"],
                        features_dict["entryage"],
                        features_dict["sex"],
                        features_dict["educ"],
                    ]
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
                    show_error(
                        "Speech analysis could not be completed. Please check the uploaded file and information, then try again.",
                        exc,
                    )

    if st.session_state.get("nlp_result"):
        nlp_result = st.session_state.nlp_result
        prediction = nlp_result["prediction"]
        confidence = nlp_result["confidence"]

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        section("Step 6 - Speech Analysis Result")

        if prediction == "ProbableAD":
            result_card(
                "Speech Analysis Result",
                "ProbableAD",
                percent(confidence),
                "The speech analysis detected language patterns that may be associated "
                "with cognitive decline. This is not a diagnosis.",
                "alert",
                "AD",
            )
        elif prediction == "Control":
            result_card(
                "Speech Analysis Result",
                "Control",
                percent(confidence),
                "The speech analysis did not detect enough language patterns associated "
                "with a probable Alzheimer's disease profile.",
                "good",
                "CN",
            )
        else:
            result_card(
                "Speech Analysis Result",
                prediction_display(prediction),
                percent(confidence),
                "This result is a decision-support output and must be reviewed by a "
                "healthcare professional.",
                "good",
                "-",
            )

        with st.container(border=True):
            metric_cols = st.columns(2)
            metric_cols[0].metric("Result", prediction_display(prediction))
            metric_cols[1].metric("Confidence Score", percent(confidence))

        section("Simple Explanation")
        with st.container(border=True):
            explanation_card(
                "What this result means",
                st.session_state.get("nlp_llm_explanation", "")
                or required_nlp_explanation(prediction, confidence),
            )
            with st.expander("Speech patterns observed"):
                st.markdown(observed_linguistic_signs(st.session_state.nlp_analysis_features))

        note(
            "The transcript was automatically extracted from the uploaded speech file. "
            "The result is not a medical diagnosis and must be reviewed by a healthcare professional.",
            "warning",
        )

        if st.session_state.get("patient_report_path"):
            st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
            section("Speech and Language Report")
            render_report_downloads(Path(st.session_state.patient_report_path), key_prefix="speech")

        st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
        chat_panel("Ask a question about this speech result", "speech")


def render_reports() -> None:
    section("My Reports", "Review past analyses and download patient-friendly reports.")
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
        <div class="report-card fade-in">
          <div class="rc-label" style="margin-bottom:8px">Patient Case Summary</div>
          <div class="report-meta"><b>Patient Case ID:</b> {html.escape(patient_id)}</div>
          <div class="report-meta"><b>Number of analyses:</b> {analyses_count}</div>
          <div class="report-meta"><b>Number of reports:</b> {len(reports)}</div>
          <div class="report-meta"><b>Last analysis date:</b> {html.escape(last_date)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
    section("Analysis Timeline")
    if history.empty:
        note("No analysis history available yet.", "info")
    else:
        history_view = history.reset_index(drop=True)
        for row_index, row in history_view.iterrows():
            timeline_card(row["date_time"], row["analysis_type"], row["result"], row["confidence"])
            if (not using_database) and st.button("Delete", key=f"delete_history_{row_index}"):
                delete_history_row(row_index)
                rerun_app()

    render_combined_multimodal_report_section(patient_id)

    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)
    section("Generated Reports")
    if not reports:
        note("No reports generated yet.", "info")
        return

    for label, prefixes in (
        ("Brain scan reports", ("brain_scan_report_", "cnn_report_")),
        ("Speech and language reports", ("speech_language_report_", "nlp_report_")),
        ("Combined multimodal reports", ("combined_multimodal_report_", "combined_multimodal_summary_")),
    ):
        st.markdown(f"##### {label}")
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
    section("About")
    st.markdown(
        """
        <div class="report-card fade-in">
          <div class="rc-label">What this platform does</div>
          <div class="report-meta" style="margin-top:6px">
            It organizes AI-assisted brain scan and speech results in a clear, patient-friendly
            way, with simple explanations, visual explanations and downloadable reports.
          </div>
        </div>
        <div class="report-card fade-in">
          <div class="rc-label">What this platform does not do</div>
          <div class="report-meta" style="margin-top:6px">
            It does not provide a medical diagnosis. Brain scan analysis and speech analysis are
            independent. The system does not combine both analyses into a final diagnosis.
          </div>
        </div>
        <div class="report-card fade-in">
          <div class="rc-label">How to understand the results</div>
          <div class="report-meta" style="margin-top:6px">
            Each result is a decision-support output with a confidence score. Visual explanations
            show which areas influenced a brain scan result; they do not confirm disease. A
            healthcare professional should interpret every result.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    medical_note_card(
        "This platform does not provide a medical diagnosis. Results must be interpreted "
        "by a healthcare professional."
    )


# ---------------------------------------------------------------------------
# App entrypoint
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Alzheimer Multimodal Assistant",
    page_icon="🧠",
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
elif _PAGE == "About":
    render_about()
else:
    render_home()

floating_chat_button()
