from __future__ import annotations

import base64
import html
import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from database import db as database
from database.db import (
    get_latest_brain_analysis,
    get_latest_speech_analysis,
    get_reports,
    save_report,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
NLP_SRC_DIR = PROJECT_DIR / "nlp_rag_module" / "src"
CNN_SRC_DIR = PROJECT_DIR / "cnn_module" / "src"

for module_dir in (NLP_SRC_DIR, CNN_SRC_DIR):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)


INDEPENDENCE_NOTE = (
    "Brain scan analysis and speech analysis are independent. This report "
    "summarizes both outputs but does not combine them into a medical diagnosis."
)
MEDICAL_NOTE = (
    "This report is not a medical diagnosis. The results must be reviewed by "
    "a healthcare professional."
)
FORBIDDEN_TERMS = (
    "CNN",
    "NLP",
    "RAG",
    "LLM",
    "FAISS",
    "embedding",
    "logits",
    "tensor",
    "Grad-CAM",
    "XAI",
)
FORBIDDEN_DIAGNOSTIC_PHRASES = (
    "final diagnosis",
    "confirmed Alzheimer",
    "patient has Alzheimer",
    "definitive conclusion",
)
FRIENDLY_TERM_REPLACEMENTS = {
    "CNN": "Brain Scan Analysis",
    "NLP": "Speech and Language Analysis",
    "RAG": "Information Sources",
    "LLM": "language helper",
    "FAISS": "medical knowledge base",
    "embedding": "information source",
    "embeddings": "information sources",
    "logits": "internal scores",
    "tensor": "internal data",
    "Grad-CAM": "Visual Explanation",
    "gradcam": "Visual Explanation",
    "XAI": "Visual Explanation",
}
SPEECH_FEATURE_LABELS = {
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


def ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[A-Za-z]:\\[^\s\"']+", "", text)
    text = re.sub(r"(?:/[\w.-]+){3,}", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^\s*#+\s*", "", text)
    text = re.sub(
        r"here(?:'|’)s\s+a\s+four[- ]paragraph\s+breakdown[:\s]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bour clinical review suggests\b",
        "the saved analysis outputs suggest",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bappears to have cognitively normal cognitive function\b",
        "was classified as a Cognitively Normal profile",
        text,
        flags=re.IGNORECASE,
    )
    for source, target in FRIENDLY_TERM_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text, flags=re.IGNORECASE)
    for phrase in FORBIDDEN_DIAGNOSTIC_PHRASES:
        if phrase == "final diagnosis":
            continue
        text = re.sub(re.escape(phrase), "decision-support summary", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def display_value(value: Any, fallback: str = "Not provided") -> str:
    text = clean_text(value)
    return text if text else fallback


def percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if number <= 1:
        number *= 100
    return f"{number:.2f}%"


def report_filename_part(value: str, fallback: str = "unknown") -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return text or fallback


def generate_combined_analysis_id() -> str:
    return "AN-COMBINED-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_project_path(value: str | Path | None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    try:
        return path.resolve()
    except Exception:
        return path


def relative_report_path(value: str | Path | None) -> str | None:
    path = resolve_project_path(value)
    if not path:
        return None
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except Exception:
        return path.name


def image_data_uri(image_path: str | Path | None) -> str | None:
    path = resolve_project_path(image_path)
    if not path or not path.exists():
        return None
    try:
        raw = path.read_bytes()
        suffix = path.suffix.lower().lstrip(".") or "png"
        if suffix == "jpg":
            suffix = "jpeg"
        return f"data:image/{suffix};base64,{base64.b64encode(raw).decode('ascii')}"
    except Exception:
        return None


def parse_features(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def friendly_feature_rows(features: Any) -> list[tuple[str, str]]:
    parsed = parse_features(features)
    rows = []
    ordered_keys = [
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
    zero_if_missing = {
        "n_phon_fragments",
        "n_paralinguistic",
        "n_unintelligible",
    }
    for key in ordered_keys:
        label = SPEECH_FEATURE_LABELS.get(key, key)
        value = parsed.get(key)
        if (value is None or value == "") and key in zero_if_missing:
            value = 0
        if key == "sex":
            if str(value) == "0":
                value = "Female"
            elif str(value) == "1":
                value = "Male"
        rows.append((label, display_value(value, fallback="0" if key in zero_if_missing else "Not provided")))
    return rows


def fetch_patient(patient_case_id: str) -> dict[str, Any]:
    engine = database.get_engine()
    if engine is None or database.text is None:
        return {"patient_case_id": patient_case_id}

    try:
        with engine.connect() as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    database.text(
                        """
                        SELECT
                            patient_case_id,
                            patient_name,
                            study_date,
                            responsible_clinician,
                            clinical_notes,
                            created_at
                        FROM patients
                        WHERE patient_case_id = :patient_case_id
                        LIMIT 1
                        """
                    ),
                    {"patient_case_id": patient_case_id},
                ).mappings().all()
            ]
        return rows[0] if rows else {"patient_case_id": patient_case_id}
    except Exception:
        return {"patient_case_id": patient_case_id}


def normalize_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    normalized = dict(row)
    for key, value in list(normalized.items()):
        if hasattr(value, "isoformat"):
            normalized[key] = value.isoformat(sep=" ", timespec="seconds")
    return normalized


def build_combined_case_data(patient_case_id: str) -> dict:
    patient_case_id = str(patient_case_id or "").strip()
    if not patient_case_id:
        raise ValueError("patient_case_id is required.")

    patient = normalize_row(fetch_patient(patient_case_id)) or {"patient_case_id": patient_case_id}
    speech_analysis = normalize_row(get_latest_speech_analysis(patient_case_id))
    brain_analysis = normalize_row(get_latest_brain_analysis(patient_case_id))
    reports = [normalize_row(row) for row in get_reports(patient_case_id)]

    case_data = {
        "patient_case_id": patient_case_id,
        "combined_analysis_id": generate_combined_analysis_id(),
        "patient": patient,
        "speech_analysis": speech_analysis,
        "brain_analysis": brain_analysis,
        "reports": [row for row in reports if row],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return case_data


def brain_result_summary(brain: dict[str, Any] | None) -> str:
    if not brain:
        return "Brain scan analysis was not available for this patient case."

    result = display_value(brain.get("result"))
    if result == "CN":
        result_text = "Cognitively Normal profile"
    elif result == "AD":
        result_text = "Alzheimer's disease-compatible profile"
    else:
        result_text = result

    explanation = clean_text(brain.get("simple_explanation"))
    if not explanation:
        if result == "CN":
            explanation = (
                "The brain scan analysis did not detect a scan pattern strongly "
                "associated with Alzheimer's disease in this scan."
            )
        elif result == "AD":
            explanation = (
                "The brain scan analysis detected a scan pattern that may be "
                "associated with Alzheimer's disease. This is not a diagnosis."
            )
        else:
            explanation = "The brain scan result is a decision-support output."

    return (
        f"The brain scan analysis result was {result_text} with a confidence "
        f"score of {percent(brain.get('confidence'))}. {explanation}"
    )


def speech_result_summary(speech: dict[str, Any] | None) -> str:
    if not speech:
        return "Speech analysis was not available for this patient case."

    result = display_value(speech.get("result"))
    explanation = clean_text(speech.get("simple_explanation"))
    if not explanation:
        if result == "Control":
            explanation = (
                "The speech analysis did not detect enough language signs "
                "associated with a probable Alzheimer's disease profile."
            )
        elif result == "ProbableAD":
            explanation = (
                "The speech analysis detected language patterns that may be "
                "associated with cognitive decline, such as hesitations, "
                "repetitions, pauses, or difficulty organizing speech."
            )
        else:
            explanation = "The speech result is a decision-support output."

    return (
        f"The speech and language analysis result was {result} with a confidence "
        f"score of {percent(speech.get('confidence'))}. {explanation}"
    )


def agreement_summary(brain: dict[str, Any] | None, speech: dict[str, Any] | None) -> str:
    if not brain and not speech:
        return (
            "Neither analysis was available, so this report can only provide "
            "general guidance."
        )
    if not brain:
        return (
            "Only the speech and language analysis was available. A brain scan "
            "summary cannot be included yet."
        )
    if not speech:
        return (
            "Only the brain scan analysis was available. A speech and language "
            "summary cannot be included yet."
        )

    brain_result = str(brain.get("result") or "").strip()
    speech_result = str(speech.get("result") or "").strip()
    both_low = brain_result == "CN" and speech_result == "Control"
    both_flagged = brain_result == "AD" and speech_result == "ProbableAD"

    if both_low:
        return (
            "Both independent outputs point toward a lower concern profile in "
            "the available data. This still requires clinical review."
        )
    if both_flagged:
        return (
            "Both independent outputs show patterns that may require closer "
            "clinical review. They still do not confirm a diagnosis."
        )
    return (
        "The two independent outputs do not point in exactly the same direction. "
        "A healthcare professional should review both results with the full "
        "clinical context."
    )


def speech_count(features: Any, key: str) -> int:
    parsed = parse_features(features)
    try:
        return int(float(parsed.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def count_text(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return f"one {singular}"
    return f"{count} {plural or singular + 's'}"


def speech_pattern_summary(speech: dict[str, Any] | None) -> str:
    if not speech:
        return "No speech feature summary was available."

    features = speech.get("extracted_features")
    specs = [
        ("n_filled_pauses", "hesitation word", "hesitation words"),
        ("n_phon_fragments", "interrupted word", "interrupted words"),
        ("n_paralinguistic", "non-verbal speech marker", "non-verbal speech markers"),
        ("n_retracings", "self-correction", "self-corrections"),
        ("n_unintelligible", "unclear word", "unclear words"),
        ("n_pauses", "pause", "pauses"),
    ]
    present = []
    for key, singular, plural in specs:
        count = speech_count(features, key)
        if count > 0:
            present.append(count_text(count, singular, plural))

    if present:
        return ", ".join(present)
    return (
        "no detected hesitation words, interrupted words, non-verbal speech "
        "markers, self-corrections, unclear words, or pauses"
    )


def safe_brain_section(brain: dict[str, Any] | None) -> str:
    if not brain:
        return "Brain scan analysis was not available for this patient case."

    result = str(brain.get("result") or "").strip()
    confidence = percent(brain.get("confidence"))
    if result == "CN":
        return (
            "The brain scan analysis was classified as a Cognitively Normal profile. "
            "This means the system did not detect a brain scan pattern strongly "
            "associated with Alzheimer's disease in this uploaded scan. "
            f"The confidence score was {confidence}."
        )
    if result == "AD":
        return (
            "The brain scan analysis was classified as compatible with an "
            "Alzheimer's disease profile. This does not confirm Alzheimer's disease. "
            f"The confidence score was {confidence}."
        )
    return (
        "The brain scan analysis produced a decision-support result of "
        f"{display_value(result)} with a confidence score of {confidence}."
    )


def safe_speech_section(speech: dict[str, Any] | None) -> str:
    if not speech:
        return "Speech analysis was not available for this patient case."

    result = display_value(speech.get("result"))
    confidence = percent(speech.get("confidence"))
    patterns = speech_pattern_summary(speech)
    return (
        "The speech and language analysis was based on the uploaded picture "
        f"description transcript. The result was {result} with a confidence "
        f"score of {confidence}. The extracted speech patterns show {patterns}."
    )


def safe_combined_interpretation(brain: dict[str, Any] | None, speech: dict[str, Any] | None) -> str:
    if brain and speech:
        brain_result = str(brain.get("result") or "").strip()
        speech_result = str(speech.get("result") or "").strip()
        patterns = speech_pattern_summary(speech)
        if brain_result == "CN" and speech_result == "Control":
            return (
                "The brain scan result and the speech result both point toward a "
                "low-concern profile in this case. However, the two analyses are "
                "independent and should not be combined into a final diagnosis. "
                f"The speech sample still shows some patterns such as {patterns}, "
                "which may be useful for clinical review."
            )
        if brain_result == "AD" and speech_result == "ProbableAD":
            return (
                "Both independent outputs show patterns that may require closer "
                "clinical review. They do not confirm Alzheimer's disease and "
                "should not be combined into a final diagnosis."
            )
        return (
            "The two independent outputs do not point in exactly the same direction. "
            "They should be reviewed separately with the full clinical context and "
            "should not be combined into a final diagnosis."
        )
    if brain:
        return (
            "Only the brain scan analysis was available for this patient case. "
            "The speech and language result is missing, so this is a partial "
            "decision-support summary."
        )
    if speech:
        return (
            "Only the speech and language analysis was available for this patient "
            "case. The brain scan result is missing, so this is a partial "
            "decision-support summary."
        )
    return "No analysis output was available for combined interpretation."


def combined_summary_sections(case_data: dict) -> list[tuple[str, str]]:
    brain = case_data.get("brain_analysis")
    speech = case_data.get("speech_analysis")
    return [
        ("Brain Scan Analysis", safe_brain_section(brain)),
        ("Speech and Language Analysis", safe_speech_section(speech)),
        ("Combined Interpretation", safe_combined_interpretation(brain, speech)),
        (
            "Recommended Next Step",
            "Review these independent analysis outputs with a healthcare professional, "
            "together with the patient's clinical history, examination, and any "
            "additional tests considered appropriate.",
        ),
        ("Medical Note", f"{INDEPENDENCE_NOTE} {MEDICAL_NOTE}"),
    ]


def fallback_combined_summary(case_data: dict) -> str:
    return "\n\n".join(f"{title}\n{text}" for title, text in combined_summary_sections(case_data))


def safe_generate_with_llm(prompt: str) -> str:
    try:
        llm_generator = importlib.import_module("llm_generator")
        answer = llm_generator.generate_with_llm(prompt)
        return clean_text(answer)
    except Exception:
        return ""


def retrieve_information_context(case_data: dict) -> str:
    speech = case_data.get("speech_analysis") or {}
    transcript = str(speech.get("cleaned_transcript") or "").strip()
    prediction = str(speech.get("result") or "").strip()
    if not transcript or not prediction:
        return ""

    try:
        rag_explainer = importlib.import_module("rag_explainer")
        vectorstore = rag_explainer.load_vectorstore()
        context, _docs = rag_explainer.retrieve_context(
            transcript=transcript,
            prediction=prediction,
            vectorstore=vectorstore,
            k=3,
        )
        return clean_text(context)
    except Exception:
        return ""


def generate_combined_rag_summary(case_data: dict) -> str:
    return fallback_combined_summary(case_data)


def md_bullets(rows: list[tuple[str, Any]]) -> str:
    return "\n".join(f"- **{label}:** {display_value(value)}" for label, value in rows)


def transcript_excerpt(text: str, max_chars: int = 900) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned or "No cleaned transcript was available."
    return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."


def build_markdown_report(case_data: dict, rag_summary: str) -> str:
    patient = case_data.get("patient") or {}
    brain = case_data.get("brain_analysis")
    speech = case_data.get("speech_analysis")
    visual_path = relative_report_path((brain or {}).get("visual_explanation_path"))
    summary_markdown = "\n\n".join(
        f"### {title}\n\n{text}"
        for title, text in combined_summary_sections(case_data)
    )

    patient_rows = [
        ("Patient Case ID", case_data.get("patient_case_id")),
        ("Combined Analysis ID", case_data.get("combined_analysis_id")),
        ("Report Date", case_data.get("created_at")),
        ("Patient Name", patient.get("patient_name")),
        ("Study Date", patient.get("study_date")),
        ("Responsible Clinician", patient.get("responsible_clinician")),
        ("Clinical Notes", patient.get("clinical_notes")),
    ]

    brain_rows = []
    if brain:
        brain_rows = [
            ("Uploaded brain scan file", Path(str(brain.get("uploaded_file_name") or "Not provided")).name),
            ("Result", brain.get("result")),
            ("Confidence", percent(brain.get("confidence"))),
            ("Probability Cognitively Normal profile", percent(brain.get("prob_cn"))),
            ("Probability Alzheimer's disease-compatible profile", percent(brain.get("prob_ad"))),
            ("Simple explanation", brain.get("simple_explanation")),
        ]

    speech_rows = []
    if speech:
        speech_rows = [
            ("Uploaded speech file", Path(str(speech.get("uploaded_file_name") or "Not provided")).name),
            ("Result", speech.get("result")),
            ("Confidence", percent(speech.get("confidence"))),
            ("Simple explanation", speech.get("simple_explanation")),
        ]

    visual_section = "Visual explanation image was not available for this patient case."
    if visual_path:
        visual_section = (
            "Brain scan visual explanation - axial, sagittal and coronal views\n\n"
            f"![Brain scan visual explanation]({visual_path})"
        )

    report = f"""# Combined Multimodal Summary Report

## Patient Information

{md_bullets(patient_rows)}

## Brain Scan Analysis Summary

{md_bullets(brain_rows) if brain else "Brain scan analysis was not available for this patient case."}

## Speech and Language Analysis Summary

{md_bullets(speech_rows) if speech else "Speech analysis was not available for this patient case."}
"""
    if speech:
        feature_rows = friendly_feature_rows(speech.get("extracted_features"))
        if feature_rows:
            report += f"""
### Automatically Extracted Speech Information

{md_bullets(feature_rows)}
"""
        report += f"""
### Cleaned Transcript Excerpt

{transcript_excerpt(speech.get("cleaned_transcript") or "")}
"""

    report += f"""
## Combined Decision-Support Explanation

{summary_markdown}

## Visual Explanation

{visual_section}

## Information Sources

This combined summary uses the saved brain scan analysis, saved speech and language analysis, and patient information linked to the same patient case.

## Important Medical Note

{INDEPENDENCE_NOTE}

{MEDICAL_NOTE}
"""
    return clean_text_preserving_markdown(report)


def clean_text_preserving_markdown(text: str) -> str:
    lines = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("!["):
            line = re.sub(r"[A-Za-z]:\\[^\)]+", "", line)
            lines.append(line)
            continue
        heading_match = re.match(r"^(#{1,6}\s+)(.*)$", line.strip())
        if heading_match:
            lines.append(heading_match.group(1) + clean_text(heading_match.group(2)))
            continue
        lines.append(clean_text(line) if line.strip() else "")
    return "\n".join(lines).strip() + "\n"


def html_meta_item(label: str, value: Any) -> str:
    return (
        '<div class="meta-item">'
        f'<div class="meta-label">{html.escape(label)}</div>'
        f'<div class="meta-value">{html.escape(display_value(value))}</div>'
        "</div>"
    )


def html_card(title: str, body: str) -> str:
    return f'<section class="card"><h2>{html.escape(title)}</h2>{body}</section>'


def html_table(rows: list[tuple[str, Any]]) -> str:
    if not rows:
        return "<p>Not available.</p>"
    body = "".join(
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        f"<td>{html.escape(display_value(value))}</td>"
        "</tr>"
        for label, value in rows
    )
    return f'<table class="info-table">{body}</table>'


def build_html_report(case_data: dict, rag_summary: str) -> str:
    patient = case_data.get("patient") or {}
    brain = case_data.get("brain_analysis")
    speech = case_data.get("speech_analysis")
    visual_uri = image_data_uri((brain or {}).get("visual_explanation_path"))
    summary_html = "".join(
        f"<h3>{html.escape(title)}</h3><p>{html.escape(text)}</p>"
        for title, text in combined_summary_sections(case_data)
    )

    patient_rows = [
        ("Patient Case ID", case_data.get("patient_case_id")),
        ("Combined Analysis ID", case_data.get("combined_analysis_id")),
        ("Report Date", case_data.get("created_at")),
        ("Patient Name", patient.get("patient_name")),
        ("Study Date", patient.get("study_date")),
        ("Responsible Clinician", patient.get("responsible_clinician")),
        ("Clinical Notes", patient.get("clinical_notes")),
    ]
    brain_rows = (
        [
            ("Uploaded brain scan file", Path(str(brain.get("uploaded_file_name") or "Not provided")).name),
            ("Result", brain.get("result")),
            ("Confidence", percent(brain.get("confidence"))),
            ("Probability Cognitively Normal profile", percent(brain.get("prob_cn"))),
            ("Probability Alzheimer's disease-compatible profile", percent(brain.get("prob_ad"))),
            ("Simple explanation", brain.get("simple_explanation")),
        ]
        if brain
        else []
    )
    speech_rows = (
        [
            ("Uploaded speech file", Path(str(speech.get("uploaded_file_name") or "Not provided")).name),
            ("Result", speech.get("result")),
            ("Confidence", percent(speech.get("confidence"))),
            ("Simple explanation", speech.get("simple_explanation")),
        ]
        if speech
        else []
    )

    visual_body = "<p>Visual explanation image was not available for this patient case.</p>"
    if visual_uri:
        visual_body = (
            f'<img class="visual-img" src="{visual_uri}" alt="Brain scan visual explanation">'
            '<p class="caption">Brain scan visual explanation - axial, sagittal and coronal views</p>'
        )

    feature_body = ""
    if speech:
        feature_rows = friendly_feature_rows(speech.get("extracted_features"))
        if feature_rows:
            feature_body = "<h3>Automatically Extracted Speech Information</h3>" + html_table(feature_rows)
        feature_body += (
            "<h3>Cleaned Transcript Excerpt</h3>"
            f"<p>{html.escape(transcript_excerpt(speech.get('cleaned_transcript') or ''))}</p>"
        )

    body = (
        '<section class="card header">'
        '<div class="eyebrow">Combined Decision-Support Summary</div>'
        '<h1>Combined Multimodal Summary Report</h1>'
        '<p>This combined report summarizes the latest brain scan analysis and '
        'speech analysis for this patient case.</p>'
        '<div class="meta-grid">'
        + html_meta_item("Patient Case ID", case_data.get("patient_case_id"))
        + html_meta_item("Report Date", case_data.get("created_at"))
        + html_meta_item("Report Type", "Combined Multimodal Summary")
        + html_meta_item("Clinical Review", "Recommended")
        + "</div></section>"
        + html_card("Patient Information", html_table(patient_rows))
        + html_card(
            "Brain Scan Analysis Summary",
            html_table(brain_rows)
            if brain
            else "<p>Brain scan analysis was not available for this patient case.</p>",
        )
        + html_card(
            "Speech and Language Analysis Summary",
            (
                html_table(speech_rows)
                + feature_body
            )
            if speech
            else "<p>Speech analysis was not available for this patient case.</p>",
        )
        + html_card(
            "Combined Decision-Support Explanation",
            summary_html,
        )
        + html_card("Visual Explanation", visual_body)
        + html_card(
            "Information Sources",
            "<p>This report uses saved analysis outputs and patient information linked "
            "to the same patient case.</p>",
        )
        + f'<section class="card warning"><p>{html.escape(INDEPENDENCE_NOTE)}</p>'
        f'<p>{html.escape(MEDICAL_NOTE)}</p></section>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Combined Multimodal Summary Report</title>
<style>
  :root {{
    --bg1:#F8F4EC; --bg2:#E7F6EF; --bg3:#F3E8FF;
    --card:#FFFDF8; --primary:#0F9D7A; --deep:#096B5A;
    --violet:#8B5CF6; --gold:#C99A2E; --text:#202124;
    --muted:#6B7280; --line:#ECE7DB; --warn:#FFF7E2;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:32px 16px 64px;
    background:
      radial-gradient(900px 480px at 88% -6%, rgba(139,92,246,.12), transparent 60%),
      radial-gradient(820px 460px at -6% 2%, rgba(15,157,122,.12), transparent 58%),
      linear-gradient(135deg, var(--bg1), var(--bg2) 55%, var(--bg3));
    color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    line-height:1.62;
  }}
  .page {{ max-width:920px; margin:0 auto; }}
  .card {{
    background:rgba(255,253,248,.94); border:1px solid var(--line);
    border-radius:20px; padding:24px 28px; margin:16px 0;
    box-shadow:0 18px 44px -30px rgba(20,30,26,.55);
  }}
  .header {{ background:linear-gradient(135deg,#fff,#eefaf4 55%,#f3ecff); border-left:6px solid var(--primary); }}
  .eyebrow {{ font-size:12px; font-weight:800; letter-spacing:.18em; text-transform:uppercase; color:var(--deep); }}
  h1 {{ margin:4px 0 8px; font-size:28px; }}
  h2 {{ margin:0 0 14px; font-size:18px; color:var(--deep); }}
  h3 {{ margin:18px 0 8px; font-size:15px; }}
  .meta-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:16px; }}
  .meta-item {{ background:#fff; border:1px solid var(--line); border-radius:13px; padding:11px 14px; }}
  .meta-label {{ font-size:11px; font-weight:800; text-transform:uppercase; color:var(--muted); }}
  .meta-value {{ margin-top:3px; font-weight:700; word-break:break-word; }}
  .info-table {{ width:100%; border-collapse:collapse; }}
  .info-table th,.info-table td {{ border:1px solid var(--line); padding:10px 12px; text-align:left; vertical-align:top; }}
  .info-table th {{ width:34%; background:#F5F2EA; color:var(--muted); }}
  .visual-img {{ width:100%; border-radius:16px; border:1px solid var(--line); display:block; }}
  .caption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
  .warning {{ background:var(--warn); border-left:6px solid var(--gold); }}
</style>
</head>
<body><main class="page">{body}</main></body>
</html>
"""


def pdf_paragraph_text(text: Any) -> str:
    return html.escape(display_value(text)).replace("\n", "<br/>")


def build_pdf_report(case_data: dict, rag_summary: str, pdf_path: Path) -> None:
    colors = importlib.import_module("reportlab.lib.colors")
    pagesizes = importlib.import_module("reportlab.lib.pagesizes")
    styles_module = importlib.import_module("reportlab.lib.styles")
    units = importlib.import_module("reportlab.lib.units")
    platypus = importlib.import_module("reportlab.platypus")

    A4 = pagesizes.A4
    cm = units.cm
    Paragraph = platypus.Paragraph
    Spacer = platypus.Spacer
    Table = platypus.Table
    TableStyle = platypus.TableStyle
    SimpleDocTemplate = platypus.SimpleDocTemplate
    ReportImage = platypus.Image
    PageBreak = platypus.PageBreak

    PRIMARY = colors.HexColor("#0F9D7A")
    DEEP = colors.HexColor("#096B5A")
    GOLD = colors.HexColor("#C99A2E")
    CARD = colors.HexColor("#FFFDF8")
    LINE = colors.HexColor("#ECE7DB")
    MUTED = colors.HexColor("#6B7280")
    WARN = colors.HexColor("#FFF7E2")

    styles = styles_module.getSampleStyleSheet()
    title_style = styles["Title"].clone("CombinedTitle")
    title_style.fontName = "Helvetica-Bold"
    title_style.fontSize = 20
    title_style.leading = 24
    title_style.textColor = colors.HexColor("#16342d")

    section_style = styles["Heading2"].clone("CombinedSection")
    section_style.fontName = "Helvetica-Bold"
    section_style.fontSize = 12
    section_style.leading = 15
    section_style.textColor = DEEP

    body_style = styles["BodyText"].clone("CombinedBody")
    body_style.fontName = "Helvetica"
    body_style.fontSize = 9.4
    body_style.leading = 13
    body_style.textColor = colors.HexColor("#1f2937")

    small_style = styles["BodyText"].clone("CombinedSmall")
    small_style.fontName = "Helvetica"
    small_style.fontSize = 8.3
    small_style.leading = 10
    small_style.textColor = MUTED

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.65 * cm,
        title="Combined Multimodal Summary Report",
    )
    page_width = A4[0] - doc.leftMargin - doc.rightMargin

    def p(text: Any, style=body_style):
        return Paragraph(pdf_paragraph_text(text), style)

    def section(title: str):
        story.append(Spacer(1, 10))
        story.append(Paragraph(html.escape(title), section_style))
        story.append(Spacer(1, 6))

    def table(rows: list[tuple[str, Any]], label_width: float = 0.38):
        data = [[p(label, body_style), p(value, body_style)] for label, value in rows]
        tbl = Table(
            data,
            colWidths=[page_width * label_width, page_width * (1 - label_width)],
            hAlign="LEFT",
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F2EA")),
                    ("BACKGROUND", (1, 0), (1, -1), CARD),
                    ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                    ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return tbl

    def note_box(text: str):
        tbl = Table([[p(text)]], colWidths=[page_width], hAlign="LEFT")
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), WARN),
                    ("BOX", (0, 0), (-1, -1), 0.8, GOLD),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, GOLD),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            )
        )
        return tbl

    def image_flowable(path_value: str | Path | None):
        path = resolve_project_path(path_value)
        if not path or not path.exists():
            return None
        try:
            pil_module = importlib.import_module("PIL.Image")
            with pil_module.open(path) as img:
                width, height = img.size
            ratio = height / width
            width_pt = page_width
            height_pt = width_pt * ratio
            max_height = 10.5 * cm
            if height_pt > max_height:
                height_pt = max_height
                width_pt = height_pt / ratio
            return ReportImage(str(path), width=width_pt, height=height_pt)
        except Exception:
            return None

    patient = case_data.get("patient") or {}
    brain = case_data.get("brain_analysis")
    speech = case_data.get("speech_analysis")

    header_rows = [
        ("Patient Case ID", case_data.get("patient_case_id")),
        ("Date", case_data.get("created_at")),
        ("Report type", "Combined Multimodal Summary"),
        ("Combined Analysis ID", case_data.get("combined_analysis_id")),
    ]
    patient_rows = [
        ("Patient Name", patient.get("patient_name")),
        ("Study Date", patient.get("study_date")),
        ("Responsible Clinician", patient.get("responsible_clinician")),
        ("Clinical Notes", patient.get("clinical_notes")),
    ]
    brain_rows = (
        [
            ("Uploaded brain scan file", Path(str(brain.get("uploaded_file_name") or "Not provided")).name),
            ("Result", brain.get("result")),
            ("Confidence", percent(brain.get("confidence"))),
            ("Probability Cognitively Normal profile", percent(brain.get("prob_cn"))),
            ("Probability Alzheimer's disease-compatible profile", percent(brain.get("prob_ad"))),
            ("Simple explanation", brain.get("simple_explanation")),
        ]
        if brain
        else []
    )
    speech_rows = (
        [
            ("Uploaded speech file", Path(str(speech.get("uploaded_file_name") or "Not provided")).name),
            ("Result", speech.get("result")),
            ("Confidence", percent(speech.get("confidence"))),
            ("Simple explanation", speech.get("simple_explanation")),
        ]
        if speech
        else []
    )

    story = [
        Table(
            [
                [Paragraph("Combined Decision-Support Summary", small_style)],
                [Paragraph("Combined Multimodal Summary Report", title_style)],
                [table(header_rows, label_width=0.32)],
            ],
            colWidths=[page_width],
            hAlign="LEFT",
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F6EF")),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B9DFCF")),
                    ("LINEBEFORE", (0, 0), (0, -1), 4.0, PRIMARY),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            ),
        ),
        Spacer(1, 12),
    ]

    section("Patient Information")
    story.append(table(patient_rows))

    section("Brain Scan Analysis Summary")
    if brain:
        story.append(table(brain_rows, label_width=0.46))
    else:
        story.append(p("Brain scan analysis was not available for this patient case."))

    section("Speech and Language Analysis Summary")
    if speech:
        story.append(table(speech_rows, label_width=0.42))
        feature_rows = friendly_feature_rows(speech.get("extracted_features"))
        if feature_rows:
            story.append(Spacer(1, 8))
            story.append(Paragraph("Automatically Extracted Speech Information", section_style))
            story.append(table(feature_rows, label_width=0.42))
        story.append(Spacer(1, 8))
        story.append(Paragraph("Cleaned Transcript Excerpt", section_style))
        story.append(p(transcript_excerpt(speech.get("cleaned_transcript") or "")))
    else:
        story.append(p("Speech analysis was not available for this patient case."))

    if brain and brain.get("visual_explanation_path"):
        story.append(PageBreak())
        section("Visual Explanation")
        image = image_flowable(brain.get("visual_explanation_path"))
        if image:
            story.append(image)
            story.append(p("Brain scan visual explanation - axial, sagittal and coronal views", small_style))
        else:
            story.append(p("Visual explanation image was not available for this patient case."))

    for title, text in combined_summary_sections(case_data):
        section(title)
        if title == "Medical Note":
            story.append(note_box(text))
        else:
            story.append(p(text))

    section("Information Sources")
    story.append(
        p(
            "This report uses saved analysis outputs and patient information linked "
            "to the same patient case."
        )
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(document.leftMargin, 0.85 * cm, "Decision-support summary - not a medical diagnosis")
        canvas.drawRightString(A4[0] - document.rightMargin, 0.85 * cm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def save_combined_multimodal_report(case_data: dict, rag_summary: str) -> dict:
    ensure_output_dirs()
    patient_case_id = str(case_data.get("patient_case_id") or "").strip()
    if not patient_case_id:
        raise ValueError("patient_case_id is required.")

    analysis_id = str(case_data.get("combined_analysis_id") or "").strip()
    if not analysis_id:
        analysis_id = generate_combined_analysis_id()
        case_data["combined_analysis_id"] = analysis_id

    base = (
        f"combined_multimodal_report_"
        f"{report_filename_part(patient_case_id, 'patient')}_"
        f"{report_filename_part(analysis_id, 'analysis')}"
    )
    md_path = REPORTS_DIR / f"{base}.md"
    html_path = REPORTS_DIR / f"{base}.html"
    pdf_path = REPORTS_DIR / f"{base}.pdf"

    md_text = build_markdown_report(case_data, rag_summary)
    html_text = build_html_report(case_data, rag_summary)

    md_path.write_text(md_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    try:
        build_pdf_report(case_data, rag_summary, pdf_path)
    except Exception:
        pdf_path = None

    paths = {"md": md_path, "html": html_path, "pdf": pdf_path}
    save_report(
        patient_case_id=patient_case_id,
        analysis_id=analysis_id,
        analysis_type="Combined Multimodal Summary",
        report_md_path=md_path,
        report_html_path=html_path,
        report_pdf_path=pdf_path,
    )
    return paths
