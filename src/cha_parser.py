from __future__ import annotations

import re
from typing import Any


FEATURE_ORDER = [
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


def read_cha_file(uploaded_file) -> str:
    """
    Read an uploaded CHAT transcript file.

    The function accepts Streamlit uploaded files and simple file-like objects.
    It tries UTF-8 first and falls back to latin-1.
    """
    if uploaded_file is None:
        raise ValueError("No speech transcript file was uploaded.")

    try:
        if hasattr(uploaded_file, "getvalue"):
            data = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
        else:
            data = uploaded_file
    except Exception as exc:
        raise ValueError("The speech transcript file could not be read.") from exc

    if isinstance(data, str):
        return data

    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("The speech transcript file has an unsupported format.")

    if not data:
        raise ValueError("The speech transcript file is empty.")

    try:
        return bytes(data).decode("utf-8-sig")
    except UnicodeDecodeError:
        return bytes(data).decode("latin-1", errors="replace")


def extract_participant_transcript(cha_text: str) -> str:
    """
    Extract only patient speech from *PAR: lines.

    Interviewer lines, metadata lines, dependent tiers, and CHAT headers are
    ignored. Continuation lines immediately after a *PAR: line are included.
    """
    if not cha_text:
        return ""

    utterances: list[str] = []
    collecting_par = False

    for raw_line in str(cha_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("@") or line.startswith("%"):
            collecting_par = False
            continue

        speaker_match = re.match(r"^\*([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
        if speaker_match:
            speaker = speaker_match.group(1).upper()
            content = speaker_match.group(2).strip()
            collecting_par = speaker == "PAR"
            if collecting_par and content:
                utterances.append(content)
            continue

        if collecting_par and not line.startswith("*"):
            utterances.append(line)

    return " ".join(utterances).strip()


def clean_cha_transcript(raw_transcript: str) -> str:
    """
    Clean CHAT annotations from patient speech for display.

    Feature extraction should use the raw CHA text before this cleaning step.
    """
    text = str(raw_transcript or "")
    if not text.strip():
        return ""

    text = _remove_timestamps(text)
    text = re.sub(r"\b([A-Za-z]+)in\(g\)", r"\1ing", text)
    text = re.sub(r"\b([A-Za-z]+):([A-Za-z]+)\b", r"\1\2", text)
    text = re.sub(r"&-(um|uh|hm)\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:xxx|yyy|www)\b", " [unclear] ", text, flags=re.IGNORECASE)

    text = re.sub(r"\[\s*(?://|/|x\s*\d+)\s*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\((?:\s*\.+\s*|\s*\d+(?:\.\d+)?\s*)\)", " ", text)
    text = re.sub(r"&=(?:laugh|laughs|cough|sigh|noise)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"&\+[A-Za-z]+", " ", text)

    text = re.sub(r"\+//|\+/|\+\.\.\.", " ", text)
    text = re.sub(r"[\u2021\x15]", " ", text)
    text = re.sub(r"[<>#=_~^]", " ", text)
    text = re.sub(r"\b0\b", " ", text)

    text = re.sub(r"\s+([.,?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_speech_features_from_cha(cha_text: str) -> dict:
    """
    Extract the 9 speech features from raw CHA content before cleaning.

    Counts are computed on raw participant speech only so interviewer speech
    does not affect the patient features. Metadata extraction uses the full
    CHA text.
    """
    raw_patient_speech = extract_participant_transcript(cha_text)
    lower_speech = raw_patient_speech.lower()

    n_filled_pauses = len(
        re.findall(
            r"(?<!\w)&-(?:um|uh|hm)\b|\b(?:um|uh|hm|erm|er)\b",
            lower_speech,
        )
    )
    n_phon_fragments = len(re.findall(r"&\+[A-Za-z]+", raw_patient_speech))
    n_phon_fragments += len(re.findall(r"\b[A-Za-z]+-\b", raw_patient_speech))
    n_phon_fragments += len(re.findall(r"\b[A-Za-z]+-\s", raw_patient_speech))

    n_paralinguistic = len(
        re.findall(
            r"&=(?:laugh|laughs|cough|sigh|noise)\b",
            lower_speech,
        )
    )
    n_paralinguistic += len(re.findall(r"\[=![^\]]+\]", raw_patient_speech, flags=re.IGNORECASE))

    n_retracings = len(re.findall(r"\[\s*//\s*\]|\[\s*/\s*\]", raw_patient_speech))
    n_unintelligible = len(re.findall(r"\b(?:xxx|yyy|www)\b", lower_speech))
    n_pauses = len(
        re.findall(
            r"\((?:\s*\.+\s*|\s*\d+(?:\.\d+)?\s*)\)",
            raw_patient_speech,
        )
    )

    return {
        "n_filled_pauses": int(n_filled_pauses),
        "n_phon_fragments": int(n_phon_fragments),
        "n_paralinguistic": int(n_paralinguistic),
        "n_retracings": int(n_retracings),
        "n_unintelligible": int(n_unintelligible),
        "n_pauses": int(n_pauses),
        "entryage": _extract_age(cha_text),
        "sex": _extract_sex(cha_text),
        "educ": _extract_education(cha_text),
    }


def build_feature_vector_from_cha(features_dict: dict) -> list:
    """Return the feature vector in the exact order used by the NLP model."""
    return [features_dict.get(key) for key in FEATURE_ORDER]


def _remove_timestamps(text: str) -> str:
    text = re.sub(r"\x15[^\x15]*\x15", " ", text)
    return re.sub(r"\b\d+_\d+\b", " ", text)


def _metadata_value(cha_text: str, names: tuple[str, ...]) -> str:
    for name in names:
        match = re.search(rf"(?im)^@{re.escape(name)}\s*:\s*(.+)$", str(cha_text or ""))
        if match:
            return match.group(1).strip()
    return ""


def _id_rows(cha_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in str(cha_text or "").splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("@id:"):
            continue
        value = line.split(":", 1)[1].strip()
        rows.append([field.strip() for field in value.split("|")])

    par_rows = [
        row for row in rows
        if len(row) > 2 and row[2].strip().upper() == "PAR"
    ]
    return par_rows or rows


def _parse_age(value: Any) -> int | None:
    text = str(value or "")
    match = re.search(r"\b(\d{1,3})(?:[;.]\d+)?\b", text)
    if not match:
        return None
    age = int(match.group(1))
    return age if 0 < age <= 120 else None


def _parse_sex(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if text in {"male", "m"}:
        return 1
    if text in {"female", "f"}:
        return 0
    return None


def _parse_education(value: Any) -> int | None:
    text = str(value or "")
    match = re.search(r"\b(\d{1,2})(?:\.\d+)?\b", text)
    if not match:
        return None
    years = int(match.group(1))
    return years if 0 <= years <= 40 else None


def _extract_age(cha_text: str) -> int | None:
    age = _parse_age(_metadata_value(cha_text, ("Age", "EntryAge")))
    if age is not None:
        return age

    for row in _id_rows(cha_text):
        if len(row) > 3:
            age = _parse_age(row[3])
            if age is not None:
                return age
    return None


def _extract_sex(cha_text: str) -> int | None:
    sex = _parse_sex(_metadata_value(cha_text, ("Sex", "Gender")))
    if sex is not None:
        return sex

    for row in _id_rows(cha_text):
        for field in row:
            sex = _parse_sex(field)
            if sex is not None:
                return sex
    return None


def _extract_education(cha_text: str) -> int | None:
    educ = _parse_education(_metadata_value(cha_text, ("Education", "Educ")))
    if educ is not None:
        return educ

    for row in _id_rows(cha_text):
        for index in (8, 9, 7):
            if len(row) > index:
                educ = _parse_education(row[index])
                if educ is not None:
                    return educ
    return None
