from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except Exception as exc:  # pragma: no cover - keeps Streamlit import safe.
    load_dotenv = None
    create_engine = None
    text = None
    SQLAlchemyError = Exception
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
_ENGINE: Any | None = None
_LAST_ERROR: str = ""


class DatabaseUnavailable(RuntimeError):
    """Raised only when explicit database initialization is requested."""


def _load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_DIR / ".env")


def database_error() -> str:
    return _LAST_ERROR


def is_database_available() -> bool:
    try:
        engine = get_engine()
        if engine is None or text is None:
            return False
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_engine() -> Any | None:
    global _ENGINE, _LAST_ERROR

    if _IMPORT_ERROR is not None:
        _LAST_ERROR = "Database dependencies are not installed."
        return None

    if _ENGINE is not None:
        return _ENGINE

    _load_environment()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        _LAST_ERROR = "DATABASE_URL is not configured."
        return None

    try:
        _ENGINE = create_engine(database_url, pool_pre_ping=True, future=True)
        return _ENGINE
    except Exception:
        _LAST_ERROR = "Database engine could not be created."
        return None


def to_relative_path(path: str | Path | None) -> str | None:
    if path is None:
        return None

    value = str(path).strip()
    if not value:
        return None

    path_obj = Path(value)
    try:
        return path_obj.resolve().relative_to(PROJECT_DIR).as_posix()
    except Exception:
        return path_obj.as_posix()


def _as_json_text(value: Any) -> str:
    if value is None:
        value = {}
    return json.dumps(value, ensure_ascii=False, default=str)


def _fetch_dicts(result) -> list[dict[str, Any]]:
    return [dict(row) for row in result.mappings().all()]


def init_db() -> bool:
    global _LAST_ERROR

    engine = get_engine()
    if engine is None or text is None:
        raise DatabaseUnavailable(_LAST_ERROR or "Database is not connected.")

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Database schema file not found: {SCHEMA_PATH}")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    try:
        with engine.begin() as connection:
            connection.execute(text(schema_sql))
        _LAST_ERROR = ""
        return True
    except Exception as exc:
        _LAST_ERROR = "Database schema initialization failed."
        raise DatabaseUnavailable(_LAST_ERROR) from exc


def save_patient(
    patient_case_id: str,
    patient_name: str | None = None,
    study_date: str | None = None,
    responsible_clinician: str | None = None,
    clinical_notes: str | None = None,
) -> bool:
    engine = get_engine()
    if engine is None or text is None:
        return False

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO patients (
                        patient_case_id,
                        patient_name,
                        study_date,
                        responsible_clinician,
                        clinical_notes
                    )
                    VALUES (
                        :patient_case_id,
                        :patient_name,
                        :study_date,
                        :responsible_clinician,
                        :clinical_notes
                    )
                    ON CONFLICT (patient_case_id)
                    DO UPDATE SET
                        patient_name = EXCLUDED.patient_name,
                        study_date = EXCLUDED.study_date,
                        responsible_clinician = EXCLUDED.responsible_clinician,
                        clinical_notes = EXCLUDED.clinical_notes,
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "patient_case_id": patient_case_id,
                    "patient_name": patient_name,
                    "study_date": study_date,
                    "responsible_clinician": responsible_clinician,
                    "clinical_notes": clinical_notes,
                },
            )
        return True
    except SQLAlchemyError:
        return False


def save_speech_analysis(
    patient_case_id: str,
    analysis_id: str,
    uploaded_file_name: str | None = None,
    speech_file_path: str | Path | None = None,
    cleaned_transcript: str | None = None,
    extracted_features: dict[str, Any] | None = None,
    result: str | None = None,
    confidence: float | None = None,
    simple_explanation: str | None = None,
) -> bool:
    engine = get_engine()
    if engine is None or text is None:
        return False

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO speech_analyses (
                        patient_case_id,
                        analysis_id,
                        uploaded_file_name,
                        speech_file_path,
                        cleaned_transcript,
                        extracted_features,
                        result,
                        confidence,
                        simple_explanation
                    )
                    VALUES (
                        :patient_case_id,
                        :analysis_id,
                        :uploaded_file_name,
                        :speech_file_path,
                        :cleaned_transcript,
                        CAST(:extracted_features AS JSONB),
                        :result,
                        :confidence,
                        :simple_explanation
                    )
                    ON CONFLICT (analysis_id)
                    DO UPDATE SET
                        uploaded_file_name = EXCLUDED.uploaded_file_name,
                        speech_file_path = EXCLUDED.speech_file_path,
                        cleaned_transcript = EXCLUDED.cleaned_transcript,
                        extracted_features = EXCLUDED.extracted_features,
                        result = EXCLUDED.result,
                        confidence = EXCLUDED.confidence,
                        simple_explanation = EXCLUDED.simple_explanation
                    """
                ),
                {
                    "patient_case_id": patient_case_id,
                    "analysis_id": analysis_id,
                    "uploaded_file_name": Path(str(uploaded_file_name)).name
                    if uploaded_file_name else None,
                    "speech_file_path": to_relative_path(speech_file_path),
                    "cleaned_transcript": cleaned_transcript,
                    "extracted_features": _as_json_text(extracted_features),
                    "result": result,
                    "confidence": confidence,
                    "simple_explanation": simple_explanation,
                },
            )
        return True
    except SQLAlchemyError:
        return False


def save_brain_analysis(
    patient_case_id: str,
    analysis_id: str,
    uploaded_file_name: str | None = None,
    mri_file_path: str | Path | None = None,
    result: str | None = None,
    confidence: float | None = None,
    prob_cn: float | None = None,
    prob_ad: float | None = None,
    visual_explanation_path: str | Path | None = None,
    simple_explanation: str | None = None,
) -> bool:
    engine = get_engine()
    if engine is None or text is None:
        return False

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO brain_analyses (
                        patient_case_id,
                        analysis_id,
                        uploaded_file_name,
                        mri_file_path,
                        result,
                        confidence,
                        prob_cn,
                        prob_ad,
                        visual_explanation_path,
                        simple_explanation
                    )
                    VALUES (
                        :patient_case_id,
                        :analysis_id,
                        :uploaded_file_name,
                        :mri_file_path,
                        :result,
                        :confidence,
                        :prob_cn,
                        :prob_ad,
                        :visual_explanation_path,
                        :simple_explanation
                    )
                    ON CONFLICT (analysis_id)
                    DO UPDATE SET
                        uploaded_file_name = EXCLUDED.uploaded_file_name,
                        mri_file_path = EXCLUDED.mri_file_path,
                        result = EXCLUDED.result,
                        confidence = EXCLUDED.confidence,
                        prob_cn = EXCLUDED.prob_cn,
                        prob_ad = EXCLUDED.prob_ad,
                        visual_explanation_path = EXCLUDED.visual_explanation_path,
                        simple_explanation = EXCLUDED.simple_explanation
                    """
                ),
                {
                    "patient_case_id": patient_case_id,
                    "analysis_id": analysis_id,
                    "uploaded_file_name": Path(str(uploaded_file_name)).name
                    if uploaded_file_name else None,
                    "mri_file_path": to_relative_path(mri_file_path),
                    "result": result,
                    "confidence": confidence,
                    "prob_cn": prob_cn,
                    "prob_ad": prob_ad,
                    "visual_explanation_path": to_relative_path(visual_explanation_path),
                    "simple_explanation": simple_explanation,
                },
            )
        return True
    except SQLAlchemyError:
        return False


def save_report(
    patient_case_id: str,
    analysis_id: str,
    analysis_type: str,
    report_md_path: str | Path | None = None,
    report_html_path: str | Path | None = None,
    report_pdf_path: str | Path | None = None,
) -> bool:
    engine = get_engine()
    if engine is None or text is None:
        return False

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO reports (
                        patient_case_id,
                        analysis_id,
                        analysis_type,
                        report_md_path,
                        report_html_path,
                        report_pdf_path
                    )
                    VALUES (
                        :patient_case_id,
                        :analysis_id,
                        :analysis_type,
                        :report_md_path,
                        :report_html_path,
                        :report_pdf_path
                    )
                    """
                ),
                {
                    "patient_case_id": patient_case_id,
                    "analysis_id": analysis_id,
                    "analysis_type": analysis_type,
                    "report_md_path": to_relative_path(report_md_path),
                    "report_html_path": to_relative_path(report_html_path),
                    "report_pdf_path": to_relative_path(report_pdf_path),
                },
            )
        return True
    except SQLAlchemyError:
        return False


def get_patient_history(patient_case_id: str) -> list[dict[str, Any]]:
    engine = get_engine()
    if engine is None or text is None:
        return []

    query = text(
        """
        SELECT
            'Speech and Language Analysis' AS analysis_type,
            analysis_id,
            result,
            confidence,
            created_at
        FROM speech_analyses
        WHERE patient_case_id = :patient_case_id
        UNION ALL
        SELECT
            'Brain Scan Analysis' AS analysis_type,
            analysis_id,
            result,
            confidence,
            created_at
        FROM brain_analyses
        WHERE patient_case_id = :patient_case_id
        ORDER BY created_at DESC
        """
    )

    try:
        with engine.connect() as connection:
            return _fetch_dicts(connection.execute(query, {"patient_case_id": patient_case_id}))
    except SQLAlchemyError:
        return []


def get_latest_speech_analysis(patient_case_id: str) -> dict[str, Any] | None:
    engine = get_engine()
    if engine is None or text is None:
        return None

    try:
        with engine.connect() as connection:
            rows = _fetch_dicts(
                connection.execute(
                    text(
                        """
                        SELECT *
                        FROM speech_analyses
                        WHERE patient_case_id = :patient_case_id
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"patient_case_id": patient_case_id},
                )
            )
        return rows[0] if rows else None
    except SQLAlchemyError:
        return None


def get_latest_brain_analysis(patient_case_id: str) -> dict[str, Any] | None:
    engine = get_engine()
    if engine is None or text is None:
        return None

    try:
        with engine.connect() as connection:
            rows = _fetch_dicts(
                connection.execute(
                    text(
                        """
                        SELECT *
                        FROM brain_analyses
                        WHERE patient_case_id = :patient_case_id
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"patient_case_id": patient_case_id},
                )
            )
        return rows[0] if rows else None
    except SQLAlchemyError:
        return None


def get_reports(patient_case_id: str) -> list[dict[str, Any]]:
    engine = get_engine()
    if engine is None or text is None:
        return []

    try:
        with engine.connect() as connection:
            return _fetch_dicts(
                connection.execute(
                    text(
                        """
                        SELECT *
                        FROM reports
                        WHERE patient_case_id = :patient_case_id
                        ORDER BY created_at DESC
                        """
                    ),
                    {"patient_case_id": patient_case_id},
                )
            )
    except SQLAlchemyError:
        return []


def get_recent_patient_cases(limit: int = 20) -> list[dict[str, Any]]:
    engine = get_engine()
    if engine is None or text is None:
        return []

    try:
        safe_limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        safe_limit = 20

    query = text(
        """
        WITH brain AS (
            SELECT patient_case_id, MAX(created_at) AS latest_brain_at
            FROM brain_analyses
            GROUP BY patient_case_id
        ),
        speech AS (
            SELECT patient_case_id, MAX(created_at) AS latest_speech_at
            FROM speech_analyses
            GROUP BY patient_case_id
        ),
        combined AS (
            SELECT patient_case_id, MAX(created_at) AS latest_combined_at
            FROM reports
            WHERE analysis_type = 'Combined Multimodal Summary'
            GROUP BY patient_case_id
        ),
        report_activity AS (
            SELECT patient_case_id, MAX(created_at) AS latest_report_at
            FROM reports
            GROUP BY patient_case_id
        )
        SELECT
            p.patient_case_id,
            p.patient_name,
            GREATEST(
                COALESCE(p.updated_at, p.created_at),
                COALESCE(brain.latest_brain_at, p.created_at),
                COALESCE(speech.latest_speech_at, p.created_at),
                COALESCE(combined.latest_combined_at, p.created_at),
                COALESCE(report_activity.latest_report_at, p.created_at)
            ) AS latest_created_at,
            (brain.latest_brain_at IS NOT NULL) AS has_brain_analysis,
            (speech.latest_speech_at IS NOT NULL) AS has_speech_analysis,
            (combined.latest_combined_at IS NOT NULL) AS has_combined_report
        FROM patients p
        LEFT JOIN brain ON brain.patient_case_id = p.patient_case_id
        LEFT JOIN speech ON speech.patient_case_id = p.patient_case_id
        LEFT JOIN combined ON combined.patient_case_id = p.patient_case_id
        LEFT JOIN report_activity ON report_activity.patient_case_id = p.patient_case_id
        ORDER BY latest_created_at DESC
        LIMIT :limit
        """
    )

    try:
        with engine.connect() as connection:
            return _fetch_dicts(connection.execute(query, {"limit": safe_limit}))
    except SQLAlchemyError:
        return []
