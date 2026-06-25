CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    patient_case_id VARCHAR(100) UNIQUE NOT NULL,
    patient_name VARCHAR(255),
    study_date VARCHAR(50),
    responsible_clinician VARCHAR(255),
    clinical_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS speech_analyses (
    id SERIAL PRIMARY KEY,
    patient_case_id VARCHAR(100) REFERENCES patients(patient_case_id) ON DELETE CASCADE,
    analysis_id VARCHAR(100) UNIQUE,
    uploaded_file_name VARCHAR(255),
    speech_file_path TEXT,
    cleaned_transcript TEXT,
    extracted_features JSONB,
    result VARCHAR(100),
    confidence FLOAT,
    simple_explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS brain_analyses (
    id SERIAL PRIMARY KEY,
    patient_case_id VARCHAR(100) REFERENCES patients(patient_case_id) ON DELETE CASCADE,
    analysis_id VARCHAR(100) UNIQUE,
    uploaded_file_name VARCHAR(255),
    mri_file_path TEXT,
    result VARCHAR(100),
    confidence FLOAT,
    prob_cn FLOAT,
    prob_ad FLOAT,
    visual_explanation_path TEXT,
    simple_explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    patient_case_id VARCHAR(100) REFERENCES patients(patient_case_id) ON DELETE CASCADE,
    analysis_id VARCHAR(100),
    analysis_type VARCHAR(100),
    report_md_path TEXT,
    report_html_path TEXT,
    report_pdf_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION update_patients_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_patients_updated_at ON patients;

CREATE TRIGGER trg_patients_updated_at
BEFORE UPDATE ON patients
FOR EACH ROW
EXECUTE FUNCTION update_patients_updated_at();
