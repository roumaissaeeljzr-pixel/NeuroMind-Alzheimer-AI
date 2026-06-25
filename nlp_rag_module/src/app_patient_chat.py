import warnings
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st

from main_pipeline import run_full_pipeline
from rag_explainer import load_vectorstore
from llm_generator import generate_with_llm


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORT_TEMPLATE_VERSION = "english-llm-v1"


# ============================================================
# Helpers
# ============================================================

def clean_source_name(source):
    return Path(str(source)).name


def source_to_google_link(source):
    """
    Convert a local PDF source into a clean Google search link.
    Example:
    nlp_dementia_detection.pdf
    -> nlp dementia detection + Google search URL
    """
    source_name = Path(str(source)).name
    title = (
        Path(source_name).stem
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    google_url = "https://www.google.com/search?q=" + quote_plus(title + " PDF")

    return title, google_url


def source_markdown(source):
    title, url = source_to_google_link(source)
    return f"- [{title}]({url})"


def sources_markdown(sources):
    lines = []
    seen = set()

    for item in sources:
        source = item.get("source", "unknown") if isinstance(item, dict) else item
        title, _ = source_to_google_link(source)
        key = title.casefold()

        if title and key not in seen:
            lines.append(source_markdown(source))
            seen.add(key)

    if not lines:
        return "- No medical sources were retrieved."

    return "\n".join(lines)


def display_value(value):
    if value is None:
        return "Not provided"

    if hasattr(value, "strftime"):
        return value.strftime("%d/%m/%Y")

    text = str(value).strip()
    return text if text else "Not provided"


def build_patient_info(
    patient_name,
    patient_id,
    study_date,
    age,
    sex_label,
    education_years,
    clinician,
    clinical_notes,
):
    return {
        "Patient Name": display_value(patient_name),
        "Patient ID": display_value(patient_id),
        "Study Date": display_value(study_date),
        "Age": display_value(age),
        "Sex": display_value(sex_label),
        "Years of Education": display_value(education_years),
        "Responsible Clinician": display_value(clinician),
        "Clinical Notes": display_value(clinical_notes),
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


def dict_to_markdown(data):
    return "\n".join(f"- **{key}:** {value}" for key, value in data.items())


def patient_info_markdown(patient_info):
    return "## Patient Information\n\n" + dict_to_markdown(patient_info)


def analysis_features_markdown(analysis_features):
    return dict_to_markdown(analysis_features)


def prediction_display(prediction):
    if prediction == "ProbableAD":
        return "Profile compatible with probable Alzheimer's disease / dementia"

    if prediction == "Control":
        return "Control profile"

    return str(prediction)


def simple_interpretation(prediction):
    if prediction == "ProbableAD":
        return (
            "The language profile contains signs that may be compatible with "
            "cognitive impairment. This does not mean the patient has "
            "Alzheimer's disease; it only indicates that professional review "
            "is needed."
        )

    if prediction == "Control":
        return (
            "The language profile is closer to the control group learned by "
            "the model. This does not rule out medical concerns and should "
            "still be interpreted by a healthcare professional."
        )

    return (
        "The system produced an analysis result that should be reviewed by a "
        "healthcare professional."
    )


def medical_warning():
    return (
        "This result is not a final medical diagnosis. It is only a "
        "decision-support tool and must be interpreted by a healthcare "
        "professional."
    )


def safe_generate_with_llm(prompt):
    """
    Use the local Ollama LLM and return a clear message if it is unavailable.
    """
    try:
        return generate_with_llm(prompt)
    except Exception as e:
        return (
            "The local LLM did not respond. Please make sure Ollama is running "
            "and run:\n\n"
            "`ollama run gemma3:1b`\n\n"
            f"Error details: {e}"
        )


def save_patient_report(report_text):
    OUTPUT_DIR.mkdir(exist_ok=True)
    file_name = f"patient_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = OUTPUT_DIR / file_name
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


# ============================================================
# RAG retrieval
# ============================================================

def retrieve_chat_context(question, transcript, prediction, k=4):
    vectorstore = load_vectorstore()

    query = f"""
    Patient prediction: {prediction}

    Patient transcript:
    {transcript}

    User question:
    {question}

    Retrieve medical evidence about Alzheimer's disease, dementia,
    language impairment, pauses, repetitions, word-finding difficulties,
    cognitive decline, speech biomarkers, and NLP-based dementia detection.
    """

    candidate_docs = vectorstore.similarity_search(query, k=20)

    selected_docs = []
    used_sources = set()

    for doc in candidate_docs:
        source = doc.metadata.get("source", "unknown")

        if source not in used_sources:
            selected_docs.append(doc)
            used_sources.add(source)

        if len(selected_docs) == k:
            break

    if len(selected_docs) < k:
        for doc in candidate_docs:
            if doc not in selected_docs:
                selected_docs.append(doc)

            if len(selected_docs) == k:
                break

    return selected_docs


def prepare_context_and_sources(docs):
    context_parts = []
    source_items = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        source_name = clean_source_name(source)
        text = doc.page_content.replace("\n", " ")
        text = " ".join(text.split())

        context_parts.append(
            f"Document {i}\n"
            f"Source: {source_name}\n"
            f"Content: {text[:1200]}"
        )

        source_items.append(source)

    context = "\n\n".join(context_parts)
    source_links = sources_markdown(source_items)

    return context, source_links


# ============================================================
# RAG chat with local LLM
# ============================================================

def build_chat_answer(question, transcript, prediction, confidence):
    docs = retrieve_chat_context(
        question=question,
        transcript=transcript,
        prediction=prediction,
        k=4
    )

    context, source_links = prepare_context_and_sources(docs)

    prompt = f"""
You are a patient-friendly medical explanation assistant.

Rules:
- Do not provide a final medical diagnosis.
- Use simple patient-friendly language.
- Answer in English.
- Use only the retrieved medical context.
- Do not invent information.
- Do not mention FAISS, embeddings, chunks, or technical implementation details.
- Mention that the result must be interpreted by a healthcare professional.
- The transcript comes from a Cookie Theft picture-description task. The image
  content is only the task used to collect speech.
- Do not interpret picture content as a medical symptom.
- Do not say that cookies, a boy, a girl, a mother, water, a sink, laughter,
  or any object/action in the picture is a disorder or medical evidence.
- Focus only on linguistic signs such as filled pauses, hesitations,
  repetitions, corrections, phonological fragments, unintelligible words,
  pauses, reduced information content, discourse organization, and
  word-finding difficulties.
- Do not invent psychological or emotional interpretations.

Patient transcript:
{transcript}

Model result:
{prediction}

Confidence score:
{confidence * 100:.1f}%

User question:
{question}

Retrieved medical context:
{context}

Answer the question clearly and simply.
"""

    answer = safe_generate_with_llm(prompt).strip()
    answer += "\n\n**Sources Consulted:**\n"
    answer += source_links

    return answer


# ============================================================
# Patient report with local LLM
# ============================================================

def generate_patient_report(transcript, result, patient_info, analysis_features):
    prediction = result["prediction"]
    confidence = result["confidence"]
    retrieved_context = result.get("context", "")
    source_links = sources_markdown(result.get("sources", []))

    if prediction == "Control":
        required_explanation = (
            f"The system classified this case as a Control profile with a "
            f"confidence score of {confidence * 100:.1f}%. This means that, "
            "based on the extracted linguistic and clinical features, the "
            "system did not detect enough linguistic signs associated with a "
            "probable Alzheimer's disease / dementia profile."
        )
    elif prediction == "ProbableAD":
        required_explanation = (
            "The system detected linguistic patterns that may be associated "
            "with cognitive decline, such as hesitations, repetitions, "
            "corrections, pauses, reduced information content, or difficulties "
            "organizing speech."
        )
    else:
        required_explanation = (
            "The system produced an analysis result based on the extracted "
            "linguistic and clinical features."
        )

    feature_meanings = {
        "Filled pauses": "hesitation markers such as repeated filler sounds",
        "Phonological fragments": "partial or interrupted word productions",
        "Paralinguistic markers": "non-word vocal or speech-related markers",
        "Repetitions / corrections": "repeated words or self-corrections",
        "Unintelligible words": "words that could not be clearly understood",
        "Pauses": "silent breaks or interruptions in speech flow",
        "Age": "patient age used as a clinical feature",
        "Sex": "patient sex used as a clinical feature",
        "Years of education": "education level used as a clinical feature",
    }

    observed_signs = "\n".join(
        (
            f"- **{name}:** {value}. "
            f"This represents {feature_meanings.get(name, 'a recorded clinical or linguistic feature')}."
        )
        for name, value in analysis_features.items()
    )

    prompt = f"""
You are a medical assistant writing a clear patient-facing report in English.

Rules:
- Do not provide a final medical diagnosis.
- Use simple patient-friendly language.
- Explain the result using only the model prediction, confidence score,
  extracted linguistic/clinical features, and retrieved medical context.
- Do not invent patient details.
- Do not mention FAISS, embeddings, chunks, or technical implementation details.
- Mention that the result must be interpreted by a healthcare professional.
- Do not include local file paths.
- Do not include page numbers.
- Do not write in French.
- Do not output labels such as SIMPLE_EXPLANATION, OBSERVED_LINGUISTIC_SIGNS,
  OBSERVE_LINGUISTIC_SIGNS, or MODEL_OUTPUT.
- Do not invent psychological or emotional interpretations.
- The Cookie Theft image is only the picture-description task used to collect
  speech. The image content is not medical evidence.
- Do not interpret picture content as a symptom or disorder.
- Do not say that cookies, a boy, a girl, a mother, running water, a sink,
  laughter, or objects/actions in the picture are signs of disease.
- Focus only on linguistic signs such as filled pauses, hesitations,
  repetitions, corrections, phonological fragments, unintelligible words,
  pauses, reduced information content, discourse organization, and
  word-finding difficulties.

Patient information:
{dict_to_markdown(patient_info)}

Model result:
{prediction_display(prediction)}

Confidence score:
{confidence * 100:.1f}%

Clinical / linguistic information:
{analysis_features_markdown(analysis_features)}

Retrieved medical context:
{retrieved_context[:3500]}

Required clinical interpretation:
{required_explanation}

Write one short patient-friendly paragraph for the Simple Explanation section.
Do not include a heading, label, bullet list, source list, or extra disclaimer.
"""

    llm_explanation = safe_generate_with_llm(prompt).strip()
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
        "picture object",
        "running water",
    ]

    for label in forbidden_labels:
        llm_explanation = llm_explanation.replace(label, "").strip()

    llm_explanation = "\n".join(
        line.strip()
        for line in llm_explanation.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    if any(term in llm_explanation.lower() for term in forbidden_content_terms):
        simple_explanation = required_explanation
    elif not llm_explanation:
        simple_explanation = required_explanation
    else:
        simple_explanation = f"{required_explanation} {llm_explanation}"

    patient_name = patient_info.get("Patient Name", "Not provided")
    title = "# Patient Report"

    if patient_name != "Not provided":
        title += f" - {patient_name}"

    report = f"""{title}

{patient_info_markdown(patient_info)}

## Analysis Result

The system result is: **{prediction_display(prediction)}**.

## Confidence Score

The confidence score is **{confidence * 100:.1f}%**.

## Simple Explanation

{simple_explanation}

## Observed Linguistic Signs

{observed_signs}

## Medical Sources Used

{source_links}

## Important Medical Note

{medical_warning()}
"""

    return report


# ============================================================
# Streamlit interface
# ============================================================

st.set_page_config(
    page_title="Patient RAG Assistant",
    page_icon="brain",
    layout="wide"
)

st.title("Patient RAG Assistant for Language Analysis")

st.warning(medical_warning())


# ============================================================
# Session state
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "result" not in st.session_state:
    st.session_state.result = None

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "patient_report" not in st.session_state:
    st.session_state.patient_report = None

if "patient_report_path" not in st.session_state:
    st.session_state.patient_report_path = None

if st.session_state.get("report_template_version") != REPORT_TEMPLATE_VERSION:
    st.session_state.patient_report = None
    st.session_state.patient_report_path = None
    st.session_state.report_template_version = REPORT_TEMPLATE_VERSION

if "patient_info" not in st.session_state:
    st.session_state.patient_info = {}

if "analysis_features" not in st.session_state:
    st.session_state.analysis_features = {}


# ============================================================
# Sidebar: patient data
# ============================================================

with st.sidebar:
    st.header("Patient Information")

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Example: Wafae"
    )

    patient_id = st.text_input(
        "Patient ID",
        placeholder="Example: P001"
    )

    study_date = st.date_input(
        "Study Date",
        value=datetime.now().date()
    )

    clinician = st.text_input(
        "Responsible Clinician",
        placeholder="Example: Dr Mohamed"
    )

    clinical_notes = st.text_area(
        "Clinical Notes",
        height=90,
        placeholder="Add relevant notes for the report."
    )

    transcript = st.text_area(
        "Patient Transcript",
        height=250,
        placeholder="Enter the patient's English picture-description transcript..."
    )

    st.subheader("Clinical / Linguistic Information")

    n_filled_pauses = st.number_input(
        "Filled pauses",
        min_value=0,
        value=0
    )

    n_phon_fragments = st.number_input(
        "Phonological fragments",
        min_value=0,
        value=0
    )

    n_paralinguistic = st.number_input(
        "Paralinguistic markers",
        min_value=0,
        value=0
    )

    n_retracings = st.number_input(
        "Repetitions / corrections",
        min_value=0,
        value=0
    )

    n_unintelligible = st.number_input(
        "Unintelligible words",
        min_value=0,
        value=0
    )

    n_pauses = st.number_input(
        "Pauses",
        min_value=0,
        value=0
    )

    entryage = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=70
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    educ = st.number_input(
        "Years of education",
        min_value=0,
        max_value=30,
        value=12
    )

    sex_label = "Female" if sex == 0 else "Male"

    patient_info = build_patient_info(
        patient_name=patient_name,
        patient_id=patient_id,
        study_date=study_date,
        age=entryage,
        sex_label=sex_label,
        education_years=educ,
        clinician=clinician,
        clinical_notes=clinical_notes,
    )

    analysis_features = build_analysis_features(
        n_filled_pauses=n_filled_pauses,
        n_phon_fragments=n_phon_fragments,
        n_paralinguistic=n_paralinguistic,
        n_retracings=n_retracings,
        n_unintelligible=n_unintelligible,
        n_pauses=n_pauses,
        entryage=entryage,
        sex=sex,
        educ=educ,
    )

    feature_values = [
        n_filled_pauses,
        n_phon_fragments,
        n_paralinguistic,
        n_retracings,
        n_unintelligible,
        n_pauses,
        entryage,
        sex,
        educ,
    ]

    if st.button("Analyze Patient"):
        if transcript.strip() == "":
            st.error("Please enter a patient transcript.")
        else:
            with st.spinner("Analysis in progress..."):
                result = run_full_pipeline(
                    transcript=transcript,
                    feature_values=feature_values
                )

            st.session_state.result = result
            st.session_state.transcript = transcript
            st.session_state.patient_info = patient_info
            st.session_state.analysis_features = analysis_features
            st.session_state.analysis_done = True
            st.session_state.patient_report = None
            st.session_state.patient_report_path = None

            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "The analysis is complete. You can now ask questions "
                        "about the result, the observed signs, or the medical "
                        "sources used."
                    )
                }
            ]


# ============================================================
# Main display
# ============================================================

if st.session_state.analysis_done:
    result = st.session_state.result
    prediction = result["prediction"]
    confidence = result["confidence"]
    active_patient_info = st.session_state.patient_info or patient_info
    active_analysis_features = st.session_state.analysis_features or analysis_features

    st.subheader("Patient Information")
    st.markdown(patient_info_markdown(active_patient_info))

    st.divider()

    st.subheader("Analysis Result")

    col1, col2 = st.columns(2)

    if prediction == "ProbableAD":
        col1.error(prediction_display(prediction))
    else:
        col1.success(prediction_display(prediction))

    col2.metric("Confidence Score", f"{confidence * 100:.1f}%")

    st.warning(medical_warning())

    st.subheader("Simple Interpretation")
    st.write(simple_interpretation(prediction))

    st.divider()

    # ========================================================
    # Chat RAG
    # ========================================================

    st.subheader("Chat RAG")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask a question about the result...")

    if user_question:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving medical evidence and generating an answer..."):
                answer = build_chat_answer(
                    question=user_question,
                    transcript=st.session_state.transcript,
                    prediction=prediction,
                    confidence=confidence
                )

            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    st.divider()

    # ========================================================
    # Patient report
    # ========================================================

    st.subheader("Patient Report")

    if st.button("Generate Patient Report"):
        with st.spinner("Generating patient report..."):
            report = generate_patient_report(
                transcript=st.session_state.transcript,
                result=result,
                patient_info=active_patient_info,
                analysis_features=active_analysis_features
            )
            report_path = save_patient_report(report)

            st.session_state.patient_report = report
            st.session_state.patient_report_path = report_path

        st.success(f"Patient report saved to: {report_path}")

    if st.session_state.patient_report is not None:
        with st.expander("Show Patient Report", expanded=True):
            st.markdown(st.session_state.patient_report)

        if st.button("Save Patient Report"):
            report_path = save_patient_report(st.session_state.patient_report)
            st.session_state.patient_report_path = report_path
            st.success(f"Patient report saved to: {report_path}")

        download_name = (
            Path(st.session_state.patient_report_path).name
            if st.session_state.patient_report_path
            else "patient_report.md"
        )

        st.download_button(
            label="Download Patient Report",
            data=st.session_state.patient_report,
            file_name=download_name,
            mime="text/markdown"
        )

else:
    st.info(
        "Enter the patient transcript and patient information in the sidebar, "
        "then click 'Analyze Patient'."
    )
