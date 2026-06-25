from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from urllib.parse import quote_plus

PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parents[1]
NLP_SRC_DIR = PROJECT_DIR / "nlp_rag_module" / "src"
FAISS_DIR = BASE_DIR / "vector_store" / "faiss_mri_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MEDICAL_NOTE = (
    "This result is not a medical diagnosis. It is only a decision-support "
    "output and must be interpreted by a healthcare professional."
)
FORBIDDEN_PHRASES = [
    "cn is a region",
    "cn is a region of the brain",
    "cn indicates a brain area",
    "cn means a brain region",
    "cn means a cognitive brain region",
    "portion identified as cn",
    "portion of the brain",
    "brain area",
    "brain region",
    "region of the brain",
    "specific subregions",
    "specific subregion",
    "subregion",
    "subregions",
    "the patient has alzheimer",
    "this confirms alzheimer",
    "confirms alzheimer",
    "red area is diseased",
    "red areas are diseased",
    "highlighted region is damaged",
    "highlighted area is damaged",
    "the mri proves alzheimer",
]
FORBIDDEN_ANATOMY_TERMS = [
    "hippocampus",
    "temporal lobe",
    "frontal lobe",
    "parietal lobe",
    "occipital lobe",
    "amygdala",
    "ventricle",
    "cortex",
    "cortical",
    "white matter",
    "gray matter",
    "grey matter",
]
TECHNICAL_TERM_REPLACEMENTS = {
    "CNN": "AI brain scan analysis",
    "cnn": "AI brain scan analysis",
    "Grad-CAM": "visual explanation",
    "grad-cam": "visual explanation",
    "GradCAM": "visual explanation",
    "gradcam": "visual explanation",
    "LLM": "simple explanation",
    "llm": "simple explanation",
    "RAG": "medical document-based explanation",
    "FAISS": "medical document search",
    "faiss": "medical document search",
    "embeddings": "medical document matching",
    "vector store": "medical document index",
    "architecture": "system",
    "model prediction": "AI result",
    "model output": "AI result",
    "model decision": "AI decision",
    "the model": "the AI system",
}
FORBIDDEN_TECHNICAL_TERMS = [
    "cnn",
    "llm",
    "rag",
    "grad-cam",
    "gradcam",
    "faiss",
    "embeddings",
    "vector store",
    "model file",
    "python",
    "architecture",
    ".pth",
]


def get_prediction_meaning(prediction):
    if prediction == "CN":
        return "Cognitively Normal profile"
    if prediction == "AD":
        return "Alzheimer's Disease compatible profile"
    return "Unknown profile"


def _prediction_meaning(prediction: str) -> str:
    return get_prediction_meaning(prediction)


def _percent_text(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "not available"


def format_mri_result_values(cnn_result: dict, gradcam_info: dict | None = None) -> dict:
    prediction = str(cnn_result.get("prediction", "Unknown"))
    confidence = cnn_result.get("confidence")
    prob_cn = cnn_result.get("prob_cn")
    prob_ad = cnn_result.get("prob_ad")
    orientation = "Not generated"
    num_slices = "Not generated"

    if isinstance(gradcam_info, dict):
        orientation = str(gradcam_info.get("orientation") or "Not generated")
        raw_num_slices = gradcam_info.get("number_of_slices")
        if isinstance(raw_num_slices, int) and raw_num_slices > 0:
            num_slices = str(raw_num_slices)

    return {
        "prediction": prediction,
        "prediction_meaning": get_prediction_meaning(prediction),
        "confidence_pct": _percent_text(confidence),
        "prob_cn_pct": _percent_text(prob_cn),
        "prob_ad_pct": _percent_text(prob_ad),
        "orientation": orientation,
        "num_slices": num_slices,
    }


def safe_mri_explanation_template(
    prediction_meaning,
    confidence_pct,
    prob_cn_pct,
    prob_ad_pct,
) -> str:
    return (
        f"The AI brain scan analysis classified this scan as {prediction_meaning}. "
        f"The confidence score is {confidence_pct}. The probability of a "
        f"Cognitively Normal profile is {prob_cn_pct}, and the probability of an "
        f"Alzheimer's Disease compatible profile is {prob_ad_pct}.\n\n"
        "The visual explanation shows the areas of the image that influenced the "
        "AI result. Warmer colors indicate stronger influence on the result, but "
        "they do not mean that these areas are diseased.\n\n"
        f"{MEDICAL_NOTE}"
    )


def _safe_mri_explanation_from_values(values: dict) -> str:
    return safe_mri_explanation_template(
        values["prediction_meaning"],
        values["confidence_pct"],
        values["prob_cn_pct"],
        values["prob_ad_pct"],
    )


def _ensure_llm_path() -> None:
    module_path = str(NLP_SRC_DIR)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)


def _load_vectorstore():
    if not (FAISS_DIR / "index.faiss").exists():
        raise FileNotFoundError("MRI RAG index is not available.")

    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _source_title(source) -> str:
    title = Path(str(source or "medical source")).stem
    title = title.replace("_", " ").replace("-", " ").strip() or "medical source"
    title = title.replace("gradcam", "visual explanation")
    title = title.replace("GradCAM", "visual explanation")
    title = title.replace("Grad Cam", "visual explanation")
    title = title.replace("xai", "explainable AI")
    return title


def _source_link(title: str) -> str:
    return "https://www.google.com/search?q=" + quote_plus(title + " MRI Alzheimer PDF")


def _format_sources(docs) -> list[dict]:
    sources = []
    seen = set()
    for doc in docs:
        title = _source_title(doc.metadata.get("source", "medical source"))
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        page = doc.metadata.get("page")
        page_number = int(page) + 1 if isinstance(page, int) else None
        sources.append(
            {
                "title": title,
                "page": page_number,
                "url": _source_link(title),
            }
        )
    return sources


def _retrieval_query(cnn_result: dict, gradcam_info: dict | None, question: str | None) -> str:
    values = format_mri_result_values(cnn_result, gradcam_info)
    gradcam_text = gradcam_info or {}

    patient_question = question or (
        "Explain a brain MRI artificial intelligence result, visual explanation, confidence, "
        "probabilities, and medical limitations for a patient."
    )

    return f"""
    Patient question:
    {patient_question}

    Brain scan analysis result:
    Prediction: {values["prediction"]}
    Prediction meaning: {values["prediction_meaning"]}
    Confidence score: {values["confidence_pct"]}
    Probability of Cognitively Normal profile: {values["prob_cn_pct"]}
    Probability of Alzheimer's Disease compatible profile: {values["prob_ad_pct"]}

    Visual explanation information:
    {gradcam_text}

    Retrieve medical context about structural MRI in Alzheimer's disease,
    cognitively normal profiles, Alzheimer-compatible MRI profiles,
    explainable artificial intelligence, heatmaps, confidence, decision support,
    and limitations of automated neuroimaging analysis.
    """


def _retrieve_context(cnn_result: dict, gradcam_info: dict | None, question: str | None, k: int = 4):
    vectorstore = _load_vectorstore()
    query = _retrieval_query(cnn_result, gradcam_info, question)
    candidate_docs = vectorstore.similarity_search(query, k=12)

    selected_docs = []
    used_sources = set()
    for doc in candidate_docs:
        source = str(doc.metadata.get("source", "medical source"))
        if source in used_sources:
            continue
        selected_docs.append(doc)
        used_sources.add(source)
        if len(selected_docs) >= k:
            break

    if len(selected_docs) < k:
        for doc in candidate_docs:
            if doc not in selected_docs:
                selected_docs.append(doc)
            if len(selected_docs) >= k:
                break

    context_parts = []
    for index, doc in enumerate(selected_docs, start=1):
        title = _source_title(doc.metadata.get("source", "medical source"))
        text = " ".join(doc.page_content.replace("\n", " ").split())
        context_parts.append(f"Document {index}: {title}\n{text[:1400]}")

    return "\n\n".join(context_parts), selected_docs


def _fallback_answer(cnn_result: dict, gradcam_info: dict | None = None) -> str:
    values = format_mri_result_values(cnn_result, gradcam_info)
    return _safe_mri_explanation_from_values(values)


def _answer_uses_only_allowed_percentages(answer: str, values: dict) -> bool:
    percentages = set(re.findall(r"\b\d+(?:\.\d+)?%", answer))
    allowed = {
        values.get("confidence_pct"),
        values.get("prob_cn_pct"),
        values.get("prob_ad_pct"),
    }
    allowed.discard("not available")
    allowed.discard(None)
    return percentages.issubset(allowed)


def _sanitize_answer(answer: str, fallback: str, values: dict) -> str:
    if not answer:
        return fallback

    cleaned = str(answer).strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    for technical_term, replacement in TECHNICAL_TERM_REPLACEMENTS.items():
        cleaned = re.sub(
            re.escape(technical_term),
            replacement,
            cleaned,
            flags=re.IGNORECASE,
        )

    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in FORBIDDEN_PHRASES):
        return fallback
    if any(term in lowered for term in FORBIDDEN_ANATOMY_TERMS):
        return fallback
    if any(term in lowered for term in FORBIDDEN_TECHNICAL_TERMS):
        return fallback
    if not _answer_uses_only_allowed_percentages(cleaned, values):
        return fallback
    if ".pth" in lowered or "python" in lowered or "c:\\" in lowered:
        return fallback
    paragraphs = [paragraph for paragraph in cleaned.split("\n\n") if paragraph.strip()]
    if len(paragraphs) > 3:
        return fallback

    if MEDICAL_NOTE not in cleaned:
        cleaned = f"{cleaned}\n\n{MEDICAL_NOTE}"

    return cleaned


def _build_prompt(
    cnn_result: dict,
    gradcam_info: dict | None,
    question: str | None,
    context: str,
) -> str:
    values = format_mri_result_values(cnn_result, gradcam_info)
    patient_question = str(question or "").strip()
    question_block = (
        f"\nPatient question:\n{patient_question}\n"
        if patient_question
        else ""
    )
    return f"""
System instructions:
You are writing a simple patient-friendly explanation for a brain scan result.

Strict meanings:
- CN means Cognitively Normal profile.
- AD means Alzheimer's Disease compatible profile.
- CN is never a brain region, never a brain area, and never a subregion.
- AD does not confirm Alzheimer's disease.

Use only the values provided by the code.
Do not recalculate percentages.
Do not invent anatomical regions.
Do not mention specific brain subregions unless they are explicitly provided.
Do not say that highlighted areas are diseased.
Do not say that the result confirms Alzheimer's disease.

Avoid technical terms in the patient-facing answer.
Do not use the words CNN, LLM, RAG, Grad-CAM, FAISS, embeddings, vector store, model file, Python, or architecture.

Use these patient-friendly terms:
- AI brain scan analysis
- visual explanation
- highlighted areas
- simple explanation
- decision-support result

Always include this medical note exactly:
{MEDICAL_NOTE}

Patient-friendly brain scan explanation.

Brain scan analysis result:
- Prediction label: {values["prediction"]}
- Prediction meaning: {values["prediction_meaning"]}
- Confidence score: {values["confidence_pct"]}
- Probability of Cognitively Normal profile: {values["prob_cn_pct"]}
- Probability of Alzheimer's Disease compatible profile: {values["prob_ad_pct"]}
- Visual explanation orientation: {values["orientation"]}
- Number of visual explanation images: {values["num_slices"]}
{question_block}
Write a short explanation for a patient.

Rules:
- Do not interpret CN as a brain region.
- Do not mention specific brain subregions.
- Do not invent medical findings.
- Do not say the patient has Alzheimer's disease.
- Do not say the result confirms Alzheimer's disease.
- Explain that the highlighted areas only show parts of the image that influenced the AI result.
- Use simple English.
- Use 3 short paragraphs maximum.
- Include the medical note.

Medical context for background only:
{context}
"""


def generate_mri_explanation_with_rag(
    cnn_result,
    gradcam_info=None,
    question=None,
):
    cnn_result = dict(cnn_result or {})
    prediction = str(cnn_result.get("prediction", "Unknown"))
    prediction_meaning = get_prediction_meaning(prediction)
    cnn_result["prediction_meaning"] = prediction_meaning

    values = format_mri_result_values(cnn_result, gradcam_info)
    fallback = _safe_mri_explanation_from_values(values)
    is_question = bool(str(question or "").strip())

    try:
        context, docs = _retrieve_context(cnn_result, gradcam_info, question)
        sources = _format_sources(docs)
    except Exception:
        context = ""
        sources = []

    if not is_question:
        return {"answer": fallback, "sources": sources}

    try:
        _ensure_llm_path()
        llm_generator = importlib.import_module("llm_generator")
        prompt = _build_prompt(cnn_result, gradcam_info, question, context)
        answer = llm_generator.generate_with_llm(prompt)
        answer = _sanitize_answer(answer, fallback, values)
        return {"answer": answer, "sources": sources}
    except Exception:
        return {
            "answer": fallback,
            "sources": sources,
        }
