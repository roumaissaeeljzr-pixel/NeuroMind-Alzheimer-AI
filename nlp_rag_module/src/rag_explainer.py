from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parents[1]
FAISS_DIR = BASE_DIR / "vector_store" / "faiss_alzheimer_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieve_context(transcript, prediction, vectorstore, k=4):
    query = f"""
    The NLP classification model predicted: {prediction}.

    Patient transcript:
    {transcript}

    Find evidence from different medical and scientific documents about:
    - Alzheimer's disease and dementia symptoms
    - language impairment in Alzheimer's disease
    - word-finding difficulties
    - pauses, hesitations and repetitions in speech
    - reduced lexical richness and reduced information content
    - speech biomarkers for dementia
    - NLP-based dementia detection using patient transcripts
    - mild cognitive impairment and cognitive decline

    Prefer diverse sources from different PDF documents.
    """

    # récupérer plus de documents au début
    candidate_docs = vectorstore.similarity_search(query, k=20)

    # garder des sources différentes
    selected_docs = []
    used_sources = set()

    for doc in candidate_docs:
        source = doc.metadata.get("source", "unknown source")

        if source not in used_sources:
            selected_docs.append(doc)
            used_sources.add(source)

        if len(selected_docs) == k:
            break

    # si on n'a pas assez de sources différentes, compléter avec les meilleurs restants
    if len(selected_docs) < k:
        for doc in candidate_docs:
            if doc not in selected_docs:
                selected_docs.append(doc)

            if len(selected_docs) == k:
                break

    context_parts = []

    for doc in selected_docs:
        source = doc.metadata.get("source", "unknown source")
        source_name = Path(str(source)).name

        context_parts.append(
            f"Source: {source_name}\n"
            f"{doc.page_content}"
        )

    context = "\n\n" + ("-" * 80 + "\n\n").join(context_parts)

    return context, selected_docs

def build_explanation(transcript, prediction, confidence, retrieved_docs):
    """
    Version simple sans API LLM.
    Elle génère une explication basée sur les passages récupérés.
    """

    evidence = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        source_name = Path(str(source)).name
        text = doc.page_content.replace("\n", " ")
        text = " ".join(text.split())

        evidence.append(
            f"{i}. Source: {source_name}\n"
            f"   Extrait : {text[:500]}..."
        )

    explanation = f"""
Prédiction : {prediction}

Niveau de confiance : {confidence:.2f}

Explication :
Le système a prédit la classe "{prediction}" pour ce patient.
Les documents récupérés concernent la démence, la maladie d'Alzheimer,
les troubles du langage, les marqueurs linguistiques et le déclin cognitif.

Cette prédiction peut être mise en relation avec des signes linguistiques
possibles dans la parole, par exemple :
- difficultés à trouver les mots,
- pauses ou hésitations,
- contenu informatif réduit,
- répétitions,
- troubles du langage,
- déclin cognitif possible.

Remarque importante :
Ce résultat n'est pas un diagnostic médical final. Il s'agit d'une aide à
l'analyse qui doit être interprétée par un professionnel de santé.

Sources récupérées :
{chr(10).join(evidence)}
"""

    return explanation


def explain_prediction_with_rag(transcript, prediction, confidence, k=4):
    vectorstore = load_vectorstore()

    context, retrieved_docs = retrieve_context(
        transcript=transcript,
        prediction=prediction,
        vectorstore=vectorstore,
        k=k
    )

    explanation = build_explanation(
        transcript=transcript,
        prediction=prediction,
        confidence=confidence,
        retrieved_docs=retrieved_docs
    )

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown")
        }
        for doc in retrieved_docs
    ]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "context": context,
        "explanation": explanation,
        "sources": sources
    }


if __name__ == "__main__":
    transcript = """
    The boy is taking cookies from the jar.
    The mother is washing dishes.
    The water is falling.
    I don't know... maybe the boy will fall.
    """

    prediction = "ProbableAD"
    confidence = 0.87

    result = explain_prediction_with_rag(
        transcript=transcript,
        prediction=prediction,
        confidence=confidence,
        k=4
    )

    print(result["explanation"])
