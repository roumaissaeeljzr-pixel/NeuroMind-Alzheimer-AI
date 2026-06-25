from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "rag_documents"
FAISS_DIR = BASE_DIR / "vector_store" / "faiss_alzheimer_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_documents():
    docs = []
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"Aucun fichier PDF trouvé dans : {PDF_DIR}")

    print("PDF trouvés :")
    for pdf_path in pdf_files:
        print("-", pdf_path.name)

    print("\nChargement des PDF...")
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        docs.extend(pages)
        print(f"{pdf_path.name} : {len(pages)} pages")

    print(f"\nNombre total de pages chargées : {len(docs)}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    print(f"Nombre total de chunks : {len(chunks)}")
    return chunks


def build_faiss_index(chunks):
    print("\nChargement du modèle d'embedding RAG...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    print("Création de l'index FAISS...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print(f"\nIndex FAISS sauvegardé dans : {FAISS_DIR}")


if __name__ == "__main__":
    print("=" * 70)
    print("BUILD RAG INDEX - ALZHEIMER")
    print("=" * 70)

    docs = load_pdf_documents()
    chunks = split_documents(docs)
    build_faiss_index(chunks)

    print("\nTerminé avec succès.")