from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "rag_documents_mri"
FAISS_DIR = BASE_DIR / "vector_store" / "faiss_mri_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_dependencies():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ModuleNotFoundError:
        return None

    return PyPDFLoader, FAISS, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter


def load_pdf_documents(pdf_loader):
    documents = []
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        return []

    for pdf_path in pdf_files:
        loader = pdf_loader(str(pdf_path))
        documents.extend(loader.load())

    return documents


def split_documents(documents, text_splitter):
    splitter = text_splitter(
        chunk_size=700,
        chunk_overlap=120,
    )
    return splitter.split_documents(documents)


def build_faiss_index(chunks, faiss_class, embedding_class):
    embeddings = embedding_class(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = faiss_class.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))


def main() -> None:
    dependencies = load_dependencies()
    if dependencies is None:
        print("MRI RAG index could not be built.")
        return

    pdf_loader, faiss_class, embedding_class, text_splitter = dependencies
    documents = load_pdf_documents(pdf_loader)
    if not documents:
        print("No MRI documents found.")
        return

    chunks = split_documents(documents, text_splitter)
    build_faiss_index(chunks, faiss_class, embedding_class)
    print("MRI RAG index built successfully.")


if __name__ == "__main__":
    main()
