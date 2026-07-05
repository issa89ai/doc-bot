from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os
import glob
import shutil

DOCS_DIR = "docs"
CHROMA_DIR = "./chroma_db"

# When running in Docker, OLLAMA_HOST points to the host machine's Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
llm = ChatOllama(model="llama3.2", base_url=OLLAMA_HOST)

prompt = ChatPromptTemplate.from_template("""
You are a document assistant. Answer questions based on the document excerpts below.

Rules:
- Base your answer on the document excerpts provided. Do not use outside knowledge.
- You CAN summarize, infer the topic, and draw conclusions from what is written in the excerpts.
- If the document contains no relevant information at all, say: "This information is not available in the uploaded documents."
- Keep answers clear and concise.

{history}Document excerpts:
{context}

Question: {question}

Answer:""")


def load_and_index(pdf_path: str) -> int:
    """Load a single PDF, embed it, add to the vector store. Returns chunk count."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for page in pages:
        page.metadata["source_file"] = os.path.basename(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    return len(chunks)


def get_retriever():
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for q, a in history[-3:]:
        lines.append(f"Human: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines) + "\n\n"


def get_sources(docs) -> list[str]:
    seen = []
    for doc in docs:
        file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page")
        label = f"{file} p.{page + 1}" if page is not None else file
        if label not in seen:
            seen.append(label)
    return seen


def list_indexed_files() -> list[str]:
    return sorted([
        os.path.basename(f)
        for f in glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    ])


def answer(question: str, history: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Returns (answer_text, sources_list)."""
    retriever = get_retriever()
    docs = retriever.invoke(question)
    sources = get_sources(docs)

    filled = prompt.invoke({
        "context": format_docs(docs),
        "question": question,
        "history": format_history(history),
    })

    response = llm.invoke(filled)
    return response.content, sources
