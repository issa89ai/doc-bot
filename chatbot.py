from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sys
import os
import glob
import shutil

# ── 1. Load all PDFs from docs/ folder ───────────────────────────────────────
DOCS_DIR = "docs"

if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    if os.path.exists("document.pdf"):
        shutil.copy("document.pdf", os.path.join(DOCS_DIR, "document.pdf"))
        print("Copied document.pdf → docs/document.pdf")

pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
if not pdf_files:
    print(f"ERROR: No PDF files found in '{DOCS_DIR}/'")
    print(f"Place one or more PDFs in the '{DOCS_DIR}/' folder, then run again.")
    sys.exit(1)

print(f"Loading {len(pdf_files)} PDF(s)...")
pages = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    loaded = loader.load()
    for page in loaded:
        page.metadata["source_file"] = os.path.basename(pdf_path)
    pages.extend(loaded)
    print(f"  {os.path.basename(pdf_path)}: {len(loaded)} pages")

# ── 2. Split into chunks ──────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(pages)
print(f"  Total chunks: {len(chunks)}")

# ── 3. Embed & store in ChromaDB (rebuild each run so new docs are picked up) ─
print("Building vector store...")
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
)
print("  Vector store ready.")

# ── 4. Prompt with conversation history ───────────────────────────────────────
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context below.
If you don't know, say "I don't have enough information."

{history}Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for q, a in history[-3:]:
        lines.append(f"Human: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines) + "\n\n"

def get_sources(docs):
    seen = []
    for doc in docs:
        file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page")
        entry = (file, page)
        if entry not in seen:
            seen.append(entry)
    return seen

# ── 5. Chat loop ──────────────────────────────────────────────────────────────
print(f"\nChatbot ready! Drop PDFs into '{DOCS_DIR}/' to add more documents.")
print("Type your question (or 'quit' to exit).\n")

history = []

while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    docs = retriever.invoke(question)
    sources = get_sources(docs)
    context_str = format_docs(docs)
    history_str = format_history(history)

    filled_prompt = prompt.invoke({
        "context": context_str,
        "question": question,
        "history": history_str,
    })

    print("\nBot: ", end="", flush=True)
    answer = ""
    for chunk in llm.stream(filled_prompt):
        print(chunk.content, end="", flush=True)
        answer += chunk.content

    if sources:
        src_parts = [
            f"{f} p.{p + 1}" if p is not None else f
            for f, p in sorted(sources)
        ]
        print(f"\n\n[Sources: {', '.join(src_parts)}]")
    print()

    history.append((question, answer))
