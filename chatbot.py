from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import sys
import os

# ── 1. Load the PDF ──────────────────────────────────────────────────────────
PDF_PATH = "document.pdf"

if not os.path.exists(PDF_PATH):
    print(f"ERROR: Could not find '{PDF_PATH}'")
    print("Place a PDF named 'document.pdf' in this folder, then run again.")
    sys.exit(1)

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"  Loaded {len(pages)} pages.")

# ── 2. Split into chunks ──────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"  Split into {len(chunks)} chunks.")

# ── 3. Embed & store in ChromaDB ──────────────────────────────────────────────
print("Building vector store (this may take a minute on first run)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("  Vector store ready.")

# ── 4. Build the RAG chain ───────────────────────────────────────────────────
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know, say "I don't have enough information."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 5. Chat loop ─────────────────────────────────────────────────────────────
print("\nChatbot ready! Type your question (or 'quit' to exit).\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    print("\nBot: ", end="", flush=True)
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")
