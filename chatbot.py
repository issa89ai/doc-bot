"""CLI using the same indexing and retrieval code as the web app."""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import rag


def main():
    files = sorted(Path(rag.DOCS_DIR).glob("*.pdf"))
    if not files:
        print("Place PDFs in docs/ first.")
        return
    for path in files:
        print(f"Indexing {path.name}: {rag.load_and_index(str(path))} chunks")
    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        reply, sources = rag.answer(question, history)
        print(f"Bot: {reply}\nSources: {', '.join(sources)}")
        history = (history + [(question, reply)])[-3:]


if __name__ == "__main__":
    main()
