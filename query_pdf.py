"""
query_pdf.py
Simple script to query the FAISS index created in Part 1.

What it does:
- Loads the HuggingFace embedding (same model used when building the index)
- Loads the local FAISS index
- Runs a similarity_search_with_score for a given question
- Prints the top-k returned chunks, their metadata, and a simple normalized score

Run:
python query_pdf.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
# Use the community vectorstores import (depends on your langchain version)
from langchain_community.vectorstores import FAISS
import textwrap

# SETTINGS - change if you used a different model or index path
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = "faiss_index"   # folder created by build_index.py
TOP_K = 2                          # how many chunks to fetch

def load_vectordb():
    """
    Load the FAISS index from disk using the same embedding class we used to build it.
    It's crucial the embedding class (model) is identical to avoid mismatches.
    """
    embed = HuggingFaceEmbeddings(model_name=HF_MODEL, model_kwargs={"device": "cpu"})
    vectordb = FAISS.load_local(VECTORSTORE_PATH, embed,allow_dangerous_deserialization=True)
    return vectordb

def normalize_score_from_distance(dist):
    """
    FAISS returns distances (smaller = more similar).
    This converts distance -> a 0..1 like score using a simple heuristic:
        score = 1 / (1 + distance)
    It's not a true cosine similarity, but gives an intuitive higher-is-better number.
    """
    try:
        return 1.0 / (1.0 + float(dist))
    except Exception:
        return 0.0

def pretty_print_result(rank, doc, dist):
    """
    Print a single search result nicely with metadata and a short content snippet.
    """
    score = normalize_score_from_distance(dist)
    # doc.page_content is the chunk text; doc.metadata should contain 'source' (filename) and maybe 'page'
    src = doc.metadata.get("source") if isinstance(doc.metadata, dict) else getattr(doc, "metadata", None)
    # Some loaders put page number in metadata under 'page' or 'page_number'; try both
    page = None
    if isinstance(doc.metadata, dict):
        page = doc.metadata.get("page") or doc.metadata.get("page_number") or doc.metadata.get("page_index")

    snippet = doc.page_content.strip().replace("\n", " ")
    if len(snippet) > 800:
        snippet = snippet[:800] + "..."

    print(f"\n--- Result #{rank} (score ~ {score:.3f}, dist={dist}) ---")
    print(f"Source: {src}" + (f" | Page: {page}" if page is not None else ""))
    print("Snippet:")
    print(textwrap.fill(snippet, width=100))
    print("-" * 80)

def query_loop():
    """
    Interactive loop: ask question, run search, print top-k results.
    """
    vectordb = load_vectordb()
    print("FAISS index loaded from:", VECTORSTORE_PATH)
    print("Model used for embeddings:", HF_MODEL)
    print("Type 'exit' or empty line to quit.\n")

    while True:
        q = input("Enter your question: ").strip()
        if q.lower() in ("", "exit", "quit"):
            print("Bye.")
            break

        # similarity_search_with_score returns list of (Document, distance)
        results = vectordb.similarity_search_with_score(q, k=TOP_K)

        if not results:
            print("No results returned.")
            continue

        print(f"\nTop {len(results)} results for query: {q!r}")

        for i, (doc, dist) in enumerate(results, start=1):
            pretty_print_result(i, doc, dist)

        # a tiny heuristic: show best normalized score
        best_score = normalize_score_from_distance(results[0][1])
        print(f"\nBest approx. score: {best_score:.3f}")
        if best_score < 0.45:
            print("Note: similarity is low — the answer might not be in your PDFs (consider fallback to Wikipedia).")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    query_loop()
