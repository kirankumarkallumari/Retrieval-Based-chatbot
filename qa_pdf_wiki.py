from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import wikipedia

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = "faiss_index"
TOP_K=2
pdf_threshold =0.55

def load_vectordb():

    # "load the fiass index from disk using same embedding model0"
    embed = HuggingFaceEmbeddings(model_name=HF_MODEL,model_kwargs={"device":"cpu"})

    vectordb = FAISS.load_local(VECTORSTORE_PATH,embed,allow_dangerous_deserialization=True)
    return vectordb

def normalize_score_from_distance(dist):
    try:
        return 1.0/(1.0+ float(dist))
    except Exception:
        return 0.0


def get_pdf_ans(question):
    vectordb =load_vectordb()
    results = vectordb.similarity_search_with_score(question,k=TOP_K)

    if not results:
        return None,0.0
    

    best_doc,best_dist = results[0]
    best_score = normalize_score_from_distance(best_dist)

    snippets = []
    for doc,dist in results:
        text = doc.page_content.strip().replace("\n"," ")
        snippets.append(text)

    combined_text = " ".join(snippets)

    if len(combined_text)>1000:
        combined_text = combined_text[:1000] + "...."

    return combined_text,best_score

def wikipedia_answer(question):
    try:
        search_results = wikipedia.search(question,results=2)
        if not search_results:
            return " NO relevant answer found on wikipedia"
        
        title = search_results[0]
        summary = wikipedia.summary(title,sentences=4)
        answer = f"From wikipedia({title}):\n\n{summary}"
        return answer
    except Exception:
        return f"Wikipedia lookup failed:"
    

def qa_loop():
    print("Loading vector database(FAISs index)...")
    vectordb = load_vectordb()
    print("embedding model:",HF_MODEL)
    print("You can ask questions. Type 'exit' to quit.\n")

    while True:
        q = input("Your question:").strip()
        if q.lower() in ("exit","quit",""):
            print("Bye!")
            break

        results = vectordb.similarity_search_with_score(q,k=TOP_K)
        if not results:
            print("No pdf results returned, going to wikipedia...\n")
            wiki_answer = wikipedia_answer(q)
            print("\n"+ wiki_answer)
            print('\n' + "="*80 + "\n")
            continue

        best_doc,best_dist = results[0]
        best_score = normalize_score_from_distance(best_dist)

        if best_score>=pdf_threshold:
            snippets=[]
            for doc,dist in results:
                text = doc.page_content.strip().replace("\n"," ")
                snippets.append(text)
            combined_text = " ".join(snippets)
            if len(combined_text)>1000:
                combined_text = combined_text[:1000] + "..."
            print(f"\nAnswe from pdfs(score ~ {best_score:.3f}):\n")
            print(combined_text)

        else:
            print(f"\npdf match score is low ({best_score:.3f}) -> using wikipedia instead.\n")
            wiki_answer = wikipedia_answer(q)
            print(wiki_answer)

        
        print("\n" + "="*80 + "\n")



if __name__ == "__main__":
    qa_loop()

