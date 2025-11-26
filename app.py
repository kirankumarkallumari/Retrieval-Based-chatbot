import streamlit as st

from qa_pdf_wiki import (
    load_vectordb,
    # get_pdf_ans,
    wikipedia_answer,
    HF_MODEL,
    VECTORSTORE_PATH,
    pdf_threshold,
    get_pdf_ans
)

@st.cache_resource
def get_vectordb():
    """
    Wrapper around load_vectordb() that caches the result in Streamlit.
    So the index is loaded only once.
    """
    return load_vectordb()


def main():
    st.set_page_config(page_title="PDF + Wikipedia Q&A", page_icon="📚")

    st.title("📚 PDF + 🌐 Wikipedia Q&A Bot")
    st.write("Ask a question. I will first search your PDFs, then fall back to Wikipedia if needed.")

    with st.spinner("Loading PDF index..."):
        vectordb = get_vectordb()

    st.success(f"Index loaded from: {VECTORSTORE_PATH}")
    st.caption(f"Embedding model: {HF_MODEL}")

    question = st.text_input("Your question:")

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            pdf_answer, score = get_pdf_ans(question)

            if pdf_answer and score >= pdf_threshold:
                st.subheader(f"Answer from PDFs (score ~ {score:.3f})")
                st.write(pdf_answer)
                
            else:
                if pdf_answer:
                    st.info(f"PDF match score is low ({score:.3f}) → using Wikipedia.")
                else:
                    st.info("No relevant chunks found in PDFs → using Wikipedia.")

                wiki_answer = wikipedia_answer(question)
                st.subheader("Answer from Wikipedia")
                st.write(wiki_answer)

if __name__ == "__main__":
    main()
