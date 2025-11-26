from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

from dotenv import load_dotenv
load_dotenv()

PDF_FOLDER = "research_papers"
VECTOR_INDEX = "faiss_index"

#load pdfs
def load_pdfs():
    loader = DirectoryLoader(PDF_FOLDER,glob="**/*.pdf",loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"loaded {len(docs)} documents")
    return docs

#2.splitting docs into smaller chunks

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"split into {len(chunks)} chunks")
    return chunks

#creating embeddings using huggingface
def build_index():
    print("loading pdf's....")
    docs = load_pdfs()

    print("splitting into chunks....")
    chunks = split_docs(docs)

    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("creating faiss index....")
    vectordb = FAISS.from_documents(chunks,embedder)

    print("Saving index...")
    vectordb.save_local(VECTOR_INDEX)
    print("FAISS index saved")


if __name__ == "__main__":
    build_index()
