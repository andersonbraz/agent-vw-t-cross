import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PATH_PDF = "data/raw/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def load_documents(path:str) -> list:
    documents = PyPDFDirectoryLoader(path, glob="*.pdf").load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def load_vetor(chunks, path):

    embeddings = GoogleGenerativeAIEmbeddings(
        model = "models/embedding-001",
        google_api_key = GEMINI_API_KEY
    )

    db = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = path
    )

    db.persist()
    return db

if __name__ == "__main__":

    # Step 01

    documents = load_documents(path=PATH_PDF)
    # print(f"Total documents:", len(documents))

    # Step 02

    chunks = create_chunks(documents)
    print(f"Total de Chunks:", len(chunks))
    # print(chunks[0].page_content[:200] + "...")


    for chunk in chunks:
        print(chunk)

    # Step 03

    # db = load_vetor(chunks, "data/curated/")

