import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PATH_PDF = "data/raw/"

def load_documents(path: str):
    return PyPDFDirectoryLoader(path, glob="*.pdf").load()

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def load_vector(chunks, path):
    # Modelo excelente, multilíngue (ótimo para português) e roda na sua máquina
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct"
        # Outras opções boas e menores:
        # "sentence-transformers/all-MiniLM-L6-v2"           # mais rápido, 384 dim
        # "nomic-ai/nomic-embed-text-v1.5"                   # SOTA open-source
        # "intfloat/multilingual-e5-base"                    # equilíbrio
    )
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=path
    )
    print(f"Vetorstore salvo em: {path}")
    return db

if __name__ == "__main__":

    # Carregar Documentos

    init_rag = datetime.now()
    
    documents = load_documents(PATH_PDF)
    print(f"Total de documentos: {len(documents)}")
    exec_time = datetime.now() - init_rag
    print("Tempo de execução:", exec_time)

    # Fatiar Documentos

    chunks = create_chunks(documents)
    print(f"Total de chunks: {len(chunks)}")
    exec_time = datetime.now() - init_rag
    print("Tempo de execução:", exec_time)

    # Vetorizar Fatias
    
    db = load_vector(chunks, "data/curated/")
    exec_time = datetime.now() - init_rag
    print("Tempo de execução:", exec_time)