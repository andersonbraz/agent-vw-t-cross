import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Configurações centralizadas
PATH_PDF = "data/raw/"
VECTOR_DB_PATH = "data/curated/t-cross_index/"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"  # você pode trocar quando quiser

# Cache do modelo para não baixar toda vez (economiza tempo e banda)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},                    # mude para "cuda" se tiver GPU
    encode_kwargs={"normalize_embeddings": True},      # melhora qualidade do retrieval
    cache_folder=os.path.expanduser("~/.cache/huggingface/hub"),  # opcional, deixa explícito
)

def load_documents(path: str):
    loader = PyPDFDirectoryLoader(path, glob="*.pdf")
    docs = loader.load()
    print(f"Documentos carregados: {len(docs)}")
    return docs

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Chunks criados: {len(chunks)}")
    return chunks

def create_or_load_vectorstore(chunks=None, path=VECTOR_DB_PATH):
    """
    Cria o vetorstore se não existir.
    Se já existir, só carrega (economiza horas na próxima execução!)
    """
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        print(f"Carregando vetorstore existente de: {path}")
        db = Chroma(persist_directory=path, embedding_function=embeddings)
    else:
        print(f"Criando novo vetorstore em: {path}")
        if chunks is None:
            raise ValueError("É necessário fornecer chunks para criar um novo vetorstore")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=path
        )
        print("Vetorização concluída e salva!")
    return db

# ======================
#        EXECUÇÃO
# ======================
if __name__ == "__main__":
    inicio_total = datetime.now()

    # 1. Carrega documentos
    docs = load_documents(PATH_PDF)

    # 2. Cria chunks
    chunks = create_chunks(docs)

    # 3. Cria OU carrega o vetorstore (não vai refazer tudo na próxima vez!)
    db = create_or_load_vectorstore(chunks=chunks, path=VECTOR_DB_PATH)

    tempo_total = datetime.now() - inicio_total
    print(f"\nRAG PRONTO PARA USO!")
    print(f"Tempo total de processamento: {tempo_total}")
    print(f"Vetorstore salvo em: {VECTOR_DB_PATH}")
    print(f"Total de vetores no banco: {db._collection.count()}")