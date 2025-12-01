# chat_tcross.py — Agente VW T-Cross com cronômetro em tempo real

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import time
import threading

# Configurações
VECTOR_DB_PATH = "data/curated/t-cross_index/"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

# Carrega embeddings e vetorstore
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 6})

# LLM local
llm = ChatOllama(
    model="gemma2:2b-instruct-q8_0",   # ou llama3.2:3b, phi3:mini, etc.
    temperature=0.3,
)

# Prompt
template = """Você é um assistente especializado na Volkswagen T-Cross 2024/2025.
Responda apenas com base nos documentos oficiais da VW que eu te forneci.
Se não souber ou não encontrar nos documentos, diga honestamente: "Não tenho essa informação nos manuais oficiais".

Use linguagem clara, objetiva e típica de vendedor/consultor de concessionária.
Se for valor, consumo, motor, sempre cite a fonte (ex: página do manual).

Contexto dos documentos:
{context}

Pergunta do cliente: {question}

Resposta:"""

prompt = ChatPromptTemplate.from_template(template)

# Chain RAG
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ================================================
# FUNÇÃO COM CRONÔMETRO ANIMADO BONITINHO
# ================================================
def cronometro_animado(parar_event):
    """Mostra um cronômetro girando enquanto o modelo pensa"""
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    start = time.time()
    while not parar_event.is_set():
        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        timer = f"{mins:02d}:{secs:02d}"
        print(f"\r{spinner[i % len(spinner)]} Pensando... {timer}s", end="", flush=True)
        i += 1
        time.sleep(0.1)
    # Limpa a linha ao finalizar
    print("\r" + " " * 50 + "\r", end="")

# ================================================
# LOOP PRINCIPAL
# ================================================
print("Agente Volkswagen T-Cross ONLINE (100% local)")
print("Digite 'sair' para encerrar\n")

while True:
    pergunta = input("Você: ").strip()
    
    if pergunta.lower() in ["sair", "exit", "tchau", "fim"]:
        print("Até logo!")
        break
    
    if not pergunta:
        continue

    # Inicia o cronômetro em uma thread separada
    parar_cronometro = threading.Event()
    thread_cronometro = threading.Thread(target=cronometro_animado, args=(parar_cronometro,))
    thread_cronometro.daemon = True
    thread_cronometro.start()

    try:
        inicio = time.time()
        resposta = chain.invoke(pergunta)
        tempo_total = time.time() - inicio
        
        # Para o cronômetro
        parar_cronometro.set()
        thread_cronometro.join(timeout=1)

        # Mostra resposta com tempo formatado
        print(f"\nT-Cross: {resposta}")
        print(f"Tempo de resposta: {tempo_total:.2f} segundos\n")
        
    except Exception as e:
        parar_cronometro.set()
        thread_cronometro.join(timeout=1)
        print(f"\nErro: {e}")
        print("Tente novamente ou reinicie o Ollama.\n")