# README

## Step 01

Instalação das bibliotecas necessárias para funcionamento da RAG

```shell
pip install python-dotenv langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama pypdf sentence-transformers
```

## Step 02

Instalação do Ollama (local)

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

## Step 03

Instalação do gemma2:2b-instruct-q8_0

```shell
ollama serve && ollama pull gemma2:2b-instruct-q8_0
```