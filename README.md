# Agent VW T-Cross

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

Iniciando o serviço do ollama

```shell
ollama serve
```

## Step 04

Instalação do gemma2:2b-instruct-q8_0

```shell
ollama pull gemma2:2b-instruct-q8_0
```

## Step 05

```shell
python main.py
```

## Step 06

```shell
python app/chatbot.py
```

## Step 07

```shell
Você: Qual o consumo do T-Cross na estrada?
```