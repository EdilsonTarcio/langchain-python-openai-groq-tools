# LangChain Playground

Projeto de estudos e experimentação com [LangChain](https://python.langchain.com/), LangGraph e LLMs (OpenAI e Groq), com foco em criação de assistentes de viagem: geração de roteiros, chat com memória, RAG sobre documentos e fluxos com grafos de estado.
Projeto desenvolvido durante meus estudos de LangChain e Python, baseado na formação "Especialista em IA" da Alura subtituindo a utilização dos LLMs da OpenAi pelo GROQ.

## Pré-requisitos

- Python 3.10+
- Uma chave de API da [Groq](https://console.groq.com/) e/ou da [OpenAI](https://platform.openai.com/), dependendo do script que for executar

## Configuração

### 1. Criar e ativar um ambiente virtual

**Linux/Mac:**
```bash
python3 -m venv langchain
source langchain/bin/activate
```

**Windows:**
```bash
python -m venv langchain
langchain\Scripts\activate
```

### 2. Instalar dependências

Na raiz do projeto (scripts `main_groq.py` e `main_langchain.py`):
```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto com as chaves necessárias:
```bash
GROQ_API_KEY="sua_chave_groq"
OPENAI_API_KEY="sua_chave_openai"
```

## Uso

```bash
# Roteiro de viagem via Groq (chamada direta)
python main_groq.py

# Mesmo roteiro, via LangChain
python main_langchain.py

## Stack

- [LangChain](https://python.langchain.com/) / [LangGraph](https://langchain-ai.github.io/langgraph/)
- OpenAI API / Groq API (compatível com a API da OpenAI)
- [FAISS](https://github.com/facebookresearch/faiss) para indexação vetorial (RAG)
- Pydantic para saída estruturada
