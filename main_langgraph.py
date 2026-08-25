import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

modelo = ChatOpenAI(
    model_name="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.5,
    api_key=api_key
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um especialista em viagens com destinos para paraia, Apresente-se como Sr Praia"),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um especialista em viagens com destinos para montanhas e atividades rádicais, Apresente-se como Sr Montanha"),
        ("human", "{query}")
    ]
)

class Rota(TypedDict):
    destino: Literal["praia", "montanha", "rios"]

prompt_consultor_rios = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um especialista em viagens com destinos para rios, Apresente-se como Sr Rios"),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()
cadeia_rios = prompt_consultor_rios | modelo | StrOutputParser()

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia', 'montanha' ou 'rios' "),
        ("human", "{query}")
    ]
)  

roteador = prompt_roteador | modelo.with_structured_output(Rota)

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str
    
async def no_roteador(estado: Estado, config: RunnableConfig):
    return {"destino": await roteador.ainvoke({"query": estado["query"]}, config)}

async def no_praia(estado: Estado, config: RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}

async def no_montanha(estado: Estado, config: RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}

async def no_rios(estado: Estado, config: RunnableConfig):
    return {"resposta": await cadeia_rios.ainvoke({"query": estado["query"]}, config)}

def escolher_no(estado:Estado)->Literal["praia", "montanha", "rios"]:
    if estado["destino"]["destino"] == "praia":
        return "praia"
    elif estado["destino"]["destino"] == "montanha":
        return "montanha"
    elif estado["destino"]["destino"] == "rios":
        return "rios"
    else:
        raise ValueError("Destino inválido")
    
grafo = StateGraph(Estado)
grafo.add_node("roteador", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)
grafo.add_node("rios", no_rios)

grafo.add_edge(START, "roteador")
grafo.add_conditional_edges("roteador", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)
grafo.add_edge("rios", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {"query": "Quero viajar para um lugar com praia e sol."}
    )
    print(resposta["resposta"])
    
asyncio.run(main())