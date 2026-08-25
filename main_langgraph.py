import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal, TypedDict
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

def responder_pergunta(pergunta: str):
    destino = roteador.invoke({"query": pergunta})["destino"]
    if destino == "praia":
        resposta = cadeia_praia.invoke({"query": pergunta})
    elif destino == "montanha":
        resposta = cadeia_montanha.invoke({"query": pergunta})
    elif destino == "rios":
        resposta = cadeia_rios.invoke({"query": pergunta})
    else:
        resposta = "Desculpe, não consegui identificar o tipo de destino."
    return resposta

print(responder_pergunta("Quero visitar uma praia com águas cristalinas e atividades de mergulho."))