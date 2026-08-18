from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.globals import set_debug
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

set_debug(True)

class Destino(BaseModel):
    cidade:str = Field("A cidade de destino para o roteiro de viagem")
    motivo:str = Field("Motivo pelo qual é interessante visitar a cidade")

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_de_prompt = PromptTemplate(
    template="""
    Sugira uma cidade de acordo com meu interesse em {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()}
)

modelo = ChatOpenAI(
    model_name="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.5,
    api_key=api_key
)

cadeia = modelo_de_prompt | modelo | parseador

response = cadeia.invoke(
    {
        "interesse": "praias e cultura local"
    }
)
print(response)