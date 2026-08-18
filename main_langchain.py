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
    
class Restaurante(BaseModel):
    cidade:str = Field("A cidade de destino para o roteiro de viagem")
    restaurante:str = Field("Restaureante recomendado na cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurante = JsonOutputParser(pydantic_object=Restaurante)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade de acordo com meu interesse em {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

prompt_restaurante = PromptTemplate(
    template="""
    Sugira um restaurante na cidade de {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurante.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="Sugira um ponto turístico cultural na cidade de {cidade}"
)

modelo = ChatOpenAI(
    model_name="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.5,
    api_key=api_key
)

cadeia_1 = prompt_cidade | modelo | parseador_destino
cadeia_2 = prompt_restaurante | modelo | parseador_restaurante
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

cadeia = (cadeia_1 | cadeia_2 | cadeia_3)

response = cadeia.invoke(
    {
        "interesse": "praias e cultura local"
    }
)
print(response)