from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

modelo_de_prompt = PromptTemplate(
    template="""
    Você é um assistente de viagem especializado em criar roteiros personalizados.
    Criar um roteiro de viagem de {numero_dias} dias para {destino},
    incluindo sugestões de atividades, restaurantes e pontos turísticos.
    """
)

prompt = modelo_de_prompt.format(
    numero_dias=3,
    destino="Gaibu - Pernambuco"
)

modelo = ChatOpenAI(
    model_name="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.5,
    api_key=api_key
)

response = modelo.invoke(prompt)
print(response.content)