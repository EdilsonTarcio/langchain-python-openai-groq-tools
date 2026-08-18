from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

modelo_de_prompt = PromptTemplate(
    template="""
    Sugira um roteiro para a cidade {cidade} de acordo com meu interesse em {interesse}
    """,
    input_variables=["cidade", "interesse"],
)

modelo = ChatOpenAI(
    model_name="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.5,
    api_key=api_key
)

cadeia = modelo_de_prompt | modelo | StrOutputParser()

response = cadeia.invoke(
    {
        "cidade": "Recife - Pernambuco",
        "interesse": "arte e cultura"
    }
)
print(response)