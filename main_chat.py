import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

modelo = ChatOpenAI(
    model_name="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.5,
    api_key=api_key
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente de viagem que sugere cidades, restaurantes e pontos turísticos culturais. Apresente-se como Sr. Viagem"),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "aula_langchain" #identificador da sessão de chat (ID dousuario)

def historico_da_sessao(sessao : str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

lista_perguntas = [
    "Quero visitar uma lugar em Recife que tenha um bom Lamen.",
    "E qual o mais recomendado?"
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_da_sessao,
    input_message_key="query",
    history_message_key="historico"
)

for uma_pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "query": uma_pergunta,
        },
        config={"session_id": sessao}
    )
    print("Usuario: ", uma_pergunta)
    print("Assistente: ", resposta, "\n")