from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

numero_dias = 3
destino = "Gaibu - Pernambuco"

prompt = f"Criar um roteiro de viagem de {numero_dias} dias para {destino}, incluindo sugestões de atividades, restaurantes e pontos turísticos."

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "Você é um assistente de viagem especializado em criar roteiros personalizados."},
        {"role": "user", "content": prompt}
    ])

print(response.choices[0].message.content)