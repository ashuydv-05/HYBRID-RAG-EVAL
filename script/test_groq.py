import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
base_url = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
)

if not api_key:
    raise ValueError("GROQ_API_KEY is not set")

print("API key found:", api_key[:8] + "...")
print("Model:", model)
print("Base URL:", base_url)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in exactly one sentence."
        }
    ],
)

print("\nGroq response:")
print(response.choices[0].message.content)