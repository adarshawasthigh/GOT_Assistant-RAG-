import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DB_DIR = "chroma_db"
K = 5
MODEL = "gemini-2.5-flash"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

def rag(query: str):
    docs = db.similarity_search(query, k=K)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer the question ONLY using the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    model = genai.GenerativeModel(MODEL)
    return model.generate_content(prompt).text

if __name__ == "__main__":
    print(rag("Who was the brother of Daenerys Targaryen?"))
