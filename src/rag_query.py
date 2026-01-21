import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Configuration ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DB_DIR = "chroma_db"
MODEL = "gemini-2.0-flash"  # or "gemini-1.5-flash" depending on availability

# --- Database Initialization ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Load existing DB
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

# --- Assistant Class ---
class GameOfThronesAssistant:
    def __init__(self, vector_db):
        self.db = vector_db
        # Use the global MODEL variable
        self.model = genai.GenerativeModel(MODEL)
        self.chat_history = []  # State: Stores [{"role": "user", "content": "..."}]
        self.history_limit = 5  # Keep last 5 turns to save tokens

    def _rewrite_query(self, user_query: str) -> str:
        """
        Query Rewriting.
        Converts vague follow-up questions into standalone search queries.
        """
        if not self.chat_history:
            return user_query  # No history, no need to rewrite

        # Condensed history for the prompt
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.chat_history[-self.history_limit:]])

        prompt = f"""
        Given the following conversation history and a new user question,
        rewrite the question to be a standalone search query.
        Replace pronouns (he, she, it) with specific names from the history.

        Chat History:
        {history_text}

        New Question: {user_query}

        Standalone Search Query (Output ONLY the query):
        """

        response = self.model.generate_content(prompt)
        rewritten = response.text.strip()
        print(f"\n🔄 Rewritten Query: '{user_query}' -> '{rewritten}'")  # Debugging
        return rewritten

    def _retrieve_documents(self, query: str):
        """
        Single-Stage Retrieval:
        Directly fetch the top K most similar documents from Vector DB.
        """
        final_k = 5
        docs = self.db.similarity_search(query, k=final_k)
        return docs

    def chat(self, user_query: str):
        # 1. Query Transformation
        search_query = self._rewrite_query(user_query)

        # 2. Retrieval
        context_docs = self._retrieve_documents(search_query)
        context_text = "\n\n".join([d.page_content for d in context_docs])

        # 3. Generation (with Negative Constraints)
        prompt = f"""
        You are a Game of Thrones expert. Use the context below to answer the user's question.

        RULES:
        1. Answer ONLY using the provided Context.
        2. If the answer is not in the context, strictly say "I don't know".
        3. Be concise.

        Context:
        {context_text}

        User Question:
        {user_query}

        Answer:
        """

        response = self.model.generate_content(prompt)
        answer = response.text.strip()

        # 4. Update State
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer, context_docs

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize the assistant with the loaded db
    bot = GameOfThronesAssistant(db)
    
    print("--- Game of Thrones RAG Assistant (Type 'quit' to exit) ---")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        response, docs = bot.chat(user_input)
        
        print(f"Bot: {response}")
        
        # Optional: Print source docs used
        # print("\n[Sources used:]")
        # for d in docs:
        #     print(f"- {d.page_content[:50]}...")
