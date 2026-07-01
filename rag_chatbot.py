# rag_chatbot_gemini.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =====================================================
# 1️⃣ FORCE EVERYTHING TO D DRIVE
# =====================================================

os.environ["HF_HOME"] = "D:\\huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface_cache"

CHROMA_DB_PATH = "D:\\chroma_db"

# =====================================================
# 2️⃣ LOAD GEMINI API
# =====================================================

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Loaded API Key:", bool(api_key))

# =====================================================
# 3️⃣ LOAD SAME EMBEDDING MODEL USED DURING BUILD
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# =====================================================
# 4️⃣ LOAD EXISTING VECTOR STORE (NO ONNX NOW)
# =====================================================

vectordb = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings,
    collection_name="company_docs"
)

print("Vector DB Loaded Successfully ✅")

# =====================================================
# 5️⃣ RAG FUNCTION
# =====================================================

def ask_question(query, department=None):

    if department:
        results = vectordb.similarity_search(
            query,
            k=3,
            filter={"department": department}
        )
    else:
        results = vectordb.similarity_search(query, k=3)

    # Combine retrieved context
    context_text = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
You are a helpful company assistant.

Context:
{context_text}

Question:
{query}

Answer clearly and professionally.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text


# =====================================================
# 6️⃣ TEST
# =====================================================

if __name__ == "__main__":
    question = "What is the company leave policy?"
    answer = ask_question(question, department="HR")

    print("\nQ:", question)
    print("\nA:", answer)