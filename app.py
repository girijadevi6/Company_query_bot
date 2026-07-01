# app.py

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# 0️⃣ Load Environment
# ----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ----------------------------
# 1️⃣ Initialize Embeddings
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ----------------------------
# 2️⃣ Load Existing Chroma DB
# ----------------------------
vectordb = Chroma(
    collection_name="company_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

# ----------------------------
# 3️⃣ Initialize Gemini
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)

# ----------------------------
# 4️⃣ Streamlit UI
# ----------------------------
st.set_page_config(page_title="💼 Company Knowledge Chatbot", layout="wide")

st.sidebar.title("💼 Company Chatbot")
department = st.sidebar.selectbox("Select Department", ["All", "HR", "IT", "Finance"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_history = []

if st.session_state.chat_history:
    chat_text = "\n".join(
        [f"User: {c['user']}\nBot: {c['bot']}" for c in st.session_state.chat_history]
    )
    st.sidebar.download_button("💾 Download Chat", chat_text, "chat.txt")

# ----------------------------
# 5️⃣ RAG Function (Manual)
# ----------------------------
def ask_question(query, selected_department):

    # Apply department filter
    if selected_department != "All":
        docs = vectordb.similarity_search(
            query,
            k=3,
            filter={"department": selected_department}
        )
    else:
        docs = vectordb.similarity_search(query, k=3)

    # Combine retrieved text
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful company assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and professionally.
"""

    response = llm.invoke(prompt)
    return response.content

# ----------------------------
# 6️⃣ User Input
# ----------------------------
user_input = st.text_input("Type your question here:")

if user_input:
    answer = ask_question(user_input, department)

    st.session_state.chat_history.append({
        "user": user_input,
        "bot": answer,
        "department": department
    })

# ----------------------------
# 7️⃣ Display Chat
# ----------------------------
for chat in st.session_state.chat_history:
    st.chat_message("user").write(chat["user"])
    st.chat_message("assistant").write(
        f"{chat['bot']}\n\n_(Department used: {chat['department']})_"
    )