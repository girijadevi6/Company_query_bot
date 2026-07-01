from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from load_documents import all_docs  # already a list of Document objects

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_chunks = text_splitter.split_documents(all_docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build Chroma vector store
vectordb = Chroma.from_documents(
    documents=docs_chunks,
    embedding=embeddings,
    collection_name="company_docs",
    persist_directory="./chroma_db"
)

print("Vector store created and saved locally ✅")