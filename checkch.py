from chromadb import Client
from chromadb.config import Settings

client = Client(Settings(persist_directory="./chroma_db"))
collection = client.get_collection(name="company_docs")

print("Total documents in collection:", len(collection.get()["ids"]))