import os
from langchain_community.document_loaders import PyPDFLoader

def load_department_docs(folder_path, department_name):
    documents = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata["department"] = department_name
            
            documents.extend(docs)
    
    return documents


# Load all departments
hr_docs = load_department_docs("data/hr", "HR")
it_docs = load_department_docs("data/it", "IT")
finance_docs = load_department_docs("data/finance", "Finance")

all_docs = hr_docs + it_docs + finance_docs

print("Total documents loaded:", len(all_docs))
print("Sample metadata:", all_docs[0].metadata)