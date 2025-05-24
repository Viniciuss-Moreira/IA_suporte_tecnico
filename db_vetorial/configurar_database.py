import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import json
from tqdm import tqdm
import os
import sys

if len(sys.argv) < 2:
    print("uso: python script.py /caminho/para/dataset_artificial.jsonl")
    sys.exit(1)

caminho_dataset = sys.argv[1]

if not os.path.isfile(caminho_dataset):
    print(f"não encontrado: {caminho_dataset}")
    sys.exit(1)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'mps'}
)

client = chromadb.PersistentClient(path="./chroma_db")
existing_collections = [c.name for c in client.list_collections()]
if "suporte_tecnico" not in existing_collections:
    collection = client.create_collection("suporte_tecnico")
else:
    collection = client.get_collection("suporte_tecnico")


batch_size = 500
all_documents = []

with open(caminho_dataset, 'r') as f:
    json_lines = f.readlines()

for i in tqdm(range(0, len(json_lines), batch_size)):
    batch = json_lines[i:i+batch_size]
    
    documents = []
    ids = []
    metadatas = []
    
    for j, line in enumerate(batch):
        item = json.loads(line)
        text = item["text"]
        
        parts = text.split("\nResposta: ")
        question = parts[0].replace("Pergunta ", "")
        answer = parts[1] if len(parts) > 1 else ""
        
        difficulty = "desconhecido"
        if "(" in question and ")" in question:
            difficulty = question.split("(")[1].split(")")[0]
        
        documents.append(text)
        ids.append(f"doc_{i+j}")
        metadatas.append({
            "difficulty": difficulty,
            "question": question,
            "answer": answer
        })
    
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    del documents, ids, metadatas