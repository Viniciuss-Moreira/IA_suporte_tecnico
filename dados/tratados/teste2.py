import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # Economiza em custos de API
from langchain.schema import Document
import json
from tqdm import tqdm

# 1. Usar embeddings locais para economizar em custos de API
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Modelo leve, bom para M2
    model_kwargs={'device': 'mps'}  # Utiliza a GPU do M2
)

# 2. Configurar cliente Chroma com persistência
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("suporte_tecnico")

# 3. Processar e inserir documentos em lotes (para evitar sobrecarga de memória)
batch_size = 500
all_documents = []

# Carregar dados
with open('dataset.jsonl', 'r') as f:
    json_lines = f.readlines()

for i in tqdm(range(0, len(json_lines), batch_size)):
    batch = json_lines[i:i+batch_size]
    
    documents = []
    ids = []
    metadatas = []
    
    for j, line in enumerate(batch):
        item = json.loads(line)
        text = item["text"]
        
        # Extrair pergunta e resposta
        parts = text.split("\nResposta: ")
        question = parts[0].replace("Pergunta ", "")
        answer = parts[1] if len(parts) > 1 else ""
        
        # Extrair nível de dificuldade
        difficulty = "desconhecido"
        if "(" in question and ")" in question:
            difficulty = question.split("(")[1].split(")")[0]
        
        # Preparar documento
        documents.append(text)
        ids.append(f"doc_{i+j}")
        metadatas.append({
            "difficulty": difficulty,
            "question": question,
            "answer": answer
        })
    
    # Adicionar lote ao Chroma
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    # Liberar memória
    del documents, ids, metadatas