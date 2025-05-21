"""
Implementação completa de um sistema RAG (Retrieval Augmented Generation)
usando Chroma DB para um dataset de suporte técnico com 75.000 exemplos.

Este script:
1. Configura o ambiente e dependências
2. Carrega e processa o dataset
3. Cria e configura o banco de dados vetorial Chroma
4. Implementa funções de consulta para buscar respostas relevantes
5. Cria uma interface simples para testes

Autor: Claude
Data: Maio 2025
"""

import os
import json
import time
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Optional

# Verificar instalação das dependências
try:
    import chromadb
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
except ImportError:
    print("Instalando dependências necessárias...")
    os.system("pip install chromadb langchain sentence-transformers tqdm openai")
    # Reimportar após instalação
    import chromadb
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

# Configurações do sistema
DATASET_PATH = "dataset.jsonl"  # Caminho para o arquivo JSONL
CHROMA_PATH = "./chroma_db"     # Diretório para persistência do Chroma
BATCH_SIZE = 500                # Tamanho dos lotes para processamento
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Modelo de embedding

class SuporteTecnicoRAG:
    """Sistema RAG para suporte técnico usando Chroma DB"""
    
    def __init__(self, dataset_path: str, db_path: str, embedding_model: str):
        """Inicializa o sistema RAG
        
        Args:
            dataset_path: Caminho para o arquivo JSONL
            db_path: Diretório para persistência do Chroma
            embedding_model: Nome do modelo HuggingFace para embeddings
        """
        self.dataset_path = dataset_path
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.embeddings = None
        self.db = None
        self.collection = None
        
    def setup_embeddings(self):
        """Configura o modelo de embeddings"""
        print("Configurando modelo de embeddings...")
        
        # Detectar tipo de dispositivo para otimização (MPS para M1/M2 Macs)
        device = "mps" if hasattr(torch, "has_mps") and torch.has_mps else "cpu"
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": device}
        )
        print(f"Modelo de embeddings configurado: {self.embedding_model} em {device}")
        
    def setup_chroma_client(self, collection_name="suporte_tecnico"):
        """Configura o cliente Chroma com persistência"""
        print(f"Configurando cliente Chroma em {self.db_path}...")
        
        # Criar diretório para persistência se não existir
        os.makedirs(self.db_path, exist_ok=True)
        
        # Configurar cliente persistente
        client = chromadb.PersistentClient(path=self.db_path)
        
        # Verificar se a coleção já existe, se não, criar
        try:
            self.collection = client.get_collection(collection_name)
            print(f"Coleção existente '{collection_name}' carregada.")
        except:
            self.collection = client.create_collection(collection_name)
            print(f"Nova coleção '{collection_name}' criada.")
    
    def count_documents(self):
        """Conta o número de documentos na coleção"""
        return self.collection.count()
        
    def process_and_load_data(self, force_reload=False):
        """Processa e carrega os dados do dataset para o Chroma
        
        Args:
            force_reload: Se True, recarrega mesmo se já existirem documentos
        """
        # Verificar se já existem documentos carregados
        if not force_reload and self.count_documents() > 0:
            print(f"Já existem {self.count_documents()} documentos carregados. Pulando carregamento.")
            return
            
        print(f"Processando dataset de {self.dataset_path}...")
        
        # Verificar se o arquivo existe
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Arquivo {self.dataset_path} não encontrado")
            
        # Carregar linhas do arquivo JSONL
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            json_lines = f.readlines()
            
        total_lines = len(json_lines)
        print(f"Processando {total_lines} exemplos em lotes de {BATCH_SIZE}...")
        
        # Processar em lotes para evitar sobrecarga de memória
        for i in tqdm(range(0, total_lines, BATCH_SIZE)):
            batch = json_lines[i:i+BATCH_SIZE]
            
            documents = []
            ids = []
            metadatas = []
            
            for j, line in enumerate(batch):
                try:
                    item = json.loads(line)
                    text = item["text"]
                    
                    # Extrair componentes do texto
                    parts = text.split(". Pergunta ")
                    ambiente = parts[0].replace("Ambiente: ", "")
                    
                    resto = parts[1].split("Resposta: ")
                    pergunta_completa = resto[0]
                    resposta = resto[1]
                    
                    # Extrair detalhes da pergunta
                    pergunta_partes = pergunta_completa.split("? ")
                    pergunta_principal = pergunta_partes[0] + "?"
                    
                    # Extrair nível de dificuldade
                    difficulty = "desconhecido"
                    if "(" in pergunta_principal and ")" in pergunta_principal:
                        difficulty = pergunta_principal.split("(")[1].split(")")[0]
                        pergunta_limpa = pergunta_principal.split("(")[0].strip() + "?"
                    else:
                        pergunta_limpa = pergunta_principal
                    
                    # Extrair mensagem de erro e sintomas
                    msg_erro = ""
                    sintomas = ""
                    for parte in pergunta_partes[1:]:
                        if parte.startswith("Mensagem de erro:"):
                            msg_erro = parte.replace("Mensagem de erro: ", "")
                        elif parte.startswith("Sintomas:"):
                            sintomas = parte.replace("Sintomas: ", "")
                    
                    # Preparar para inserção no Chroma
                    doc_id = f"doc_{i+j}"
                    
                    # Identificar sistema operacional básico
                    sistema = ambiente.split()[0].lower()  # windows, macos, linux
                    
                    # Metadados para filtragem e recuperação
                    metadata = {
                        "ambiente": ambiente,
                        "sistema": sistema,
                        "dificuldade": difficulty,
                        "pergunta": pergunta_limpa,
                        "mensagem_erro": msg_erro,
                        "sintomas": sintomas,
                        "resposta": resposta
                    }
                    
                    # Adicionar à lista do lote
                    documents.append(text)
                    ids.append(doc_id)
                    metadatas.append(metadata)
                    
                except json.JSONDecodeError:
                    print(f"Erro ao processar linha {i+j}: formato JSON inválido")
                except Exception as e:
                    print(f"Erro ao processar linha {i+j}: {str(e)}")
            
            # Adicionar lote ao Chroma
            if documents:
                self.collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas
                )
            
            # Liberar memória
            del documents, ids, metadatas
            
        print(f"Carregamento concluído. Total de {self.count_documents()} documentos indexados.")
    
    def setup_langchain_retriever(self):
        """Configura o retriever do LangChain para integração com LLMs"""
        # Configurar embeddings para LangChain se ainda não estiver configurado
        if self.embeddings is None:
            self.setup_embeddings()
            
        # Configurar vectorstore do LangChain
        self.db = Chroma(
            client=self.collection.client,
            collection_name=self.collection.name,
            embedding_function=self.embeddings
        )
        
        print("Retriever do LangChain configurado com sucesso.")
        
    def buscar_respostas(self, query: str, filtros: Optional[Dict[str, str]] = None, k: int = 3):
        """Busca respostas relevantes para uma pergunta
        
        Args:
            query: Pergunta do usuário
            filtros: Dicionário com filtros a serem aplicados (sistema, dificuldade, etc)
            k: Número de documentos a serem retornados
            
        Returns:
            Lista de documentos relevantes
        """
        # Preparar filtro no formato esperado pelo Chroma
        where = {}
        if filtros:
            for key, value in filtros.items():
                if value:  # Ignorar valores vazios
                    where[key] = value
                    
        # Converter filtro para where_document se necessário
        where_document = {}
        if where:
            where_document = {"$and": [{"metadatas": {k: v}} for k, v in where.items()]}
            
        # Realizar consulta
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_document if where_document else None
        )
        
        # Organizar resultados
        documentos = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            documentos.append({
                "texto": doc,
                "metadata": metadata,
                "distancia": results['distances'][0][i] if 'distances' in results else None
            })
            
        return documentos
    
    def consultar_com_llm(self, query: str, sistema: Optional[str] = None, 
                         api_key: Optional[str] = None, modelo: str = "gpt-3.5-turbo"):
        """Consulta usando LLM com resultados do Chroma como contexto
        
        Args:
            query: Pergunta do usuário
            sistema: Sistema operacional (windows, macos, linux) para filtrar
            api_key: API key do provedor LLM
            modelo: Nome do modelo LLM a ser usado
            
        Returns:
            Resposta gerada pelo LLM
        """
        # Verificar API key
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API key não fornecida e não encontrada nas variáveis de ambiente")
        
        # Configurar filtros
        filtros = {"sistema": sistema.lower()} if sistema else {}
        
        # Buscar documentos relevantes
        documentos = self.buscar_respostas(query, filtros=filtros, k=3)
        
        # Criar contexto
        contexto = ""
        for i, doc in enumerate(documentos):
            contexto += f"Documento {i+1}:\n{doc['texto']}\n\n"
        
        # Template do prompt
        prompt_template = """
        Você é um especialista em suporte técnico. Com base na pergunta do usuário 
        e nas informações dos documentos, forneça uma resposta clara e útil.
        
        Contexto:
        {contexto}
        
        Pergunta do usuário: {query}
        
        Sua resposta deve:
        1. Considerar o sistema operacional mencionado (se houver)
        2. Priorizar as soluções mais relevantes para o problema específico
        3. Fornecer passos claros e simples
        4. Incluir uma solução alternativa se a primeira não funcionar
        
        Resposta:
        """
        
        # Configurar LLM
        llm = ChatOpenAI(temperature=0.2, model=modelo, api_key=api_key)
        
        # Criar prompt com contexto e pergunta
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Gerar resposta
        chain = prompt | llm
        response = chain.invoke({"contexto": contexto, "query": query})
        
        return response.content

def iniciar_sistema_rag():
    """Inicializa o sistema RAG completo"""
    # Verificar dependências extras
    try:
        import torch
    except ImportError:
        print("Instalando PyTorch...")
        os.system("pip install torch")
        import torch
    
    print("Iniciando sistema RAG para suporte técnico...")
    start_time = time.time()
    
    # Criar instância do sistema
    rag = SuporteTecnicoRAG(
        dataset_path=DATASET_PATH,
        db_path=CHROMA_PATH,
        embedding_model=EMBEDDING_MODEL
    )
    
    # Configurar componentes
    rag.setup_embeddings()
    rag.setup_chroma_client()
    
    # Carregar dados
    rag.process_and_load_data()
    
    # Configurar retriever
    rag.setup_langchain_retriever()
    
    elapsed = time.time() - start_time
    print(f"Sistema RAG inicializado em {elapsed:.2f} segundos!")
    
    return rag

def exemplo_uso_simples():
    """Exemplo de uso simples do sistema RAG"""
    rag = iniciar_sistema_rag()
    
    # Exemplo de busca simples
    query = "Minha tela está piscando quando abro aplicativos no Windows"
    resultados = rag.buscar_respostas(query, filtros={"sistema": "windows"}, k=3)
    
    print("\n=== Resultados da Busca ===")
    for i, doc in enumerate(resultados):
        print(f"\nDocumento {i+1}:")
        print(f"Texto: {doc['texto'][:200]}...")
        print(f"Sistema: {doc['metadata']['sistema']}")
        print(f"Dificuldade: {doc['metadata']['dificuldade']}")
        print(f"Resposta: {doc['metadata']['resposta']}")
    
    # Exemplo com LLM (necessita API key)
    if os.environ.get("OPENAI_API_KEY"):
        resposta = rag.consultar_com_llm(query, sistema="windows")
        print("\n=== Resposta Gerada pelo LLM ===")
        print(resposta)
    else:
        print("\nPara usar o LLM, defina a variável de ambiente OPENAI_API_KEY")

def interface_simples():
    """Interface de linha de comando simples para teste do sistema RAG"""
    rag = iniciar_sistema_rag()
    
    print("\n===== Sistema de Suporte Técnico RAG =====")
    print("Digite 'sair' para encerrar")
    
    while True:
        query = input("\nDigite sua pergunta: ")
        if query.lower() == 'sair':
            break
            
        sistema = input("Sistema operacional (windows/macos/linux) ou enter para ignorar: ")
        sistema = sistema.lower() if sistema else None
        
        # Usar apenas busca sem LLM
        resultados = rag.buscar_respostas(query, filtros={"sistema": sistema} if sistema else {}, k=3)
        
        print("\n=== Resultados Mais Relevantes ===")
        for i, doc in enumerate(resultados):
            print(f"\nDocumento {i+1}:")
            print(f"Ambiente: {doc['metadata']['ambiente']}")
            print(f"Problema: {doc['metadata']['pergunta']}")
            if doc['metadata']['mensagem_erro']:
                print(f"Erro: {doc['metadata']['mensagem_erro']}")
            if doc['metadata']['sintomas']:
                print(f"Sintomas: {doc['metadata']['sintomas']}")
            print(f"Resposta: {doc['metadata']['resposta']}")
        
        # Perguntar se quer usar LLM
        usar_llm = input("\nGerar resposta com IA? (s/n): ")
        if usar_llm.lower() == 's':
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                api_key = input("Digite sua API key do OpenAI (ou configure OPENAI_API_KEY): ")
                
            if api_key:
                try:
                    resposta = rag.consultar_com_llm(query, sistema=sistema, api_key=api_key)
                    print("\n=== Resposta Gerada pela IA ===")
                    print(resposta)
                except Exception as e:
                    print(f"Erro ao gerar resposta: {str(e)}")
            else:
                print("API key não fornecida. Usando apenas resultados de busca.")

if __name__ == "__main__":
    # Verificar se o usuário quer interface de linha de comando
    print("1. Executar exemplo simples")
    print("2. Iniciar interface de linha de comando")
    opcao = input("Escolha uma opção (1/2): ")
    
    if opcao == "1":
        exemplo_uso_simples()
    else:
        interface_simples()