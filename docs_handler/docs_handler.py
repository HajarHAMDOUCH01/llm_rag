import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

import sys
sys.path.append("/kaggle/working/llm_rag")

# Configuration
PDF_FOLDER = "/kaggle/working/llm_rag/pdfs"  # Folder where your PDFs are stored
VECTOR_DB_PATH = "/kaggle/working/llm_rag/vector_db"  # Where to save the vector database
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast embedding model

def load_pdfs(pdf_folder):
    """
    Load all PDFs from a folder
    Returns: List of Document objects
    """
    documents = []
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {pdf_folder}")
        return documents
    
    print(f"Found {len(pdf_files)} PDF(s). Loading...")
    
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            # Add filename to metadata for reference
            for doc in docs:
                doc.metadata["source"] = pdf_path.name
            documents.extend(docs)
        except Exception as e:
            print(f"  Error loading {pdf_path.name}: {e}")
    
    print(f"Loaded {len(documents)} pages total\n")
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into manageable chunks
    Returns: List of chunked Document objects
    """
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks\n")
    return chunks

def create_embeddings_and_vector_db(chunks, vector_db_path):
    """
    Create embeddings and store in FAISS vector database
    Returns: FAISS vector store object
    """
    print("Creating embeddings (this may take a moment)...")
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    # Create vector store from chunks
    print("Building vector database...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store locally
    vector_store.save_local(vector_db_path)
    print(f"Vector database saved to {vector_db_path}\n")
    
    return vector_store

def load_vector_db(vector_db_path):
    """
    Load existing vector database from disk
    Returns: FAISS vector store object
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def setup_pipeline(pdf_folder, vector_db_path):
    """
    Main setup pipeline: Load PDFs -> Chunk -> Create Vector DB
    """
    print("=" * 50)
    print("PDF RAG Setup Phase")
    print("=" * 50 + "\n")
    
    # Create PDF folder if it doesn't exist
    os.makedirs(pdf_folder, exist_ok=True)
    
    # Check if vector DB already exists
    if os.path.exists(vector_db_path):
        print(f"Vector database found at {vector_db_path}")
        response = input("Recreate it? (yes/no): ").strip().lower()
        if response != "yes":
            print("Loading existing vector database...\n")
            return load_vector_db(vector_db_path)
    
    # Step 1: Load PDFs
    documents = load_pdfs(pdf_folder)
    
    if not documents:
        print("No documents to process. Add PDFs to the pdfs/ folder.")
        return None
    
    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    
    # Step 3: Create embeddings and vector DB
    vector_store = create_embeddings_and_vector_db(chunks, vector_db_path)
    
    print("=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    
    return vector_store

# Test the pipeline
if __name__ == "__main__":
    vector_store = setup_pipeline(PDF_FOLDER, VECTOR_DB_PATH)
    
    if vector_store:
        # Quick test: search for something
        query = "What is this documents about?"
        results = vector_store.similarity_search(query, k=3)
        print(f"\nTest search for '{query}':")
        print(f"Found {len(results)} results")
        if results:
            print(f"Top result: {results[0].page_content}")

