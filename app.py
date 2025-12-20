import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration
VECTOR_DB_PATH = "./vector_db"
OLLAMA_MODEL = "gemini-3-pro-preview"  
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"  

class LocalRAGPipeline:
    def __init__(self, vector_db_path, ollama_model, embeddings_model=EMBEDDINGS_MODEL):
        """
        Initialize the local RAG pipeline with Ollama and vector DB
        
        Args:
            vector_db_path: Path to FAISS vector database
            ollama_model: Ollama model name (e.g., "mistral", "llama2", "neural-chat")
            embeddings_model: HuggingFace embeddings model
        """
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        
        print("Loading vector database...")
        self.vector_store = FAISS.load_local(
            vector_db_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        print(f"Initializing Ollama LLM: {ollama_model}")
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            num_ctx=4096,  # Context window size
        )
        
        print("Building retrieval chain...")
        self.chain = self._build_chain()
        
        print("Local RAG Pipeline ready!\n")
    
    def _build_chain(self):
        """Build RetrievalQA chain with custom prompt"""
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return chain
    
    def query(self, question):
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            dict with 'answer' and 'sources'
        """
        result = self.chain({"query": question})
        
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        source_info = []
        for doc in sources:
            source_info.append({
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "text": doc.page_content[:200] + "..."
            })
        
        return {
            "answer": answer,
            "sources": source_info
        }


def main():
    """Initialize pipeline and run interactive chat"""
    
    # Initialize RAG pipeline
    rag = LocalRAGPipeline(
        vector_db_path=VECTOR_DB_PATH,
        ollama_model=OLLAMA_MODEL,
        embeddings_model=EMBEDDINGS_MODEL
    )
    
    # Interactive chat loop
    print("=" * 60)
    print("Local RAG Chatbot (Ollama)")
    print("Type 'exit' to quit")
    print("=" * 60 + "\n")
    
    while True:
        question = input("Q: ").strip()
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nProcessing...\n")
        result = rag.query(question)
        
        print(f"A: {result['answer']}\n")
        print("Sources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source['file']} (Page {source['page']})")
            print(f"   {source['text']}\n")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()