import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configuration
VECTOR_DB_PATH = "./vector_db"
OLLAMA_MODEL = "nemotron-3-nano:30b-cloud"  
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"

app = Flask(__name__)
CORS(app)

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
            num_ctx=4096,
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

# Initialize RAG pipeline
print("Initializing RAG Pipeline...")
rag = LocalRAGPipeline(
    vector_db_path=VECTOR_DB_PATH,
    ollama_model=OLLAMA_MODEL,
    embeddings_model=EMBEDDINGS_MODEL
)

@app.route('/api/query', methods=['POST'])
def query():
    """API endpoint for querying the RAG system"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        result = rag.query(question)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

def main():
    """Run Flask server"""
    print("=" * 60)
    print("Local RAG API Server")
    print("Access the web interface at: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()