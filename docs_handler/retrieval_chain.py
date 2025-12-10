import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

import sys
sys.path.append("/kaggle/working/llm_rag")

# Configuration
VECTOR_DB_PATH = "/kaggle/working/llm_rag/vector_db"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

class RAGPipeline:
    def __init__(self, vector_db_path, model_name, device="cuda"):
        """
        Initialize the RAG pipeline with vector DB and LLM
        
        Args:
            vector_db_path: Path to FAISS vector database
            model_name: HuggingFace model identifier
            device: "cuda" for GPU or "cpu" for CPU
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        
        print("Loading vector database...")
        self.vector_store = FAISS.load_local(
            vector_db_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        print(f"Loading LLM: {model_name}")
        self.llm = self._load_llm(model_name)
        
        print("Building retrieval chain...")
        self.chain = self._build_chain()
        
        print("RAG Pipeline ready!\n")
    
    def _load_llm(self, model_name):
        """Load Mistral model with optimizations"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        model = model.to(self.device)
        
        # Create text generation pipeline
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            device=0 if self.device == "cuda" else -1,  # -1 for CPU
        )
        
        return HuggingFacePipeline(pipeline=text_gen_pipeline)
    
    def _build_chain(self):
        """Build RetrievalQA chain with custom prompt"""
        
        # Custom prompt template for better results
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
        
        # Extract answer and sources
        answer = result["result"]
        sources = result.get("source_documents", [])
        
        # Format sources
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
    """Initialize pipeline and run test queries"""
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        vector_db_path=VECTOR_DB_PATH,
        model_name=MODEL_NAME,
        device="cuda"  # Change to "cpu" if no GPU
    )
    
    # Test queries
    test_queries = [
        "What are the main topics covered in these documents?",
        "Explain the key concepts in mutioutput regression",
        "What is multioutput regression ?"
    ]
    
    for question in test_queries:
        print("=" * 60)
        print(f"Q: {question}")
        print("=" * 60)
        
        result = rag.query(question)
        
        print(f"\nA: {result['answer']}\n")
        print("Sources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source['file']} (Page {source['page']})")
            print(f"   {source['text']}\n")


if __name__ == "__main__":
    main()