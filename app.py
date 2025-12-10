import streamlit as st
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# Configuration
VECTOR_DB_PATH = "./vector_db"
PDF_FOLDER = "./pdfs"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# Page config
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        display: flex;
        gap: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .source-box {
        background-color: #fff9c4;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        border-left: 3px solid #FBC02D;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline once and cache it"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("Loading embeddings model..."):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    with st.spinner("Loading vector database..."):
        if not os.path.exists(VECTOR_DB_PATH):
            st.error(f"Vector database not found at {VECTOR_DB_PATH}")
            st.stop()
        
        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    with st.spinner("Loading Mistral-7B model (this may take a moment)..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "cpu":
            model = model.to("cpu")
        
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            device=0 if device == "cuda" else -1,
        )
        
        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    
    # Build chain
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
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

def process_query(question):
    """Process user question through RAG pipeline"""
    chain = load_rag_pipeline()
    result = chain.invoke({"query": question})
    
    return {
        "answer": result["result"],
        "sources": result.get("source_documents", [])
    }

def rebuild_vector_db():
    """Rebuild vector database from PDFs"""
    st.info("Starting vector database rebuild...")
    
    # Load PDFs
    documents = []
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    progress_bar = st.progress(0)
    for idx, pdf_path in enumerate(pdf_files):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_path.name
            documents.extend(docs)
            progress_bar.progress((idx + 1) / len(pdf_files))
        except Exception as e:
            st.error(f"Error loading {pdf_path.name}: {e}")
    
    if not documents:
        st.error("No documents loaded!")
        return
    
    # Chunk documents
    with st.spinner("Chunking documents..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
    
    # Create vector store
    with st.spinner("Creating embeddings and vector database..."):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(VECTOR_DB_PATH)
    
    st.success("Vector database rebuilt successfully!")
    # Clear cache to reload the pipeline
    st.cache_resource.clear()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Rebuild Vector DB", use_container_width=True):
            rebuild_vector_db()
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    st.subheader("ℹ️ About")
    st.write("""
    This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions 
    based on your PDF documents.
    
    **Model**: Mistral-7B-Instruct  
    **Embeddings**: All-MiniLM-L6-v2  
    **Vector DB**: FAISS
    """)

# Main chat interface
st.title("📚 PDF RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div style="flex: 1;">
                <strong>You</strong><br/>
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if message.get("sources"):
            sources_html = "<div class='source-box'><strong>📄 Sources:</strong><br/>"
            for i, source in enumerate(message["sources"], 1):
                sources_html += f"{i}. <strong>{source['file']}</strong> (Page {source['page']})<br/>"
            sources_html += "</div>"
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div style="flex: 1;">
                <strong>Assistant</strong><br/>
                {message["content"]}
                {sources_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
col1, col2 = st.columns([0.9, 0.1])

with col1:
    user_input = st.text_input(
        "Ask a question about your documents...",
        placeholder="What is multi-output regression?",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", use_container_width=True, key="send")

# Process user input
if send_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Get response from RAG pipeline
    with st.spinner("Thinking..."):
        result = process_query(user_input)
        
        # Format sources
        sources = []
        for doc in result["sources"]:
            sources.append({
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            })
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": sources
        })
    
    # Rerun to display new message
    st.rerun()