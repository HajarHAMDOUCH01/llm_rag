import streamlit as st
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from langchain_huggingface import HuggingFaceEndpoint

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
        background-color: black;
        border-left: 4px solid blue;
    }
    .assistant-message {
        background-color: black;
        border-left: 4px solid blue;
    }
    .source-box {
        background-color: black;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        border-left: 3px solid blue;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline using Hugging Face Inference API"""
    
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        st.error("❌ HF_TOKEN environment variable not set.")
        st.info("📋 Get a token from: https://huggingface.co/settings/tokens")
        st.info("📝 On Streamlit Cloud: Manage App → Secrets → Add HF_TOKEN")
        st.stop()
    
    with st.spinner("Loading embeddings model..."):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    with st.spinner("Loading vector database..."):
        if not os.path.exists(VECTOR_DB_PATH):
            st.error(f"Vector database not found at {VECTOR_DB_PATH}")
            st.info("Run the setup script first: `python docs_handler.py`")
            st.stop()
        
        vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    with st.spinner("Loading Mistral-7B model from Hugging Face Inference..."):
        try:
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-v0.1", 
                huggingfacehub_api_token=hf_token,
                max_new_tokens=512
            )
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.stop()
    
    # Build chain
    prompt_template = ChatPromptTemplate.from_template(
        """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {input}

Answer:"""
    )
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return chain

def process_query(question):
    """Process user question through RAG pipeline"""
    chain = load_rag_pipeline()
    
    try:
        result = chain.invoke({"input": question})
        
        answer = result.get("answer", result.get("output", ""))
        if not answer or answer.strip() == "":
            return {
                "answer": "❌ No response from model. Check your HF token or try again.",
                "sources": []
            }
        
        return {
            "answer": answer,
            "sources": result.get("context", [])
        }
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

def rebuild_vector_db():
    """Rebuild vector database from PDFs"""
    st.info("Starting vector database rebuild...")
    
    # Load PDFs
    documents = []
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        st.error(f"No PDFs found in {PDF_FOLDER}")
        return
    
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
    
    **Model**: Mistral-7B-Instruct (via HF Inference)  
    **Embeddings**: All-MiniLM-L6-v2  
    **Vector DB**: FAISS
    """)
    
    st.divider()
    st.subheader("📝 Setup Required")
    st.write("""
    1. Get a Hugging Face API token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    2. Add it as an environment variable: `HF_TOKEN`
    3. On Streamlit Cloud, use "Manage App" → "Secrets"
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
                file_name = source.metadata.get("source", "Unknown") if hasattr(source, 'metadata') else "Unknown"
                page_num = source.metadata.get("page", "N/A") if hasattr(source, 'metadata') else "N/A"
                sources_html += f"{i}. <strong>{file_name}</strong> (Page {page_num})<br/>"
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
        try:
            result = process_query(user_input)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", [])
            })
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    
    # Rerun to display new message
    st.rerun()