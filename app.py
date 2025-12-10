import streamlit as st
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
VECTOR_DB_PATH = "./vector_db"
PDF_FOLDER = "./pdfs"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# Updated models that work with Inference API
SUPPORTED_MODELS = {
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Flan-T5-Large": "google/flan-t5-large",
    "Flan-T5-Base": "google/flan-t5-base",
}

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
        background-color: #1a1a1a;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #1a1a1a;
        border-left: 4px solid #4CAF50;
    }
    .source-box {
        background-color: #2a2a2a;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        border-left: 3px solid #FBC02D;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline(model_id):
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
    
    with st.spinner(f"Loading {model_id}..."):
        try:
            # Configure based on model type
            if "flan-t5" in model_id.lower():
                llm = HuggingFaceEndpoint(
                    repo_id=model_id,
                    huggingfacehub_api_token=hf_token,
                    task="text2text-generation",
                    temperature=0.5,
                    max_new_tokens=256,
                    top_p=0.9
                )
                
                prompt_template = ChatPromptTemplate.from_template(
                    """Context: {context}

Question: {input}

Answer the question based only on the context above. Keep the answer concise."""
                )
            else:  # Mistral and other instruct models
                llm = HuggingFaceEndpoint(
                    repo_id=model_id,
                    huggingfacehub_api_token=hf_token,
                    temperature=0.7,
                    max_new_tokens=512,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                
                prompt_template = ChatPromptTemplate.from_template(
                    """You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {input}

Answer:"""
                )
            
            st.success(f"✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            
            # Provide helpful error messages
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg:
                st.warning("🔑 Check your HF_TOKEN - authentication failed")
            elif "404" in error_msg or "not found" in error_msg:
                st.warning(f"🔍 Model not accessible. Try a different model.")
            elif "rate limit" in error_msg:
                st.warning("🚫 Rate limit exceeded. Wait a few minutes.")
            else:
                st.info("💡 Try using 'Flan-T5-Base' - it's more reliable")
            
            st.stop()
    
    # Build chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return chain

def process_query(question, model_id):
    """Process a user query through the RAG pipeline"""
    chain = load_rag_pipeline(model_id)
    
    try:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Model: {model_id}")
        print('='*60)
        
        # Invoke the chain
        result = chain.invoke({"input": question})
        
        # Debug output
        print(f"\nResult type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        # Extract answer with fallbacks
        answer = None
        sources = []
        
        if isinstance(result, dict):
            # Try different possible keys
            for key in ["answer", "output", "output_text", "result", "response"]:
                if key in result and result[key]:
                    answer = result[key]
                    print(f"✓ Answer found in key: '{key}'")
                    break
            
            # Extract sources
            if "context" in result:
                sources = result["context"]
            elif "source_documents" in result:
                sources = result["source_documents"]
            
            print(f"Sources found: {len(sources)}")
        
        # Validate answer
        if not answer or (isinstance(answer, str) and not answer.strip()):
            answer = "⚠️ I couldn't generate a proper answer. Please try rephrasing your question."
            print("⚠ No valid answer extracted")
        else:
            print(f"✓ Answer length: {len(str(answer))} chars")
        
        print('='*60 + '\n')
        
        return {
            "answer": str(answer).strip(),
            "sources": sources
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        print("\n" + "="*60)
        print("❌ ERROR:")
        print(error_details)
        print("="*60 + "\n")
        
        # User-friendly error message
        error_msg = str(e)
        if "StopIteration" in error_msg:
            user_error = "Model configuration error. Please try:\n1. Select 'Flan-T5-Base' model\n2. Clear cache and reload\n3. Check your HF_TOKEN"
        elif "timeout" in error_msg.lower():
            user_error = "Request timeout. The model might be loading. Please try again."
        elif "rate" in error_msg.lower():
            user_error = "Rate limit exceeded. Please wait a moment and try again."
        else:
            user_error = f"Error: {str(e)[:200]}"
        
        return {
            "answer": f"❌ {user_error}",
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
    
    st.success("✅ Vector database rebuilt successfully!")
    st.cache_resource.clear()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selection
    st.subheader("🤖 Select Model")
    selected_model_name = st.selectbox(
        "Choose a model:",
        options=list(SUPPORTED_MODELS.keys()),
        index=1,  # Default to Flan-T5-Large
        help="Flan-T5 models are recommended for best results"
    )
    selected_model_id = SUPPORTED_MODELS[selected_model_name]
    
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
    """)
    st.write(f"**Model**: {selected_model_name}")
    st.write("**Embeddings**: All-MiniLM-L6-v2")
    st.write("**Vector DB**: FAISS")
    
    st.divider()

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
user_input = st.chat_input("Ask a question about your documents...")

# Process user input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Get response from RAG pipeline
    with st.spinner("🤔 Thinking..."):
        try:
            result = process_query(user_input, selected_model_id)
            
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