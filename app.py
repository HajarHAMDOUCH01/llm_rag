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
import traceback
from datetime import datetime

# Configuration
VECTOR_DB_PATH = "./vector_db"
PDF_FOLDER = "./pdfs"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# Updated models that work with Inference API
SUPPORTED_MODELS = {
    "google/gemma-3-12b-it": "google/gemma-3-12b-it",
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
    .debug-box {
        background-color: #2a2a2a;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border-left: 3px solid #FF5722;
        font-family: monospace;
    }
    .error-box {
        background-color: #3d1f1f;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border-left: 3px solid #f44336;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

def log_debug(message, level="INFO"):
    """Centralized logging function"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_prefix = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "DEBUG": "🔍"
    }
    prefix = log_prefix.get(level, "ℹ️")
    print(f"[{timestamp}] {prefix} {message}")
    return f"[{timestamp}] {prefix} {message}"

@st.cache_resource
def load_rag_pipeline(model_id):
    """Load RAG pipeline using Hugging Face Inference API with detailed logging"""
    
    debug_logs = []
    
    try:
        # STEP 1: Check HF Token
        log_debug("STEP 1: Checking HF_TOKEN", "DEBUG")
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            log_debug("HF_TOKEN not found", "ERROR")
            st.error("❌ HF_TOKEN environment variable not set.")
            st.info("📋 Get a token from: https://huggingface.co/settings/tokens")
            st.info("📝 On Streamlit Cloud: Manage App → Secrets → Add HF_TOKEN")
            st.stop()
        
        debug_logs.append(log_debug(f"HF_TOKEN found (length: {len(hf_token)})", "SUCCESS"))
        
        # STEP 2: Load Embeddings
        log_debug("STEP 2: Loading embeddings model", "DEBUG")
        with st.spinner("Loading embeddings model..."):
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
            debug_logs.append(log_debug(f"Embeddings model loaded: {EMBEDDINGS_MODEL}", "SUCCESS"))
        
        # STEP 3: Load Vector Store
        log_debug("STEP 3: Loading vector database", "DEBUG")
        with st.spinner("Loading vector database..."):
            if not os.path.exists(VECTOR_DB_PATH):
                log_debug(f"Vector database not found at {VECTOR_DB_PATH}", "ERROR")
                st.error(f"Vector database not found at {VECTOR_DB_PATH}")
                st.info("Run the setup script first: `python docs_handler.py`")
                st.stop()
            
            vector_store = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Verify vector store
            num_docs = vector_store.index.ntotal
            debug_logs.append(log_debug(f"Vector store loaded with {num_docs} vectors", "SUCCESS"))
            
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            debug_logs.append(log_debug("Retriever created with k=3", "SUCCESS"))
        
        # STEP 4: Load LLM
        log_debug(f"STEP 4: Loading LLM: {model_id}", "DEBUG")
        with st.spinner(f"Loading {model_id}..."):
            try:
                # Configure based on model type
                if "flan-t5" in model_id.lower():
                    debug_logs.append(log_debug("Configuring Flan-T5 model", "DEBUG"))
                    
                    llm = HuggingFaceEndpoint(
                        repo_id=model_id,
                        huggingfacehub_api_token=hf_token,
                        task="image-text-to-text",
                        temperature=0.5,
                        max_new_tokens=512,
                        top_p=0.9
                    )
                    
                    prompt_template = ChatPromptTemplate.from_template(
                        """Answer the question based on the context below. Be concise and informative.

Context: {context}

Question: {input}

Answer:"""
                    )
                    debug_logs.append(log_debug("Flan-T5 prompt template created", "SUCCESS"))
                    
                else:  # Mistral and other instruct models
                    debug_logs.append(log_debug("Configuring Mistral/Instruct model", "DEBUG"))
                    
                    llm = HuggingFaceEndpoint(
                        repo_id=model_id,
                        huggingfacehub_api_token=hf_token,
                        temperature=0.7,
                        max_new_tokens=512,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant. Answer questions based on the provided context. Be clear and concise."),
                        ("user", "Context: {context}\n\nQuestion: {input}\n\nAnswer:")
                    ])
                    debug_logs.append(log_debug("Mistral prompt template created", "SUCCESS"))
                
                debug_logs.append(log_debug(f"LLM endpoint initialized: {model_id}", "SUCCESS"))
                
            except Exception as llm_error:
                error_msg = str(llm_error)
                debug_logs.append(log_debug(f"LLM loading failed: {error_msg}", "ERROR"))
                
                st.error(f"❌ Failed to load model: {error_msg}")
                
                # Provide helpful error messages
                if "401" in error_msg or "unauthorized" in error_msg:
                    st.warning("🔑 Check your HF_TOKEN - authentication failed")
                elif "404" in error_msg or "not found" in error_msg:
                    st.warning(f"🔍 Model not accessible. Try a different model.")
                elif "rate limit" in error_msg:
                    st.warning("🚫 Rate limit exceeded. Wait a few minutes.")
                else:
                    st.info("💡 Try using 'Flan-T5-Base' - it's more reliable")
                
                st.stop()
        
        # STEP 5: Build Chain
        log_debug("STEP 5: Building retrieval chain", "DEBUG")
        try:
            combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
            debug_logs.append(log_debug("Document combination chain created", "SUCCESS"))
            
            chain = create_retrieval_chain(retriever, combine_docs_chain)
            debug_logs.append(log_debug("Final retrieval chain created", "SUCCESS"))
            
        except Exception as chain_error:
            debug_logs.append(log_debug(f"Chain building failed: {str(chain_error)}", "ERROR"))
            raise
        
        st.success(f"✅ Model loaded successfully!")
        
        # Store debug logs in session state
        st.session_state.initialization_logs = debug_logs
        
        return chain
        
    except Exception as e:
        error_trace = traceback.format_exc()
        log_debug(f"Pipeline initialization failed:\n{error_trace}", "ERROR")
        st.error(f"❌ Failed to initialize RAG pipeline: {str(e)}")
        st.code(error_trace, language="python")
        st.stop()

def process_query(question, model_id):
    """Process a user query through the RAG pipeline with detailed debugging"""
    
    debug_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "model": model_id,
        "steps": [],
        "errors": []
    }
    
    try:
        # STEP 1: Load chain
        debug_info["steps"].append("Loading RAG pipeline...")
        log_debug(f"Processing query: {question[:50]}...", "DEBUG")
        
        chain = load_rag_pipeline(model_id)
        debug_info["steps"].append("✅ Chain loaded successfully")
        
        # STEP 2: Invoke chain
        debug_info["steps"].append("Invoking chain with input...")
        log_debug("Invoking retrieval chain", "DEBUG")
        
        try:
            # Try to invoke with detailed error capture
            result = chain.invoke({"input": question})
            debug_info["steps"].append(f"✅ Chain invoked, result type: {type(result).__name__}")
        except StopIteration as e:
            debug_info["steps"].append(f"❌ StopIteration error during chain.invoke()")
            debug_info["errors"].append(f"StopIteration: {str(e)}")
            raise Exception("StopIteration error - Model did not return expected output. This usually means the model endpoint is not responding correctly. Try a different model or check HuggingFace API status.")
        except Exception as invoke_error:
            debug_info["steps"].append(f"❌ Chain invocation error: {type(invoke_error).__name__}")
            debug_info["errors"].append(f"Invocation error: {str(invoke_error)}")
            raise
        
        # STEP 3: Inspect result
        debug_info["steps"].append("Inspecting result structure...")
        log_debug(f"Result type: {type(result)}", "DEBUG")
        
        if isinstance(result, dict):
            debug_info["result_keys"] = list(result.keys())
            log_debug(f"Result keys: {result.keys()}", "DEBUG")
            
            for key, value in result.items():
                value_type = type(value).__name__
                value_preview = str(value)[:100] if value else "None"
                debug_info["steps"].append(f"  Key '{key}': {value_type} - {value_preview}...")
                log_debug(f"  Key '{key}': {value_type}", "DEBUG")
        else:
            debug_info["steps"].append(f"⚠️ Result is not a dict: {type(result)}")
            debug_info["result_raw"] = str(result)[:500]
        
        # STEP 4: Extract answer
        debug_info["steps"].append("Extracting answer...")
        answer = None
        sources = []
        
        if isinstance(result, dict):
            # Try different possible keys
            possible_answer_keys = ["answer", "result", "output", "response"]
            for key in possible_answer_keys:
                if key in result and result[key]:
                    answer = result[key]
                    debug_info["steps"].append(f"✅ Answer found in '{key}' key")
                    log_debug(f"Answer extracted from '{key}' key", "SUCCESS")
                    break
            
            # Extract sources
            possible_source_keys = ["context", "source_documents", "sources"]
            for key in possible_source_keys:
                if key in result:
                    sources = result[key]
                    debug_info["steps"].append(f"✅ Found {len(sources) if sources else 0} sources in '{key}' key")
                    log_debug(f"Sources extracted from '{key}' key: {len(sources) if sources else 0}", "SUCCESS")
                    break
        
        # STEP 5: Validate answer
        debug_info["steps"].append("Validating answer...")
        if not answer:
            debug_info["errors"].append("No answer found in result")
            debug_info["steps"].append("❌ No answer key found in result")
            answer = "⚠️ Error: The model returned a response but I couldn't extract the answer. Please check the debug info below."
            log_debug("No answer found in result dictionary", "ERROR")
        elif isinstance(answer, str) and not answer.strip():
            debug_info["errors"].append("Answer is empty string")
            debug_info["steps"].append("❌ Answer is empty")
            answer = "⚠️ The model returned an empty answer. Please try rephrasing your question."
            log_debug("Answer is empty string", "WARNING")
        else:
            answer_length = len(str(answer))
            debug_info["answer_length"] = answer_length
            debug_info["answer_preview"] = str(answer)[:200]
            debug_info["steps"].append(f"✅ Valid answer extracted ({answer_length} chars)")
            log_debug(f"Valid answer extracted: {answer_length} characters", "SUCCESS")
        
        # STEP 6: Process sources
        debug_info["steps"].append(f"Processing {len(sources) if sources else 0} source documents...")
        processed_sources = []
        if sources:
            for i, source in enumerate(sources):
                if hasattr(source, 'metadata'):
                    processed_sources.append(source)
                    debug_info["steps"].append(f"  Source {i+1}: {source.metadata.get('source', 'Unknown')}")
        
        debug_info["steps"].append("✅ Query processing complete")
        log_debug("Query processing completed successfully", "SUCCESS")
        
        return {
            "answer": str(answer).strip(),
            "sources": processed_sources,
            "debug_info": debug_info
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        error_type = type(e).__name__
        error_message = str(e) if str(e) else "No error message (empty exception)"
        
        debug_info["errors"].append(f"{error_type}: {error_message}")
        debug_info["traceback"] = error_trace
        debug_info["steps"].append(f"❌ FATAL ERROR: {error_type} - {error_message}")
        
        log_debug(f"Query processing failed: {error_type} - {error_message}", "ERROR")
        print("\n" + "="*80)
        print(f"ERROR TYPE: {error_type}")
        print(f"ERROR MESSAGE: {error_message}")
        print("FULL ERROR TRACEBACK:")
        print(error_trace)
        print("="*80 + "\n")
        
        # Categorize error
        error_msg = error_message.lower()
        error_type_lower = error_type.lower()
        
        if "stopiteration" in error_type_lower or "stopiteration" in error_msg:
            user_error = "Model configuration error. The model didn't return expected output."
            suggestion = "Try: 1) Switch to 'Flan-T5-Base', 2) Clear cache (sidebar), 3) Check your HF_TOKEN"
        elif "timeout" in error_msg:
            user_error = "Request timeout. The model might be loading or overloaded."
            suggestion = "Wait a moment and try again."
        elif "rate" in error_msg:
            user_error = "Rate limit exceeded on Hugging Face API."
            suggestion = "Wait a few minutes before trying again."
        elif "token" in error_msg or "auth" in error_msg:
            user_error = "Authentication error with Hugging Face."
            suggestion = "Check your HF_TOKEN in the environment variables."
        elif not error_message or error_message == "No error message (empty exception)":
            user_error = "Unknown error with empty message. Likely a StopIteration or generator issue."
            suggestion = "1) Check console logs for full traceback, 2) Try 'Flan-T5-Large' instead, 3) Verify HuggingFace API status"
        else:
            user_error = f"Unexpected error ({error_type}): {error_message}"
            suggestion = "Check the debug information below for details."
        
        return {
            "answer": f"❌ {user_error}\n\n💡 Suggestion: {suggestion}",
            "sources": [],
            "debug_info": debug_info
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
    
    # Debug mode toggle
    st.subheader("🔧 Debug Options")
    show_debug = st.checkbox("Show Debug Info", value=True)
    
    if st.button("Clear Cache & Reload"):
        st.cache_resource.clear()
        st.success("Cache cleared!")
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
    
    # Show initialization logs
    if hasattr(st.session_state, 'initialization_logs'):
        with st.expander("📋 Initialization Logs"):
            for log in st.session_state.initialization_logs:
                st.text(log)

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
        # Build sources HTML
        sources_html = ""
        if message.get("sources"):
            sources_html = "<div class='source-box'><strong>📄 Sources:</strong><br/>"
            for i, source in enumerate(message["sources"], 1):
                file_name = source.metadata.get("source", "Unknown") if hasattr(source, 'metadata') else "Unknown"
                page_num = source.metadata.get("page", "N/A") if hasattr(source, 'metadata') else "N/A"
                sources_html += f"{i}. <strong>{file_name}</strong> (Page {page_num})<br/>"
            sources_html += "</div>"
        
        # Build debug HTML
        debug_html = ""
        if show_debug and message.get("debug_info"):
            debug = message["debug_info"]
            debug_html = "<div class='debug-box'><strong>🔍 Debug Information:</strong><br/>"
            debug_html += f"<strong>Timestamp:</strong> {debug.get('timestamp', 'N/A')}<br/>"
            debug_html += f"<strong>Model:</strong> {debug.get('model', 'N/A')}<br/>"
            
            if debug.get("result_keys"):
                debug_html += f"<strong>Result Keys:</strong> {', '.join(debug['result_keys'])}<br/>"
            
            if debug.get("steps"):
                debug_html += "<strong>Processing Steps:</strong><br/>"
                for step in debug["steps"]:
                    debug_html += f"  {step}<br/>"
            
            if debug.get("errors"):
                debug_html += "<strong style='color:#f44336;'>Errors:</strong><br/>"
                for error in debug["errors"]:
                    debug_html += f"  {error}<br/>"
            
            if debug.get("traceback"):
                debug_html += "<details><summary><strong>Full Traceback (click to expand)</strong></summary>"
                debug_html += f"<pre style='font-size:0.8em;'>{debug['traceback']}</pre>"
                debug_html += "</details>"
            
            debug_html += "</div>"
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div style="flex: 1;">
                <strong>Assistant</strong><br/>
                {message["content"]}
                {sources_html}
                {debug_html}
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
                "content": result.get("answer", "No answer generated"),
                "sources": result.get("sources", []),
                "debug_info": result.get("debug_info", {})
            })
            
        except Exception as e:
            error_trace = traceback.format_exc()
            st.error(f"Critical error processing query: {str(e)}")
            st.code(error_trace, language="python")
            
            # Add error message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Critical Error: {str(e)}",
                "sources": [],
                "debug_info": {"error": str(e), "traceback": error_trace}
            })
    
    # Rerun to display new message
    st.rerun()