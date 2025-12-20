# Local RAG System 

A local Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents using Ollama and LangChain.

## What It Does

Upload PDFs, ask questions, and get accurate answers backed by your documents - all running **100% locally** with no API costs.

## Architecture

![image alt](https://github.com/HajarHAMDOUCH01/llm_rag/blob/92dbddbe791d4024f6d160d04a27e9f0e4781741/imgs_%26_video/Archi.png)

![image alt](https://github.com/HajarHAMDOUCH01/llm_rag/blob/92dbddbe791d4024f6d160d04a27e9f0e4781741/imgs_%26_video/screen1.png)

## Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/HajarHAMDOUCH01/llm_rag
cd llm_rag

# Install dependencies
pip install -r requierements.txt

# Pull the Ollama model
ollama pull nemotron-3-nano:30b-cloud
```

### Setup Your Documents

```bash
# 1. Add your PDFs to the pdfs/ folder
mkdir -p pdfs
cp your-documents.pdf pdfs/

# 2. Create the vector database
python docs_handler.py
```

### Run the Server

```bash
python app.py
```

Server runs at `http://localhost:5000` -> open index.html

## API Usage

### Query Endpoint
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

### Response Format
```json
{
  "answer": "The document discusses...",
  "sources": [
    {
      "file": "document.pdf",
      "page": 3,
      "text": "Relevant excerpt..."
    }
  ]
}
```

## Tech Stack

- **LLM**: Ollama (nemotron-3-nano:30b-cloud)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **Framework**: LangChain
- **API**: Flask + CORS
```
