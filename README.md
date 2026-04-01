# Intelligent RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their content using AI. Built with Streamlit, LangChain, and Ollama.

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF documents
- **Intelligent Q&A**: Ask questions about your documents using AI
- **Vector Storage**: Uses ChromaDB for efficient document embedding storage
- **Source Attribution**: View source chunks used to generate answers
- **Chat Interface**: Interactive chat-based interface with conversation history
- **Real-time Processing**: Fast document indexing and response generation

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- Ollama installed and running
- At least 4GB RAM recommended

### Ollama Setup
1. Install Ollama: [https://ollama.com/download](https://ollama.com/download)
2. Pull the required model:
   ```bash
   ollama pull llama3
   ```
3. Start Ollama service:
   ```bash
   ollama serve
   ```

## 🛠️ Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate 
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Project Structure

```
Intelligent-RAG/
├── app.py                 # Streamlit web application
├── rag_pipeline.py        # Core RAG pipeline functions
├── requirements.txt       # Python dependencies
├── chroma_db/            # Vector database storage
├── .venv/                 # Virtual environment
└── README.md             # This file
```

## 🤖 Architecture Overview

### Core Components

#### 1. Document Processing (`rag_pipeline.py`)
- **`load_pdf()`**: Loads PDF documents using PyPDFLoader
- **`split_text()`**: Splits documents into chunks using RecursiveCharacterTextSplitter
- **`create_vector_store()`**: Creates vector embeddings and stores in ChromaDB
- **`create_qa_chain()`**: Sets up the RAG question-answering chain

#### 2. Web Interface (`app.py`)
- **Streamlit UI**: Interactive web interface with sidebar settings
- **Chat Interface**: Real-time chat with conversation history
- **File Upload**: PDF upload and processing controls
- **Response Display**: Shows answers with source attribution

### Technical Stack

- **Frontend**: Streamlit (web UI)
- **Backend**: Python with LangChain
- **LLM**: Ollama with Llama3 model
- **Vector Database**: ChromaDB
- **Document Processing**: PyPDF
- **Embeddings**: Ollama Embeddings

## 🚀 Usage

### Step 1: Start the Application
```bash
streamlit run app.py
```

### Step 2: Upload and Process Document
1. Open your browser to `http://localhost:8501`
2. In the sidebar, click "Upload a PDF" and select your document
3. Click "Process Document" to index the PDF
4. Wait for processing to complete (you'll see the number of chunks indexed)

### Step 3: Ask Questions
1. Use the chat input at the bottom to ask questions about your document
2. The AI will provide answers based on the document content
3. Click "Sources" to see which document chunks were used
4. Response time is displayed for each query

## ⚙️ Configuration

### Text Splitting Parameters
- **Chunk Size**: 2000 characters
- **Chunk Overlap**: 300 characters
- **Separators**: `["\n\n", "\n", ".", " ", ""]`

### Retrieval Parameters
- **Search Type**: Maximal Marginal Relevance (MMR)
- **Retrieved Chunks**: 8 (k=8)
- **Candidate Chunks**: 20 (fetch_k=20)

### Model Configuration
- **LLM Model**: llama3 (via Ollama)
- **Embedding Model**: llama3 (via Ollama)

## 🧠 How It Works

### 1. Document Ingestion
- PDF files are loaded and parsed into text documents
- Documents are split into manageable chunks with overlap
- Each chunk is converted into a vector embedding

### 2. Vector Storage
- Embeddings are stored in ChromaDB for fast similarity search
- The vector database persists between sessions

### 3. Question Answering
- User questions are converted to embeddings
- Similar document chunks are retrieved using MMR
- Retrieved context and question are sent to the LLM
- LLM generates answers based on the provided context

### 4. Response Generation
- The system follows strict rules for answer generation
- Answers are based only on the provided context
- Source documents are always referenced

## 🔧 Customization

### Changing the Model
Edit `rag_pipeline.py` to use different Ollama models:
```python
llm = OllamaLLM(model="your-model-name")
embeddings = OllamaEmbeddings(model="your-model-name")
```

### Adjusting Chunk Size
Modify the text splitter parameters in `split_text()`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust as needed
    chunk_overlap=200,  # Adjust as needed
    separators=["\n\n", "\n", ".", " ", ""]
)
```

### Customizing the Prompt
Edit the `prompt_template` in `create_qa_chain()` to modify how the AI responds.

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Verify the model is installed: `ollama list`

2. **Memory Issues**
   - Reduce chunk size for large documents
   - Close other applications to free up RAM

3. **Slow Response Times**
   - Check if Ollama is running efficiently
   - Consider reducing the number of retrieved chunks

4. **PDF Processing Errors**
   - Ensure the PDF is not password-protected
   - Try converting the PDF to text if issues persist

### Performance Tips
- Use smaller chunk sizes for faster processing
- Limit the number of documents processed at once
- Ensure sufficient RAM is available for Ollama

## 📊 Performance Metrics

- **Document Processing**: Typically 2-5 seconds per page
- **Query Response**: 3-10 seconds depending on complexity
- **Memory Usage**: ~500MB base + document size
- **Storage**: Vector embeddings stored in `./chroma_db`

## 🎯 Use Cases

- **Research**: Query academic papers and research documents
- **Legal**: Analyze legal documents and contracts
- **Education**: Study textbooks and course materials
- **Business**: Process reports, manuals, and documentation
- **Personal**: Organize and query personal document libraries

---

**Note**: This application requires Ollama to be running locally. Ensure you have sufficient system resources for optimal performance.
