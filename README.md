# Multi-Document RAG Chatbot Backend

A full-stack Retrieval Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions using natural language.

## Features

- 📄 Upload PDF documents
- 💬 Ask questions about uploaded documents
- 🔍 Semantic search using FAISS vector store
- 🤖 AI-powered responses using Groq LLM
- 🔄 Multi-document support (query across multiple PDFs)
- 🔒 Session-based storage (privacy-focused)

## Tech Stack

- **Backend:** FastAPI, Python
- **AI/ML:** LangChain, FAISS, FastEmbed
- **LLM:** Groq API (Llama 3.1)
- **Document Processing:** PyPDF
- **Deployment:** Railway

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

4. Run the application:
```bash
uvicorn app:app --reload
```

5. Open http://localhost:8000

## API Endpoints

- `GET /` - Health check
- `POST /upload` - Upload PDF document
- `POST /ask` - Ask questions about uploaded documents
- `GET /status` - Check vectorstore status

## Environment Variables

- `GROQ_API_KEY` - Your Groq API key

## 👨‍💻 Author
**Built by Amit Sharma**

**⭐ Star the repository if you like this project!**

