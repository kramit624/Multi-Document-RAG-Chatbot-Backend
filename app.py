from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from ingest import ingest_pdf
from query import answer_question

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Multi-Doc RAG API")

# 🔥 CORS FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 OPTIONS HANDLER (CRITICAL)
@app.options("/{path:path}")
async def options_handler(path: str):
    return {}

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile | None = File(None)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)

        result = ingest_pdf(path)

        return {
            "status": "success",
            "filename": file.filename,
            **result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer = answer_question(payload.question)
        return {
            "question": payload.question,
            "answer": answer
        }
    except Exception as e:
        print(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/status")
def status():
    """Check if vectorstore exists and how many documents"""
    vectorstore_exists = os.path.exists("vectorstore/index.faiss")
    uploaded_files = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
    
    return {
        "vectorstore_exists": vectorstore_exists,
        "uploaded_files": uploaded_files,
        "num_files": len(uploaded_files)
    }
