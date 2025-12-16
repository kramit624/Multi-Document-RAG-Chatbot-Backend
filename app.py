from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from ingest import ingest_pdf
from query import answer_question

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Multi-Doc RAG API")

# 🔥 CORS FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://multi-document-rag-chatbot-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def options_handler(path: str):
    return {}

class QuestionRequest(BaseModel):
    question: str

def ingest_single_pdf(file_path: str):
    ingest_pdf(file_path)

@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
      
        background_tasks.add_task(ingest_single_pdf, path)
        
        return {
            "status": "uploaded",
            "filename": file.filename,
            "ingestion": "started"
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
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/status")
def status():
    vectorstore_exists = os.path.exists("vectorstore/index.faiss")
    uploaded_files = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
    
    return {
        "vectorstore_exists": vectorstore_exists,
        "uploaded_files": uploaded_files,
        "num_files": len(uploaded_files)
    }


@app.delete("/clear")
def clear_all():
    """Clear all uploaded documents and vectorstore"""
    try:
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
        
        if os.path.exists(UPLOAD_DIR):
            for file in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        return {
            "status": "success", 
            "message": "All data cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")
