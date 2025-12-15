import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_DIR, "index.faiss")

def ingest_pdf(pdf_path: str):
    chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Add source metadata to all documents
    for doc in docs:
        doc.metadata["source"] = os.path.basename(pdf_path)
    
    # Split documents
    split_docs = text_splitter.split_documents(docs)
    
    # Only filter if you're SURE these pages are useless
    # Otherwise, comment out this filtering
    filtered_chunks = []
    for doc in split_docs:
        text_lower = doc.page_content.lower()
        # Only skip if the ENTIRE chunk is boilerplate
        if len(doc.page_content.strip()) < 50:  # Skip very short chunks
            continue
        if text_lower.startswith("legal notice") or text_lower.startswith("disclaimer"):
            continue
        filtered_chunks.append(doc)
    
    chunks = filtered_chunks if filtered_chunks else split_docs
    
    # Limit chunks to prevent memory issues on Railway
    MAX_CHUNKS = 200
    if len(chunks) > MAX_CHUNKS:
        print(f"⚠️ Limiting to {MAX_CHUNKS} chunks (out of {len(chunks)}) to prevent memory issues")
        chunks = chunks[:MAX_CHUNKS]

    if not chunks:
        raise ValueError("No useful chunks created from PDF")

    print(f"📄 Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")

    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=256
    )

    if os.path.exists(INDEX_FILE):
        print("📚 Loading existing vectorstore...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
        print("✅ Added to existing vectorstore")
    else:
        print("🆕 Creating new vectorstore...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✅ Created new vectorstore")

    vectorstore.save_local(VECTORSTORE_DIR)

    return {
        "status": "success",
        "chunks_added": len(chunks),
        "file": os.path.basename(pdf_path)
    }
