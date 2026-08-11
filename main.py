from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import os
import shutil

from rag import RAGSystem


app = FastAPI(title="PDF RAG API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

rag = RAGSystem()


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...)
):
    # Maximum 3 PDFs
    if len(files) > 3:
        raise HTTPException(
            status_code=400,
            detail="You can upload a maximum of 3 PDFs."
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one PDF."
        )

    saved_files = []

    for file in files:

        # Check extension
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is not a PDF."
            )

        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files.append(file_path)

    # Send PDFs to your RAG system
    rag.ingest_documents(saved_files)

    return {
        "message": "PDFs successfully uploaded and indexed.",
        "files": [file.filename for file in files]
    }