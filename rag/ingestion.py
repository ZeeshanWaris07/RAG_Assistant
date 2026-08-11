from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

class Ingestion:
    def ingest_documents(self, file_paths):

        pages = []

        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            pages.extend(loader.load())

        chunks = self.text_splitter.split_documents(pages)

        self.vector_store.add_documents(chunks)