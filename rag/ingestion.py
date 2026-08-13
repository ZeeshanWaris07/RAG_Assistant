from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
load_dotenv()

class Ingestion:
    def ingest_documents(self, file_paths):

        pages = []

        try:
            from langchain_community.document_loaders import PDFPlumberLoader
        except Exception:
            PDFPlumberLoader = None

        for file_path in file_paths:
            # Prefer PDFPlumberLoader which usually preserves spaces better
            if PDFPlumberLoader is not None:
                try:
                    loader = PDFPlumberLoader(file_path)
                    pages.extend(loader.load())
                    continue
                except Exception as e:
                    # If PDFPlumber fails for any reason, fall back to PyPDFLoader
                    print(f"PDFPlumberLoader failed for {file_path}, falling back to PyPDFLoader: {e}")

            loader = PyPDFLoader(file_path)
            pages.extend(loader.load())

        
        chunks = self.text_splitter.split_documents(pages)

        self.bm25_retriever = BM25Retriever.from_documents(
             documents = chunks,
             k = 10
        )

        if not os.path.exists(self.persist_directory):      
                self.vector_store = Chroma.from_documents(
                    documents= chunks,
                    embedding= self.embeddings,
                    persist_directory= self.persist_directory
                )
        else:
            self.vector_store = Chroma(
                persist_directory= self.persist_directory,
                embedding_function= self.embeddings
            )

        