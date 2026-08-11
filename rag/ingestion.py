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

        self.bm25_retrieval(
             documents = chunks
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

        