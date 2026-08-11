from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import CrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
from .ingestion import Ingestion
from .retrieval import Retrieval


load_dotenv()

class RAGSystem(Ingestion,Retrieval):
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = 'BAAI/bge-small-en-v1.5'
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )

        self.persist_directory = "./chroma_db"
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

        self.llm = GoogleGenerativeAI(
            model = "gemini-3.6-flash" 
        )

        self.vector_store = None
        


        

        