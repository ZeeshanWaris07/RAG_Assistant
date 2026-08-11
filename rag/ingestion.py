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

class RAGSystem():
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = 'BAAI/bge-small-en-v1.5'
        )

        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

        self.llm = GoogleGenerativeAI(
            model_name="gpt-4o",
            temperature=0.2
        )

        print('LLM initialized.')

        self.prompt_template = PromptTemplate(
            template="""You are a helpful assistant that answers questions based on the provided context.