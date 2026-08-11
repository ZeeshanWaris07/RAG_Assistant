from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

class Retrieval:

    def retrieve_chunks(self,question):

        pass

    def chat(self,question):

        search_retriever = self.vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                "k" : 10,
                "top_k" : 20
            }
        )

        retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever,search_retriever],
            weights=[0.4,0.6]
        )

        retrieved_chunks = retriever.invoke(question)
        

        

        