from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

class Retrieval:

    def make_pairs(self,question,documents):
        pairs = []

        for doc in documents:
            pairs.append((question,doc.page_content))

        return pairs

    def make_reranked_docs(scores,docs):
        reranked_docs = sorted(
            zip(scores,docs),
            key= lambda x: x[0],
            reverse=True
        )

        return [doc for score,doc in reranked_docs]

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

        pairs = self.make_pairs(question,retrieved_chunks)

        scores = self.reranker.predict(pairs)

        reranked_docs = self.make_reranked_docs(scores,retrieved_chunks)

        top_chunks = reranked_docs[:5]

        






        

        

        