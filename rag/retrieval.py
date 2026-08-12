from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.messages import AIMessage, HumanMessage
class Retrieval:

    def make_pairs(self,question,documents):
        pairs = []

        for doc in documents:
            pairs.append((question,doc.page_content))

        return pairs


    def format_docs(self,docs):
    
                context = []
    
                for doc in docs:
                    context.append(
                        f"""
    Source: {doc.metadata['source']}
    Page: {doc.metadata['page']}
    
    
    {doc.page_content}
                        """
                    )

                return "\n\n".join(context)

    def make_reranked_docs(self,scores,docs):
        reranked_docs = sorted(
            zip(scores,docs),
            key= lambda x: x[0],
            reverse=True
        )

        return [doc for score,doc in reranked_docs]


    def generate(self,question,top_chunks):

        formatted_chunks = self.format_docs(top_chunks)

        response = self.generation_chain.invoke({
             'chat_history' : self.chat_history,
             'question' : question,
             'context' : formatted_chunks
        })

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))

        return response



    def chat(self,question):

        search_retriever = self.vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                "k" : 10,
                "fetch_k" : 20
            }
        )

        retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever,search_retriever],
            weights=[0.4,0.6]
        )

        print(f"Question : {question}")
        
        rewritted_question = self.rewrite_chain.invoke({
                    'chat_history' : self.chat_history,
                    'question' : question
        })

        print(f"Rewritten Question : {rewritted_question}")
        
        retrieved_chunks = retriever.invoke(rewritted_question)

        for i, doc in enumerate(retrieved_chunks):
            print("=" * 80)
            print(f"BEFORE RERANKING - CHUNK {i + 1}")
            print("PAGE:", doc.metadata.get("page"))
            print(doc.page_content[:500])

        pairs = self.make_pairs(rewritted_question,retrieved_chunks)

        scores = self.reranker.predict(pairs)

        reranked_docs = self.make_reranked_docs(scores,retrieved_chunks)

        top_chunks = reranked_docs[:5]

        for i, doc in enumerate(top_chunks):
            print("=" * 80)
            print(f"CHUNK {i + 1}")
            print("SOURCE:", doc.metadata.get("source"))
            print("PAGE:", doc.metadata.get("page"))
            print(doc.page_content)

        response = self.generate(question,top_chunks)

        return response








        

        

        