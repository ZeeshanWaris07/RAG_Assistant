# RAG Question Answering System

A Retrieval-Augmented Generation (RAG) question-answering system that allows users to ask questions about uploaded documents. The system extracts and chunks documents, performs semantic and keyword-based retrieval, and uses an LLM to generate grounded answers from the retrieved context.

## Overview

This project implements a RAG pipeline designed for question answering over documents such as PDF textbooks.

The system combines:

* PDF document ingestion
* Text extraction with `PDFPlumberLoader`
* Recursive text chunking
* Dense vector search
* BM25 keyword search
* Metadata-aware retrieval
* Cross-encoder reranking
* LLM-based answer generation
* FastAPI backend
* React frontend
* Markdown and LaTeX rendering

The goal is to retrieve the most relevant pieces of information from the documents and provide accurate answers based on that retrieved context.

---

## Architecture

```text
                         ┌──────────────────┐
                         │    User Query    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   FastAPI API    │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Query Processing /       │
                    │ Metadata Filtering       │
                    └────────────┬─────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
          ┌─────────────────┐         ┌─────────────────┐
          │ Semantic Search │         │   BM25 Search   │
          │    Chroma       │         │ Keyword Search  │
          └────────┬────────┘         └────────┬────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                       ┌──────────────────┐
                       │ Candidate Chunks │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Cross-Encoder    │
                       │    Reranker      │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Top Relevant     │
                       │     Chunks       │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │       LLM        │
                       │ Answer Generation│
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Markdown /       │
                       │ LaTeX Response   │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ React Frontend   │
                       │ Rendering        │
                       └──────────────────┘
```

---

# 1. Document Ingestion

The ingestion pipeline starts by loading the user's documents.

For PDF files, `PDFPlumberLoader` is preferred over `PyPDFLoader` because some PDFs do not preserve whitespace correctly when extracted using `pypdf`.

For example, `PyPDFLoader` may extract:

```text
1Introduction
1.1WhoShouldReadThisBook?
2.1Scalars,Vectors,MatricesandTensors
```

while `PDFPlumberLoader` correctly extracts:

```text
1 Introduction
1.1 Who Should Read This Book?
2.1 Scalars, Vectors, Matrices and Tensors
```

Preserving whitespace is important because poor extraction quality directly affects chunking, embeddings, keyword retrieval, and ultimately answer quality.

### Current ingestion strategy

```python
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyPDFLoader

for file_path in file_paths:
    try:
        loader = PDFPlumberLoader(file_path)
        pages.extend(loader.load())
    except Exception as e:
        print(f"PDFPlumber failed: {e}")

        loader = PyPDFLoader(file_path)
        pages.extend(loader.load())
```

`PyPDFLoader` is used as a fallback if PDFPlumber fails.

---

# 2. Text Chunking

After extraction, documents are split into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

Current configuration:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### Chunk size

`chunk_size=500` means approximately 500 characters, not 500 tokens.

### Chunk overlap

`chunk_overlap=50` allows neighboring chunks to share some context.

For example:

```text
Chunk 1:
A B C D E F G H I J

Chunk 2:
                    I J K L M N O P
```

This helps reduce the chance of losing important information at chunk boundaries.

The current configuration is intentionally kept relatively small for experimentation. Different chunk sizes and overlaps can later be evaluated systematically.

---

# 3. Metadata

Each chunk retains document metadata such as:

```text
source
page
topic
subtopic
difficulty
```

Example:

```json
{
  "source": "uploads/deeplearning.pdf",
  "page": 305,
  "topic": "Deep Learning",
  "subtopic": "Recurrent Neural Networks"
}
```

Metadata can be used during retrieval to narrow the search space and improve relevance.

---

# 4. Vector Database

The chunks are stored in Chroma using their embeddings.

```python
self.vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=self.embeddings,
    persist_directory=self.persist_directory
)
```

The vector database allows semantic similarity search.

For example, a query such as:

```text
What happens when gradients become very small in an RNN?
```

can retrieve chunks containing:

```text
vanishing gradient problem
```

even when the exact words used in the query do not appear in the document.

---

# 5. Embeddings

The system uses a Hugging Face embedding model to convert text into numerical vectors.

The embedding model represents the semantic meaning of each chunk.

Conceptually:

```text
Text Chunk
    ↓
Embedding Model
    ↓
[0.12, -0.43, 0.81, ...]
    ↓
Vector Database
```

The same embedding model is used to encode the user's query before performing semantic similarity search.

---

# 6. BM25 Retrieval

In addition to semantic search, the system uses BM25 keyword retrieval.

```python
self.bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,
    k=10
)
```

BM25 is useful when exact terminology matters.

For example, a query containing:

```text
"vanishing gradient problem"
```

can benefit from keyword matching against chunks containing those exact terms.

This complements dense semantic retrieval.

---

# 7. Hybrid Retrieval

The system combines:

```text
Semantic Retrieval
        +
BM25 Keyword Retrieval
        ↓
Candidate Documents
```

This provides both:

### Semantic understanding

Useful when the query and document use different wording.

### Keyword matching

Useful for:

* technical terminology
* names
* formulas
* exact phrases
* identifiers

Combining the two approaches can improve retrieval robustness compared with relying on only one retrieval method.

---

# 8. Cross-Encoder Reranking

The initial retrievers may return several candidate chunks.

A cross-encoder reranker then evaluates the relationship between:

```text
Query + Candidate Chunk
```

rather than comparing independently generated embeddings.

Conceptually:

```text
Query
  │
  ├── Chunk 1 → relevance score
  ├── Chunk 2 → relevance score
  ├── Chunk 3 → relevance score
  ├── ...
  └── Chunk N → relevance score

              ↓

       Sort by relevance

              ↓

        Top-K chunks
```

This allows the system to reduce irrelevant results before sending context to the LLM.

---

# 9. LLM Answer Generation

The final retrieved chunks are passed to the LLM along with the user's question.

Conceptually:

```text
Question
   +
Retrieved Context
   ↓
LLM
   ↓
Final Answer
```

The LLM is instructed to answer using the retrieved context rather than relying entirely on its internal knowledge.

This reduces hallucination and allows the system to answer questions specifically about the uploaded documents.

---

# 10. Markdown and LaTeX Rendering

The backend intentionally allows the LLM to generate Markdown and LaTeX.

For example:

```text
The gradient is scaled according to
$\text{diag}(\lambda)^t$.
```

The backend does not strip these characters.

Instead, the frontend is responsible for rendering them.

This is preferable to manually removing characters because removing symbols such as:

```text
$
\
_
^
*
#
```

can damage mathematical expressions and formatting.

The React frontend can use libraries such as:

```text
react-markdown
remark-math
rehype-katex
```

to render the response.

Example:

```jsx
<ReactMarkdown
    remarkPlugins={[remarkMath]}
    rehypePlugins={[rehypeKatex]}
>
    {answer}
</ReactMarkdown>
```

This allows mathematical expressions to be displayed properly.

---

# 11. Example

A user asks:

```text
What is the vanishing gradient problem?
```

The retrieval system may return:

```text
Chunk 1
Page: 305

Any eigenvalues that are not near an absolute value of 1
will either explode if they are greater than 1 in magnitude
or vanish if they are less than 1 in magnitude...
```

and:

```text
Chunk 5
Page: 430

Gradient clipping helps to deal with exploding gradients,
but it does not help with vanishing gradients...
LSTMs and other self-loops and gating mechanisms...
```

The LLM can then combine the retrieved information into an answer.

---

# 12. Current Chunk Quality

During testing, the system successfully retrieved multiple relevant chunks for questions about vanishing gradients.

Example retrieval:

```text
Chunk 1 → Page 305 → Relevant
Chunk 2 → Page 428 → Relevant
Chunk 3 → Page 418 → Relevant
Chunk 4 → Page 203 → Mostly irrelevant
Chunk 5 → Page 430 → Highly relevant
```

This demonstrates that the retriever can identify relevant information from different sections of a large textbook.

One weakness observed was a very short fragment:

```text
can vanish near zero, making it difficult to learn parameters that are squared.
```

This should be investigated during future chunk-quality evaluation.

---

# 13. RAG Evaluation

The system should eventually be evaluated using separate retrieval and generation metrics.

## Retrieval Evaluation

Important metrics include:

* Recall@K
* Precision@K
* MRR
* Hit Rate
* NDCG

For example:

```text
Question
   ↓
Retrieve Top-K
   ↓
Does the relevant chunk appear?
   ↓
Recall@K
```

## Generation Evaluation

The generated answer can be evaluated for:

* Correctness
* Relevance
* Faithfulness
* Context utilization
* Hallucination
* Completeness

The goal is not simply to produce a fluent answer.

The answer should be:

```text
Relevant
    +
Correct
    +
Grounded in retrieved context
```

---

# 14. Project Structure

A possible project structure:

```text
RAG/
│
├── backend/
│   ├── app.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── embeddings.py
│   └── models.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
│
├── uploads/
│
├── chroma_db/
│
├── .env
├── requirements.txt
└── README.md
```

---

# 15. Technologies

### Backend

* Python
* FastAPI
* LangChain

### Document Processing

* PDFPlumber
* PyPDF

### Retrieval

* Chroma
* BM25
* Dense vector search
* Hybrid retrieval

### Reranking

* Cross-Encoder
* BGE Reranker

### Embeddings

* Hugging Face sentence-transformer models

### LLM

The system can be configured to use a locally hosted or API-based LLM.

### Frontend

* React
* Markdown rendering
* KaTeX/LaTeX rendering

---

# 16. Key Design Decisions

### PDFPlumber over PyPDF for problematic PDFs

Some PDFs do not preserve whitespace correctly when processed with `PyPDFLoader`. PDFPlumber produced significantly better extraction for the Deep Learning textbook used during testing.

### Hybrid retrieval

Semantic retrieval and BM25 complement each other and improve robustness for both conceptual and exact-term queries.

### Reranking

The cross-encoder provides a second relevance evaluation after initial retrieval and helps remove weaker candidates.

### Frontend rendering

Markdown and mathematical notation are preserved in the backend response and rendered by the frontend instead of being stripped.

---

# 17. Future Improvements

Potential improvements include:

* Semantic chunking
* Parent-child retrieval
* Contextual chunk headers
* Better metadata extraction
* Hybrid retrieval score fusion
* Query expansion
* Query rewriting
* Better reranking strategies
* Retrieval evaluation dataset
* Automated RAG evaluation
* Hallucination/faithfulness evaluation
* Streaming LLM responses
* Conversation memory
* Multiple document collections
* User-specific document isolation
* Improved citation generation

---

# 18. Evaluation Goal

The ultimate objective is to build a RAG pipeline where:

```text
                    User Question
                          ↓
                   Query Processing
                          ↓
                Hybrid Retrieval
                          ↓
                    Reranking
                          ↓
                 Relevant Context
                          ↓
                        LLM
                          ↓
                Grounded Answer
                          ↓
               Markdown / LaTeX UI
```

The system should retrieve the right information, provide sufficient context to the LLM, and generate an answer that is both useful and grounded in the uploaded documents.
