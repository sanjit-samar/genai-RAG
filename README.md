🔥Full RAG pipeline

(Indexing)
PDF → Load → Clean → Chunk → Embed → Store

(Query time)
User Query → Embed → retreval → LLM → Answer

🎯 In RAG Systems retrivers maily use 3 core retreval strategy

👉Similarity Search 
👉MMR - Max Marginal Relevance
👉MultiQuery Retriever


🎯 RAG project structure

rag_project/
│
├── app.py
│
├── config/
│   └── settings.py
│
├── ingestion/
│   ├── loader.py
│   ├── chunking.py
│   └── embedding.py
│
├── vectordb/
│   └── chroma_manager.py
│
├── retrieval/
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── query_rewriter.py
│
├── llm/
│   ├── model.py
│   ├── prompts.py
│   └── chains.py
│
├── memory/
│   └── chat_memory.py
│
├── security/
│   └── guardrails.py
│
├── evaluation/
│   └── evaluator.py
│
├── utils/
│   └── helpers.py
│
├── data/
│   └── sample.pdf
│
└── requirements.txt