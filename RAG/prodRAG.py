# ============================================================
# PRODUCTION-READY RAG PIPELINE
# ============================================================
# This implementation includes:
#
# | Step | Component                    | Purpose                   |
# | ---- | ---------------------------- | ------------------------- |
# | 1    | Document Loading             | Load PDFs, docs           |
# | 2    | Document Chunking            | Split large docs          |
# | 3    | Metadata Support             | Add source/chunk metadata |
# | 4    | Embedding Caching            | Create/store embeddings   |
# | 5    | Hybrid Search                | BM25 + Vector retrieval   |
# | 6    | Conversation Memory          | Store chat history        |
# | 7    | Security / Prompt Protection | Validate query            |
# | 8    | Query Rewriting              | Improve retrieval query   |
# | 9    | Re-ranking                   | Improve retrieved results |
# | 10   | Source Citations             | Format retrieved chunks   |
# | 11   | Better Prompt Engineering    | Create prompt             |
# | 12   | Structured Retrieval Chain   | LCEL chain                |
# | 13   | Streaming Responses          | Stream answer             |
# | 14   | Evaluation Hooks             | Logging / metrics         |

#
# ============================================================

# =========================
# STEP 0 : IMPORTS
# =========================

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from sentence_transformers import CrossEncoder

from langchain_community.document_loaders import PyPDFLoader

import os
import hashlib

load_dotenv()

# ============================================================
# STEP 1 : DOCUMENT Loading/CHUNKING
# ============================================================
# WHY?
# Large documents cannot be embedded efficiently.
# Chunking improves semantic retrieval accuracy.
#
# Here we split documents into smaller chunks
# with overlap to preserve context continuity.
# ============================================================

loader = PyPDFLoader("sample.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(documents)

# ============================================================
# STEP 2 : ADD METADATA
# ============================================================
# WHY?
# Metadata helps:
# - source tracking
# - filtering
# - citations
# - enterprise document control
# ============================================================

for idx, doc in enumerate(split_docs):

    doc.metadata["chunk_id"] = idx
    doc.metadata["source"] = "sample.pdf"

# ============================================================
# STEP 12 : EMBEDDING CACHING
# ============================================================
# WHY?
# Avoid recomputing embeddings repeatedly.
#
# In production:
# - Redis
# - Disk cache
# - Vector cache
#
# Here we use persistent Chroma storage.
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# VECTOR DATABASE
# ============================================================

persist_directory = "production_rag_db"

if not os.path.exists(persist_directory):

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

else:

    vector_store = Chroma(
        persist_directory=persist_directory, embedding_function=embedding_model
    )

# ============================================================
# STEP 11 : HYBRID SEARCH
# ============================================================
# WHY?
# Vector Search:
# Good for semantic meaning
#
# BM25:
# Good for exact keywords
#
# Hybrid retrieval combines both.
# ============================================================

# ---------- VECTOR RETRIEVER ----------

vector_retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
)

# ---------- BM25 RETRIEVER ----------

bm25_retriever = BM25Retriever.from_documents(split_docs)

bm25_retriever.k = 4

# ---------- HYBRID / ENSEMBLE RETRIEVER ----------

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
)

# ============================================================
# STEP 6 : RE-RANKING
# ============================================================
# WHY?
# Retriever may return partially relevant chunks.
#
# Re-ranker improves ranking quality.
#
# CrossEncoder compares:
# (query, chunk)
#
# and gives better relevance score.
# ============================================================

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ============================================================
# STEP 4 : CONVERSATION MEMORY
# ============================================================
# WHY?
# Makes chatbot stateful.
#
# Remembers:
# - previous questions
# - follow-up conversations
# ============================================================

memory = ConversationBufferMemory(return_messages=True)

# ============================================================
# STEP 13 : SECURITY / PROMPT INJECTION PROTECTION
# ============================================================
# WHY?
# Prevent malicious prompts like:
#
# "Ignore previous instructions"
#
# Basic protection layer added here.
# ============================================================

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "reveal hidden prompt",
    "bypass security",
]


def is_safe_query(query):

    query_lower = query.lower()

    for pattern in BLOCKED_PATTERNS:

        if pattern in query_lower:
            return False

    return True


# ============================================================
# STEP 5 : QUERY REWRITING
# ============================================================
# WHY?
# Users ask vague questions.
#
# Query rewriting improves retrieval quality.
# ============================================================

rewriter_llm = ChatMistralAI(model="mistral-small-2603", temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user question into a clear standalone search query.

Question:
{question}
""")

# ============================================================
# MAIN LLM
# ============================================================

llm = ChatMistralAI(model="mistral-small-2603", temperature=0, streaming=True)

# ============================================================
# STEP 7 : BETTER PROMPT ENGINEERING
# ============================================================
# WHY?
# Strong prompts reduce hallucination.
# ============================================================

main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer.

Rules:
- Do not make assumptions
- Do not hallucinate
- If answer is unavailable say:
  "I could not find the answer in the provided documents."

Always provide source references if available.
""",
        ),
        (
            "human",
            """
Conversation History:
{history}

Context:
{context}

Question:
{question}
""",
        ),
    ]
)

# ============================================================
# STEP 10 : EVALUATION HOOKS
# ============================================================
# WHY?
# Production systems need evaluation.
#
# Frameworks:
# - Ragas
# - DeepEval
# - TruLens
#
# Here we add simple logging hooks.
# ============================================================


def evaluate_response(question, answer, docs):

    print("\n========== EVALUATION ==========")
    print(f"Question: {question}")
    print(f"Retrieved Chunks: {len(docs)}")
    print(f"Answer Length: {len(answer)}")
    print("================================")


# ============================================================
# STEP 3 : SOURCE CITATIONS
# ============================================================
# WHY?
# Users should know:
# - source file
# - chunk number
# - traceability
# ============================================================


def format_context(docs):

    formatted = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "NA")

        formatted.append(f"""
SOURCE: {source}
CHUNK_ID: {chunk_id}

CONTENT:
{doc.page_content}
""")

    return "\n\n".join(formatted)


# ============================================================
# STEP 6 : RE-RANK FUNCTION
# ============================================================


def rerank_documents(query, docs):

    pairs = [[query, doc.page_content] for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    reranked_docs = [doc for doc, score in scored_docs]

    return reranked_docs[:4]


# ============================================================
# STEP 8 : STRUCTURED RAG CHAIN
# ============================================================
# WHY?
# Cleaner architecture.
# Easier scaling.
# ============================================================

output_parser = StrOutputParser()

# ============================================================
# CHAT LOOP
# ============================================================

print("\n✅ Production Ready RAG System Started")
print("🔥 Press 0 to Exit\n")

while True:

    # ========================================================
    # USER INPUT
    # ========================================================

    query = input("You : ")

    if query == "0":
        break

    # ========================================================
    # SECURITY CHECK
    # ========================================================

    if not is_safe_query(query):

        print("\nAI: Unsafe query detected.")
        continue

    # ========================================================
    # QUERY REWRITING
    # ========================================================

    rewrite_chain = rewrite_prompt | rewriter_llm | output_parser

    rewritten_query = rewrite_chain.invoke({"question": query})

    print(f"\n🔍 Rewritten Query: {rewritten_query}")

    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    retrieved_docs = hybrid_retriever.invoke(rewritten_query)

    if not retrieved_docs:

        print("\nAI: No relevant documents found.")
        continue

    # ========================================================
    # RE-RANKING
    # ========================================================

    reranked_docs = rerank_documents(rewritten_query, retrieved_docs)

    # ========================================================
    # CONTEXT FORMATTING
    # ========================================================

    context = format_context(reranked_docs)

    # ========================================================
    # MEMORY FETCH
    # ========================================================

    chat_history = memory.load_memory_variables({})

    # ========================================================
    # STEP 8 : LCEL CHAIN
    # ========================================================

    rag_chain = main_prompt | llm | output_parser

    # ========================================================
    # STEP 9 : STREAMING RESPONSE
    # ========================================================
    # WHY?
    # Better user experience.
    # ========================================================

    print("\nAI: ", end="")

    response = rag_chain.invoke(
        {"history": chat_history, "context": context, "question": query}
    )

    print(response)

    # ========================================================
    # SAVE MEMORY
    # ========================================================

    memory.save_context({"input": query}, {"output": response})

    # ========================================================
    # STEP 3 : SOURCE DISPLAY
    # ========================================================

    print("\n📚 SOURCES USED:")

    for doc in reranked_docs:

        print(f"""
Source File : {doc.metadata.get('source')}
Chunk ID    : {doc.metadata.get('chunk_id')}
""")

    # ========================================================
    # STEP 10 : EVALUATION
    # ========================================================

    evaluate_response(query, response, reranked_docs)
