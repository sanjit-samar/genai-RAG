# ============================================================
# PRODUCTION-READY RAG PIPELINE (corrected)
# ============================================================
# | Step | Component                    | Purpose                   |
# | ---- | ----------------------------- | ------------------------- |
# | 1    | Document Loading             | Load PDFs, docs           |
# | 2    | Document Chunking            | Split large docs          |
# | 3    | Metadata Support             | Add source/chunk metadata |
# | 4    | Embedding Caching            | Create/store embeddings   |
# | 5    | Hybrid Search                | BM25 + Vector retrieval   |
# | 6    | Conversation Memory          | Store chat history        |
# | 7    | Security / Prompt Protection | Validate query             |
# | 8    | Query Rewriting              | Improve retrieval query   |
# | 9    | Re-ranking                   | Improve retrieved results |
# | 10   | Source Citations             | Format retrieved chunks   |
# | 11   | Better Prompt Engineering    | Create prompt             |
# | 12   | Structured Retrieval Chain   | LCEL chain                |
# | 13   | Streaming Responses          | Stream answer              |
# | 14   | Evaluation Hooks             | Logging / metrics         |
#
# Fixes applied vs. the original draft:
#   - ConversationBufferMemory was used but never imported (and is legacy/
#     deprecated in current LangChain). Replaced with ChatMessageHistory,
#     which was already imported but unused.
#   - EnsembleRetriever / BM25Retriever were imported from
#     langchain_community.retrievers, which does not export them. They now
#     come from langchain.retrievers (the correct home).
#   - memory.load_memory_variables({}) returned a dict, but the dict itself
#     was being passed into the prompt's {history} slot. It's now formatted
#     into a plain string first.
#   - streaming=True was set on the LLM but .invoke() was called, so nothing
#     actually streamed. Replaced with rag_chain.stream() and incremental
#     printing.
#   - No error handling anywhere; a missing PDF, a model load failure, or
#     an API error would kill the whole loop. Added try/except around the
#     risky steps with clear messages.
#   - Step labels in comments now match the table above (they previously
#     jumped around and repeated, e.g. two things both labeled "Step 6").
# ============================================================

# =========================
# STEP 0 : IMPORTS
# =========================

import os
import sys

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

# NOTE: EnsembleRetriever and BM25Retriever live in langchain.retrievers,
# not langchain_community.retrievers. If you're on LangChain 1.0+, these
# may instead live in langchain_classic.retrievers -- check your installed
# version if this import fails.
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

PDF_PATH = "sample.pdf"
PERSIST_DIRECTORY = "production_rag_db"

# ============================================================
# STEP 1 : DOCUMENT LOADING
# ============================================================
# WHY?
# Large documents cannot be embedded efficiently as a single blob,
# and we need the raw pages before we can chunk them.
# ============================================================

if not os.path.exists(PDF_PATH):
    print(
        f"❌ Could not find '{PDF_PATH}'. Place the file next to this "
        f"script, or update PDF_PATH, then re-run."
    )
    sys.exit(1)

try:
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
except Exception as e:
    print(f"❌ Failed to load '{PDF_PATH}': {e}")
    sys.exit(1)

# ============================================================
# STEP 2 : DOCUMENT CHUNKING
# ============================================================
# WHY?
# Chunking improves semantic retrieval accuracy and keeps each
# chunk small enough to embed and to fit into the LLM's context.
# Overlap preserves continuity across chunk boundaries.
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

if not split_docs:
    print(
        "❌ No chunks were produced from the document. Is the PDF empty "
        "or scanned (image-only) with no extractable text?"
    )
    sys.exit(1)

# ============================================================
# STEP 3 : METADATA SUPPORT
# ============================================================
# WHY?
# Metadata helps with source tracking, filtering, citations, and
# enterprise document control.
# ============================================================

for idx, doc in enumerate(split_docs):
    doc.metadata["chunk_id"] = idx
    doc.metadata["source"] = PDF_PATH

# ============================================================
# STEP 4 : EMBEDDING CACHING / VECTOR DATABASE
# ============================================================
# WHY?
# Avoid recomputing embeddings on every run. In production you'd
# typically back this with Redis or a disk cache; here we rely on
# Chroma's own persistent storage.
# ============================================================

try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
except Exception as e:
    print(f"❌ Failed to load the embedding model: {e}")
    sys.exit(1)

try:
    if not os.path.exists(PERSIST_DIRECTORY):
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY,
        )
    else:
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model,
        )
except Exception as e:
    print(f"❌ Failed to initialize the vector store: {e}")
    sys.exit(1)

# ============================================================
# STEP 5 : HYBRID SEARCH
# ============================================================
# WHY?
# Vector search is good for semantic meaning; BM25 is good for
# exact keyword matches. Combining both (hybrid retrieval) covers
# more query types than either alone.
# ============================================================

vector_retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
)

bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 4

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
)

# ============================================================
# STEP 6 : CONVERSATION MEMORY
# ============================================================
# WHY?
# Makes the chatbot stateful -- it remembers previous questions
# and answers within the session. ChatMessageHistory is the
# current, non-deprecated way to do this (ConversationBufferMemory
# is legacy).
# ============================================================

chat_memory = ChatMessageHistory()


def format_history(history: ChatMessageHistory) -> str:
    """Render stored messages as plain text for the prompt."""
    if not history.messages:
        return "(no previous conversation)"

    lines = []
    for msg in history.messages:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


# ============================================================
# STEP 7 : SECURITY / PROMPT INJECTION PROTECTION
# ============================================================
# WHY?
# Prevent obvious prompt-injection attempts such as
# "ignore previous instructions". This is a basic keyword filter,
# not a real defense -- it's trivially bypassed by paraphrasing.
# For production, pair this with an actual moderation/guardrail
# model rather than relying on substring matching alone.
# ============================================================

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "reveal hidden prompt",
    "bypass security",
]


def is_safe_query(query: str) -> bool:
    query_lower = query.lower()
    return not any(pattern in query_lower for pattern in BLOCKED_PATTERNS)


# ============================================================
# STEP 8 : QUERY REWRITING
# ============================================================
# WHY?
# Users often ask vague or context-dependent questions. Rewriting
# the query into a clear, standalone search query improves
# retrieval quality.
# ============================================================

rewriter_llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user question into a clear standalone search query.

Question:
{question}
""")

rewrite_chain = rewrite_prompt | rewriter_llm | StrOutputParser()

# ============================================================
# MAIN LLM
# ============================================================

llm = ChatMistralAI(model="mistral-small-latest", temperature=0, streaming=True)

# ============================================================
# STEP 9 : RE-RANKING
# ============================================================
# WHY?
# The retriever may return partially relevant chunks. A cross-
# encoder compares (query, chunk) pairs directly and gives a more
# reliable relevance score than the initial retrieval score.
# ============================================================

try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    print(f"❌ Failed to load the reranker model: {e}")
    sys.exit(1)


def rerank_documents(query, docs, top_n=4):
    if not docs:
        return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]


# ============================================================
# STEP 10 : SOURCE CITATIONS
# ============================================================
# WHY?
# Users should be able to trace an answer back to its source file
# and chunk for verification.
# ============================================================


def format_context(docs) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "NA")
        formatted.append(
            f"SOURCE: {source}\nCHUNK_ID: {chunk_id}\n\nCONTENT:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


# ============================================================
# STEP 11 : BETTER PROMPT ENGINEERING
# ============================================================
# WHY?
# A strong system prompt reduces hallucination and keeps the model
# grounded in the retrieved context.
# ============================================================

main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer.

Rules:
- Do not make assumptions
- Do not hallucinate
- If the answer is unavailable, say:
  "I could not find the answer in the provided documents."

Always provide source references if available.""",
        ),
        (
            "human",
            """Conversation History:
{history}

Context:
{context}

Question:
{question}""",
        ),
    ]
)

# ============================================================
# STEP 12 : STRUCTURED RETRIEVAL CHAIN (LCEL)
# ============================================================
# WHY?
# A declarative chain (prompt | llm | parser) is cleaner and
# easier to extend than manual orchestration.
# ============================================================

rag_chain = main_prompt | llm | StrOutputParser()

# ============================================================
# STEP 14 (helper) : EVALUATION HOOK
# ============================================================
# WHY?
# Production systems need evaluation. Frameworks like Ragas,
# DeepEval or TruLens can replace this simple logging hook.
# ============================================================


def evaluate_response(question, answer, docs):
    print("\n========== EVALUATION ==========")
    print(f"Question: {question}")
    print(f"Retrieved Chunks: {len(docs)}")
    print(f"Answer Length: {len(answer)}")
    print("================================")


# ============================================================
# CHAT LOOP
# ============================================================

print("\n✅ Production Ready RAG System Started")
print("🔥 Press 0 to Exit\n")

while True:

    # ------------------------------------------------------
    # USER INPUT
    # ------------------------------------------------------
    try:
        query = input("You : ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        break

    if query == "0":
        break

    if not query:
        continue

    # ------------------------------------------------------
    # STEP 7 : SECURITY CHECK
    # ------------------------------------------------------
    if not is_safe_query(query):
        print("\nAI: Unsafe query detected.")
        continue

    # ------------------------------------------------------
    # STEP 8 : QUERY REWRITING
    # ------------------------------------------------------
    try:
        rewritten_query = rewrite_chain.invoke({"question": query})
    except Exception as e:
        print(
            f"\n⚠️ Query rewriting failed ({e}); falling back to the "
            f"original question."
        )
        rewritten_query = query

    print(f"\n🔍 Rewritten Query: {rewritten_query}")

    # ------------------------------------------------------
    # STEP 5 : HYBRID RETRIEVAL
    # ------------------------------------------------------
    try:
        retrieved_docs = hybrid_retriever.invoke(rewritten_query)
    except Exception as e:
        print(f"\n❌ Retrieval failed: {e}")
        continue

    if not retrieved_docs:
        print("\nAI: No relevant documents found.")
        continue

    # ------------------------------------------------------
    # STEP 9 : RE-RANKING
    # ------------------------------------------------------
    reranked_docs = rerank_documents(rewritten_query, retrieved_docs)

    # ------------------------------------------------------
    # STEP 10 : CONTEXT FORMATTING
    # ------------------------------------------------------
    context = format_context(reranked_docs)

    # ------------------------------------------------------
    # STEP 6 : MEMORY FETCH
    # ------------------------------------------------------
    history_text = format_history(chat_memory)

    # ------------------------------------------------------
    # STEP 12 + 13 : INVOKE CHAIN WITH REAL STREAMING
    # ------------------------------------------------------
    print("\nAI: ", end="", flush=True)

    response_chunks = []
    try:
        for chunk in rag_chain.stream(
            {"history": history_text, "context": context, "question": query}
        ):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        continue

    response = "".join(response_chunks)
    print()  # newline after the streamed answer

    # ------------------------------------------------------
    # STEP 6 : SAVE MEMORY
    # ------------------------------------------------------
    chat_memory.add_user_message(query)
    chat_memory.add_ai_message(response)

    # ------------------------------------------------------
    # STEP 10 : SOURCE DISPLAY
    # ------------------------------------------------------
    print("\n📚 SOURCES USED:")
    for doc in reranked_docs:
        print(f"  Source File : {doc.metadata.get('source')}")
        print(f"  Chunk ID    : {doc.metadata.get('chunk_id')}\n")

    # ------------------------------------------------------
    # STEP 14 : EVALUATION
    # ------------------------------------------------------
    evaluate_response(query, response, reranked_docs)
