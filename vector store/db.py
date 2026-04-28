from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize with a Hugging Face model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(
        page_content="Python is a high-level, interpreted programming language known for its clear, English-like syntax that emphasizes code readability. It is a versatile",
        metadata={"source": "Python Book"},
    ),
    Document(
        page_content="FastAPI is a modern, high-performance web framework for building APIs with Python based on standard type hints. It is designed to be fast to code and production-ready",
        metadata={"source": "FastAPI Book"},
    ),
    Document(
        page_content="RAG is an AI framework that optimizes large language model output by referencing authoritative external knowledge bases before generating a response. This process allows models to provide more accurate, up-to-date, and contextually relevant answers without needing expensive retraining.",
        metadata={"source": "RAG Book"},
    ),
]

# It will invoked while running the file and this will duplicate in db

# vector_store = Chroma.from_documents(
#     documents=docs, embedding=embedding_model, persist_directory="chroma-db"
# )

persist_dir = "chroma-db"
# Check if the database already exists and is non-empty
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    # First run: create and persist the database
    vector_store = Chroma.from_documents(
        documents=docs, embedding=embedding_model, persist_directory=persist_dir
    )
else:
    # Subsequent runs: load existing database
    vector_store = Chroma(
        persist_directory=persist_dir, embedding_function=embedding_model
    )

result = vector_store.similarity_search("What is used for Fast Api?", k=2)

for R in result:
    print(R)
