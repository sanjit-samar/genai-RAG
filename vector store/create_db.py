# load pdf
# chunking the pdf
# creating embeddings
# store in chroma db
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("RAG/Learning_Python.pdf")
documents = loader.load()

# 2. Add metadata (VERY IMPORTANT)
for i, doc in enumerate(documents):
    doc.metadata["page_number"] = i + 1
    doc.metadata["source"] = "Learning_Python.pdf"

# 3. Chunking (better overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

chunks = splitter.split_documents(documents)

# 4. Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Create empty DB first
vector_store = Chroma(persist_directory="create_db", embedding_function=embedding_model)

# 6. Batch insert (CRITICAL for large PDFs)
batch_size = 500

for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    vector_store.add_documents(batch)
    print(f"Processed batch {i // batch_size + 1}")

# 7. Persist DB for older version of chroma from version 0.4+ no need to do manually
# vector_store.persist()

print("✅ Embedding completed successfully!")
