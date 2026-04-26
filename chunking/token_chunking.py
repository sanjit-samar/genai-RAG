from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("document loaders/GRU.pdf").load()

splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)

chunks = splitter.split_documents(docs)


# view frist page chunks only
print(chunks[0].page_content)
