from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("GRU.pdf")

pdf = data.load()

print(pdf[len(pdf) - 1])
