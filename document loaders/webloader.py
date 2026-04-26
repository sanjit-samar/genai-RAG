from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os

load_dotenv()

IPHONE_URL = os.getenv("IPHONE_URL")

data = WebBaseLoader(IPHONE_URL)

docs = data.load()

print(docs[0].page_content)
