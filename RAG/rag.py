from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a AI for best Sumarization of provided PDF file"),
        ("user", "{data}"),
    ]
)

llm = ChatMistralAI(model="mistral-small-2603")
