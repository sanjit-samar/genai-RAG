from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Text Loader -----
# text_loader = TextLoader("document loaders/notes.txt")
# docs = text_loader.load()

# PDF Loader ------
pdf_loader = PyPDFLoader("document loaders/GRU.pdf")
docs = pdf_loader.load()

template = ChatPromptTemplate.from_messages(
    [
        # System Prompt for Text file
        # ("system", "what is name of collage and when it got established ?"),
        # System Prompt for PDF file
        ("system", "You are a AI for best Sumarization of provided PDF file"),
        ("user", "{data}"),
    ]
)

llm = ChatMistralAI(model="mistral-small-2603")

chat_prompt = template.format_messages(data=docs[14].page_content)

result = llm.invoke(chat_prompt)

print(result.content)
