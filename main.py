from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

text_loader = TextLoader("document loaders/notes.txt")
docs = text_loader.load()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "what is name of collage and when it got established ?"),
        ("user", "{data}"),
    ]
)

llm = ChatMistralAI(model="mistral-small-2603")

chat_prompt = template.format_messages(data=docs[0].page_content)

result = llm.invoke(chat_prompt)

print(result.content)
