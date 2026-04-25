from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

llm = ChatMistralAI(model="mistral-small-2603")

result = llm.invoke("hello")

print(result)
