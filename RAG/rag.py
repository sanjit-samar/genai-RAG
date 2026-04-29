from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(persist_directory="create_db", embedding_function=embedding_model)

# default semantic search
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 4}
# )

# MMR (Max Marginal Relevance) Search type
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
)

llm = ChatMistralAI(model="mistral-small-2603")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a Helpful AI Assistant.
                Use Only provided context to answer the question.
                If the answer is not present in the context,
                say: "I coudnot find a suitable answer in the document"
            """,
        ),
        (
            "human",
            """Context: {context}
         Question: {question}
         """,
        ),
    ]
)

print("✅ Rag system completed successfully!")

print("🔥 Press 0 to exit !")

while True:
    query = input("You : ")
    if query == "0":
        break
    docs = retriever.invoke(query)

    if not docs:
        print("AI: No relevant documents found.")
        continue

    context = "\n".join([doc.page_content for doc in docs[:4]])

    final_prompt = prompt.invoke({"context": context, "question": query})

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")
