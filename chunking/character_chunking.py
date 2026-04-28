from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

docs = TextLoader("document loaders/notes.txt").load()

splitter = CharacterTextSplitter(separator="", chunk_size=10, chunk_overlap=1)

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()


# The main difference is that CharacterTextSplitter is a simple length-based splitter
# that cuts text at fixed character boundaries (often ignoring sentence or word structure),
# while RecursiveCharacterTextSplitter is smarter and tries to preserve natural text boundaries
# (paragraphs → sentences → words → characters) by recursively breaking down text
# until chunks fit the desired size.
