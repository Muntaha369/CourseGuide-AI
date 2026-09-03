from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# Create an embedding model
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)
# Test docs
docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

# Create the vector store and mention which docs to work on and with which model
vector_store = Chroma.from_documents(
    documents = docs,
    embedding= embeddings,
    persist_directory= "chroma-db"
)

result = vector_store.similarity_search("what is used for data analysis?",k=2)

print(result)
