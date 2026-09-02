from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-latest",  # pyright: ignore[reportCallIssue]
    max_tokens=500
)

template = ChatPromptTemplate.from_messages([("system", "You are an AI that summarizes text"), ("human", "{text}")])

# Initialize the text loader with your file path
loader = TextLoader("./index.txt")

# Load the file contents into a list of Document objects
documents = loader.load()

prompt = template.format_messages(text =  documents[0].page_content)
res = llm.invoke(prompt)  

print(res.content)

# print(loader)
# print(documents[0])
# print("\n____________________________________________")
# print(documents[0].metadata)
# print("\n____________________________________________")
# print(documents[0].page_content)