from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-latest",  # pyright: ignore[reportCallIssue]
    max_tokens=500
)


# Initialize the text loader with your file path
loader = PyPDFLoader("./rnn.pdf")

# Load the file contents into a list of Document objects
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = text_splitter.split_text(documents[0].page_content)

template = ChatPromptTemplate.from_messages([("system", "You are an AI that summarizes text"), ("human", "{text}")])

prompt = template.format_messages(text =  documents[0].page_content)
res = llm.invoke(prompt)  

print(res.content)
# print(loader)
# print(documents[0])
# print("\n____________________________________________")
# print(documents[0].metadata)
# print("\n____________________________________________")
# print(documents[0].page_content)