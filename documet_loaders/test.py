from langchain_community.document_loaders import TextLoader

# Initialize the text loader with your file path
loader = TextLoader("./index.txt")

# Load the file contents into a list of Document objects
documents = loader.load()

# print(loader)
print(documents[0])