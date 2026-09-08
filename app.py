from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()

llm = ChatOpenRouter(
    model="openrouter/free",
    max_tokens=500
)

embedding_models = MistralAIEmbeddings()

vectorstore = Chroma(persist_directory="vector_store/chroma_db",embedding_function=embedding_models)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

while True:

    ques = input("")
    if ques == "exit":
        break
    
    docs = retriever.invoke(ques)
    
    # 2. Convert documents into context
    context = "\n\n".join(doc.page_content for doc in docs)
    
    messages = prompt.invoke({
        "context": context,
        "question": ques
    })
    
    response = llm.invoke(messages)
    
    print("===CONTENT===")
    
    print(response.content)