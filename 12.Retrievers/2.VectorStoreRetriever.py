from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vectorStore=Chroma.from_documents(
    documents=documents,
    embedding=model,
    collection_name="my_collection"
)

retriever=vectorStore.as_retriever(search_kwargs={"k":2})

query="What is Chroma used for?"

results=retriever.invoke(query)

print(results)