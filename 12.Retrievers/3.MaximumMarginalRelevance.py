from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vectorStore=FAISS.from_documents(
    documents=docs,
    embedding=model
)

retriever=vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3,"lambda_mult":0.5}
)

query="What is Langchain?"

results=retriever.invoke(query)

print(results)