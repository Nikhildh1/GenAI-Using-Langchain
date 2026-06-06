from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# ans=model.embed_query("India is the capital of India")


documents=[
    "The capital of India is New Delhi",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

ans=model.embed_documents(documents)

print(str(ans))