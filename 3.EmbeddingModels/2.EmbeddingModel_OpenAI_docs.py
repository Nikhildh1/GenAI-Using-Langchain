from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents=[
    "The capital of India is New Delhi",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

ans=model.embed_query(documents)

# It will print the embedding vector generated for the input query for all the inputs in the array
print(str(ans))