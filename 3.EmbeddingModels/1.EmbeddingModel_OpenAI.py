from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

ans=model.embed_query("What is the capital of India?")

# It will print the embedding vector generated for the input query
print(str(ans))