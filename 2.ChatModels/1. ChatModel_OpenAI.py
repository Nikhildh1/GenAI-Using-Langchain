from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model="gpt-4", temperature=0, max_completion_tokens=10)

ans=model.invoke("What is the capital of India?")

# This will print many things like content (Which will contain actual ans for the question), 
# additional kwargs, tokens, model_names and many other things.
print(ans)

# If we want only answer
print(ans.content)