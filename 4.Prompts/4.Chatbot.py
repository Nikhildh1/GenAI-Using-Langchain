from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

chatHistory=[]

while True:
    user_input=input("You: ")
    chatHistory.append(user_input)
    if user_input=='exit':
        break
    ans=model.invoke(chatHistory)
    chatHistory.append(ans)
    print("AI: ",ans.content)
