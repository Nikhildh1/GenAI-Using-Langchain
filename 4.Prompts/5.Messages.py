from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

chatHistory=[
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    userInput=input("You: ")
    chatHistory.append(HumanMessage(content=userInput))
    if userInput=='exit':
        break
    result=model.invoke(userInput)
    chatHistory.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chatHistory)