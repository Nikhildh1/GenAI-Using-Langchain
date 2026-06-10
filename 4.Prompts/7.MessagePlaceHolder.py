from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

chat_template=ChatPromptTemplate([
    ("system","You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name='chatHistory'),
    ("human",'{query}')
])

chatHistory=[]

with open('chat_history.txt') as f:
    chatHistory.extend(f.readlines())

prompt=chat_template.invoke({"chatHistory":chatHistory,"query":"Where is my refund"})