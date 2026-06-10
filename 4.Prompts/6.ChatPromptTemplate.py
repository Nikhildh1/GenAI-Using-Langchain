from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

chat_template=ChatPromptTemplate([
    # SystemMessage(content="You are a helpful {domain} expert"),
    # HumanMessage(content="Explain in simple terms, what is {topic}")
    ("system","You are a helpful {domain} expert"),
    ("human","Explain in simple terms, what is {topic}")
])

prompt=chat_template.invoke({"domain":"cricket","topic":"no ball"})

ans=model.invoke(prompt)

print(ans.content)