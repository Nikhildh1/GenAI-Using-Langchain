from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

template1=PromptTemplate(
    template="Write a joke on the topic {topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="Explain the joke - {joke}",
    input_variables=["joke"]
)

chain=RunnableSequence(template1,model,parser,template2,model,parser)

print(chain.invoke({"topic":"AI"}))