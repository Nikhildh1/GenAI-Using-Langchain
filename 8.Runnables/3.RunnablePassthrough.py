from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

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

jokeGen=RunnableSequence(template1,model,parser)

template2=PromptTemplate(
    template="Explain the joke - {joke}",
    input_variables=["joke"]
)

parallelChain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "explaination":RunnableSequence(template2,model,parser)
})

finalChain=RunnableSequence(jokeGen,parallelChain)

result=finalChain.invoke({"topic":"AI"})

print(result['joke'])
print(result['explaination'])