from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

def wordCounter(text):
    return len(text.split())

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

template1=PromptTemplate(
    template="Write a joke on the topic {topic}",
    input_variables=["topic"]
)

jokeGen=RunnableSequence(template1,model,parser)

parallelChain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "wordCount":RunnableLambda(wordCounter)
})

finalChain=RunnableSequence(jokeGen,parallelChain)

print(finalChain.invoke({"topic":"AI"}))