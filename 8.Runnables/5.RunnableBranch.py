from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch

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
    template="Write a report on the topic {topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="Summarize the text in less than 100 words {text}",
    input_variables=["text"]
)

jokeGen=RunnableSequence(template1,model,parser)

conditionalChain=RunnableBranch(
    (lambda x: len(x.split())>100,RunnableSequence(template2,model,parser)),
    RunnablePassthrough()
)

finalChain=RunnableSequence(jokeGen,conditionalChain)

print(finalChain.invoke({"topic":"AI"}))

finalChain.get_graph().print_ascii