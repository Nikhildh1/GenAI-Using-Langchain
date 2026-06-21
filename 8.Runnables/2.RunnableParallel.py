from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

template1=PromptTemplate(
    template="Generate a tweet on topic {topic}",
    input_variables=["topic"]
)

template2=PromptTemplate(
    template="Generate a linkedin post content on topic {topic}",
    input_variables=["topic"]
)

parallelChain=RunnableParallel({
    "tweet":RunnableSequence(template1,model,parser),
    "linkedinPost":RunnableSequence(template2,model,parser)
})

result=parallelChain.invoke({"topic":"AI"})

print(result['tweet'])

print(result['linkedinPost'])