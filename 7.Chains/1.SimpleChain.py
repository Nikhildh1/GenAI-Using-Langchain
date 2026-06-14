from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template="Give me 5 interesting facts on the topic {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain=template | model | parser

result=chain.invoke({"topic":"Black Hole"})

print(result)

chain.get_graph().print_ascii()