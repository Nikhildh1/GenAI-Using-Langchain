from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=50
)

model=ChatHuggingFace(llm=llm)

class Feeback(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="Give the sentiment of the feedback as positive or negative")

parser=PydanticOutputParser(pydantic_object=Feeback)

strParser=StrOutputParser()

template1=PromptTemplate(
    template="Give the sentiment of the following feedback text into positive or negative \n {text} \n {format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

template2=PromptTemplate(
    template="Give an appropriate response to this positive feedback  \n {text}",
    input_variables=["text"]    
)

template3=PromptTemplate(
    template="Give an appropriate response to this negative feedback  \n {text}",
    input_variables=["text"]    
)

chain=template1 | model | parser

branchChain=RunnableBranch(
    (lambda x:x.sentiment=="positive",template2 | model | strParser),
    (lambda x:x.sentiment=="negative",template3 | model | strParser),
    RunnableLambda(lambda x:"Could not find the sentiment")
)

finalChain=chain | branchChain

ans=finalChain.invoke({"text":"This is a terrible phone"})

print(ans)