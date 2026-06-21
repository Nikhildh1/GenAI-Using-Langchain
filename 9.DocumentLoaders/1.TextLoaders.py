from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

loader=TextLoader('cricket.txt',encoding='utf-8')

docs=loader.load()

# print(docs[0].page_content)

template=PromptTemplate(
    template="Write a summary for following poem \n {poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()

chain=template|model|parser

print(chain.invoke({"poem":docs[0].page_content}))