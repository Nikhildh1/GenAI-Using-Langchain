from langchain_community.document_loaders import WebBaseLoader
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

url='https://www.flipkart.com/apple-iphone-17-pro-cosmic-orange-256-gb/p/itm76fe37ca9ea8c?pid=MOBHFN6YR8HF5BQ9&lid=LSTMOBHFN6YR8HF5BQ9RBYDOE&marketplace=FLIPKART&store=tyy%2F4io&srno=b_1_1&otracker=browse&fm=organic&iid=b899dec1-805e-4600-8106-1b31723be41a.MOBHFN6YR8HF5BQ9.SEARCH&ppt=None&ppn=None&ssid=fvz4ph5fls0000001782058337552&ov_redirect=true'
loader=WebBaseLoader(url)

docs=loader.load()

parser=StrOutputParser()

template=PromptTemplate(
    template="What is the price of the following product \n {product}",
    input_variables=["product"]
)

chain=template|model|parser

print(chain.invoke({"product":docs[0].page_content}))
