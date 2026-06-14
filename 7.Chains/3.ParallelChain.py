from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model1=ChatHuggingFace(llm=llm)
model2=ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

template2=PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

template3=PromptTemplate(
    template="Merge the following notes and the quiz into a single document \n notes->{notes} , quiz-> {quiz}",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()

parallelChain=RunnableParallel({
    "notes":template1 | model1 | parser,
    "quiz":template2 | model2 | parser
})

mergeChain=template3 | model1 | parser

chain=parallelChain | mergeChain

text="""
Artificial Intelligence (AI) is one of the most rapidly growing fields in technology and computer science. It refers to the development of computer systems that can perform tasks which normally require human intelligence. These tasks include problem-solving, decision-making, speech recognition, language translation, visual perception, and learning from experience. AI systems are designed to simulate human intelligence using algorithms, data, and computational power.

Machine Learning (ML) is a subset of Artificial Intelligence that enables machines to learn from data without being explicitly programmed for every task. Instead of writing fixed instructions, developers train machine learning models using large datasets. These models identify patterns in data and make predictions or decisions based on those patterns. Machine Learning is widely used in recommendation systems, spam email filtering, fraud detection, medical diagnosis, and online search engines.

Deep Learning is a specialized branch of Machine Learning that uses artificial neural networks inspired by the structure of the human brain. Deep learning models consist of multiple layers of neurons that process information step by step. These models are highly effective for complex tasks such as image recognition, speech processing, natural language understanding, and autonomous driving. Technologies like virtual assistants, facial recognition systems, and self-driving cars heavily rely on deep learning.

Natural Language Processing (NLP) is another important area of AI that focuses on enabling computers to understand and process human language. NLP powers applications such as chatbots, translation tools, voice assistants, sentiment analysis systems, and text summarization tools. Modern AI chatbots use large language models trained on massive amounts of text data to generate human-like responses.

AI is transforming industries across the world. In healthcare, AI helps doctors detect diseases, analyze medical images, and recommend treatments. In finance, AI is used for fraud detection, algorithmic trading, and customer service automation. In education, AI-powered systems provide personalized learning experiences for students. In transportation, AI contributes to traffic prediction, route optimization, and autonomous vehicles. E-commerce companies use AI for product recommendations and customer behavior analysis.

Despite its advantages, Artificial Intelligence also presents several challenges and ethical concerns. AI systems require massive amounts of data, which raises privacy and security issues. Automation through AI may replace certain jobs, creating concerns about unemployment in some sectors. Bias in training data can also lead to unfair or discriminatory AI decisions. Therefore, researchers and governments are working on developing ethical AI systems that are transparent, fair, and safe for society.

The future of AI is expected to bring significant advancements in robotics, healthcare, education, cybersecurity, and scientific research. Researchers are exploring Artificial General Intelligence (AGI), a concept where machines could perform any intellectual task that humans can do. While AGI is still theoretical, current advancements in Generative AI and Agentic AI are already changing how people interact with technology. Tools powered by AI are helping developers write code, assisting students in learning, and enabling businesses to automate repetitive tasks more efficiently than ever before.
"""

result=chain.invoke({"text":text})

print(result)

chain.get_graph().print_ascii()