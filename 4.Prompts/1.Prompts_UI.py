from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

st.header("Research Tool")

# Static Prompt
# userInput=st.text_input("Enter your text")

# Dynamic Prompt
paperInput=st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
styleInput=st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template=load_prompt('3.Template.json')

# prompt=template.invoke({
#     'paper_input':paperInput,
#     'style_input':styleInput,
#     'length_input':length_input
# })


if st.button("Summarize"):
    chain=template | model
    result=chain.invoke({
        'paper_input':paperInput,
        'style_input':styleInput,
        'length_input':length_input
    })
    # prompt=template.invoke({
    #     'paper_input':paperInput,
    #     'style_input':styleInput,
    #     'length_input':length_input
    # })
    # result=model.invoke(prompt)
    st.write(result.content)