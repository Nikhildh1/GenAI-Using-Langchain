from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser=StrOutputParser()

embeddingModel=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

def format_docs(retrievedDocs):
    context='\n\n'.join(doc.page_content for doc in retrievedDocs)
    return context

# STEP-1a : INDEXING (Docuement Ingestion)

videoId='Gfr50f6ZBvo'
try:
    yt_api=YouTubeTranscriptApi()
    transcriptList=yt_api.fetch(video_id=videoId, languages=['en'])
    transcript=" ".join(snippet.text for snippet in transcriptList)
    # print(transcript)
except TranscriptsDisabled:
    print("No Transcript available for this video.")

# STEP-1b : INDEXING (Text Splitting)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks=splitter.create_documents([transcript])

# print(len(chunks))

# STEP-1c : INDEXING (Embedding Generation & Storing in Vector Store)

vectorStore=FAISS.from_documents(chunks,embeddingModel)

# print(vectorStore.index_to_docstore_id)


# STEP-2 (Retrieval)

retrievar=vectorStore.as_retriever(search_type='similarity', search_kwargs={"k":4})

# ans=retrievar.invoke("What is deepmind")

# print(ans)

#  STEP-3 (Augmentation)

prompt=PromptTemplate(
    template="""
        You are a helpful AI assistant.
        Answer only from the provided context.
        If the context is insufficient, just say that Something went wrong

        {context}
        Question: {question}
    """,
    input_variables=["context","question"]
)

question="Is the topic of alient discussed in this video? If yes, then what was discussed."

# retrievedDocs=retrievar.invoke(question)

# context='\n\n'.join(doc.page_content for doc in retrievedDocs)

# finalPrompt=prompt.invoke({"context":context,"question":question})

# ans=model.invoke(finalPrompt)

# print(ans)

parallelChain=RunnableParallel({
    "question":RunnablePassthrough(),
    "context":retrievar | RunnableLambda(format_docs)
})

finalChain=parallelChain | prompt | model | parser

result=finalChain.invoke("Can you summarize this video")

print(result)