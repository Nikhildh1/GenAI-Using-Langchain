from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs=[doc1,doc2,doc3,doc4,doc5]

vectorStore=Chroma(
    embedding_function=model,
    persist_directory="chroma_db",
    collection_name="sample"
)

# *******************************************************************************************

# addedDocs=vectorStore.add_documents(docs)

# print(addedDocs)

# *******************************************************************************************

# getVector=vectorStore.get(include=['embeddings','documents','metadatas'])

# print(getVector)

# *******************************************************************************************

# getBowler=vectorStore.similarity_search(
#     query="Who among these are bowler?",
#     k=1
# )

# print(getBowler)

# *******************************************************************************************

# getBowler=vectorStore.similarity_search_with_score(
#     query="Who among these are bowler?",
#     k=2
# )

# print(getBowler)

# *******************************************************************************************

# getPlayers=vectorStore.similarity_search_with_score(
#     query='',
#     filter={"team":"Chennai Super Kings"}
# )

# print(getPlayers)

# *******************************************************************************************

# updatedDoc=Document(
#     page_content="MSD is the best of all the captains.",
#     metadata={"team":"Chennai Super Kings"}
# )

# vectorStore.update_document(document_id='a4c5278a-5af7-4df1-9126-84ba4ce41b8a', document=updatedDoc)

# getAll=vectorStore.get(include=['embeddings','documents','metadatas'])

# print(getAll)

# *******************************************************************************************

# vectorStore.delete(ids=['3d8a0dbc-a016-473d-aeae-76ecc3862762'])

# getAll=vectorStore.get(include=['embeddings','documents','metadatas'])

# print(getAll)