from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class Review(TypedDict):
    key_theme:Annotated[list[str],"Write down all the key themes discussed in review in a list"]
    summary: Annotated[str, "Give a brief description of summary"]
    sentiment:Annotated[Literal["pos","neg"], "Give the sentiment of review as positive or negative"]
    pros:Annotated[Optional[list[str]], "Write down all the pros mentioned in the review"]
    cons:Annotated[Optional[list[str]], "Write down all the cons mentioned in the review"]
    reviewer_name:Annotated[Optional[str],"Write name of the reviewer"]

structured_model=model.with_structured_output(Review)

result=structured_model.invoke(
    """
        I bought this laptop around two months ago for office work, coding, 
        and some casual gaming. The performance is really impressive because 
        applications open very fast and multitasking feels smooth even when 
        many browser tabs are running. The display quality is sharp and colors 
        look vibrant while watching movies. Battery backup is also decent and 
        easily lasts around 7 to 8 hours on normal usage.

        However, there are a few drawbacks too. The laptop heats up a little 
        during gaming sessions and the fan noise becomes noticeable. The speaker 
        quality is average and could have been better at this price range. 
        Also, the webcam quality is not very good for video meetings.

        Overall, I am satisfied with the purchase because the performance and 
        build quality are excellent for the price.

        Reviewed By Nikhil Dhawan
    """
)

print(result)