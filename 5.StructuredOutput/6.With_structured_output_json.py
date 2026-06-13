from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

jsonSchema={
    "title":"Review",
    "type":"object",
    "properties":{
        "key_themes":{
            "type":"array",
            "items":{
                "type":"string"
            }
        },
        "summary":{
            "type":"string",
            "description": "A brief summary of the review"
        },
        "sentiment":{
            "type":"string",
            "enum":["pos","neg"],
            "description": "Return sentiment of the review either negative, positive or neutral"
        },
        "pros":{
            "type":["array","null"],
            "items":{
                "type":"string",
            },
            "description": "Write down all the pros inside a list"
        },
        "cons":{
            "type":["array","null"],
            "items":{
                "type":"string",
            },
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model=model.with_structured_output(jsonSchema)

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