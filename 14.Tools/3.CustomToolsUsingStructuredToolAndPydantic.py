from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a:int=Field(required=True, description="The first number for multiplication")
    b:int=Field(required=True, description="The second number for multiplication")

def multiply(a:int,b:int) -> int:
    return a*b

multiplyTool=StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Two number multiplication",
    args_schema=MultiplyInput
)

ans=multiplyTool.invoke({"a":5,"b":2})
print(ans)
print(multiplyTool.name)
print(multiplyTool.description)
print(multiplyTool.args)