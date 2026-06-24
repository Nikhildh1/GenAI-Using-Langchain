from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
    a:int=Field(required=True, description="The first number for multiplication")
    b:int=Field(required=True, description="The second number for multiplication")

class MultiplyTool(BaseTool):
    name:str="multiply"
    description:str="Multiplication of 2 numbers"
    args_schema:Type[BaseModel]=MultiplyInput
    def _run(self, a:int, b:int) -> int:
        return a*b

multiplyTool=MultiplyTool()

ans=multiplyTool.invoke({"a":10,"b":20})
print(ans)
print(multiplyTool.name)
print(multiplyTool.description)
print(multiplyTool.args)