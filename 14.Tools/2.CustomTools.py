from langchain_community.tools import tool

@tool
def multiply(a:int,b:int) -> int:
    "Multiply 2 numbers"
    return a*b

result=multiply.invoke({"a":4,"b":3})

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)