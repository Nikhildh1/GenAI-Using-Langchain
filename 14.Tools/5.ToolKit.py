from langchain_community.tools import tool

@tool
def addTwoNumbers(a:int,b:int)->int:
    "Add 2 number"
    return a+b

@tool
def multiplyTwoNumbers(a:int,b:int)->int:
    "Multiply 2 number"
    return a*b

class MathToolkit:
    def get_tools(self):
        return[addTwoNumbers,multiplyTwoNumbers]
    
toolkit=MathToolkit()
tools=toolkit.get_tools()

for tool in tools:
    print(tool.name,"=>",tool.description)