from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

newPerson: Person = {"name":"Nikhil","age":23}

print(newPerson)