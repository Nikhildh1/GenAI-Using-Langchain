from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Nikhil'
    age: Optional[int] = None
    email:EmailStr
    cgpa:float = Field(gt=0,lt=10,default=9.55,description="CGPA represents overall score of a student during all semesters")

newStudent={'name':'Nikhil','age':23,'email':'nikhildhawan356@gmail.com','cgpa':9.3}

student=Student(**newStudent)

print(dict(student))
print(student.model_dump_json())