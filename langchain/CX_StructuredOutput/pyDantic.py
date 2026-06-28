from pydantic import BaseModel

class Student(BaseModel):
    name:str 


student_1={'name':'ishu'}
student=Student(**student_1)

print(student)
print(student.model_dump_json())