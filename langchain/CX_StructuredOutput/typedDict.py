from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

person_1:Person={'name':'ishu','age':12}
print(person_1)
