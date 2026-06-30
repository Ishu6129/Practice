from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

loader = TextLoader("../FILES/cricket.txt", encoding="utf-8")
docs = loader.load()

# print(type(docs)) # type:list
# print(len(docs)) # 1
# print(docs[0].page_content)
# print(docs[0].metadata) # source=cricket.txt

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  

prompt = PromptTemplate(
    template="Write a summary for the following text:\n\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"text": docs[0].page_content})

print("\nSummary:\n")
print(response)