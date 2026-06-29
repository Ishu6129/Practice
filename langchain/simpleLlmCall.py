from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_1=PromptTemplate(
    template="write about {topic} in 200 words\n",
    input_variables=['topic']
)

topic=input("Enter your topic here : ")

parser=StrOutputParser()

chain=prompt_1|model|parser

result=chain.invoke(topic)

print(result)