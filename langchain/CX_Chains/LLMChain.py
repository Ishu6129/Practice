from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(
    template="Write about {topic} in 200 words.",
    input_variables=["topic"]
)

topic = input("Enter your topic here: ")

chain = LLMChain(
    llm=model,
    prompt=prompt
)

result = chain.invoke(topic)

print(result['text'])   