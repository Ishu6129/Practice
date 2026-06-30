from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3,
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season.

The Indian Premier League (IPL) is the biggest cricket league in the world.

Terrorism is a big danger to peace and safety.
"""

docs = text_splitter.create_documents([sample])

print(len(docs))

for i, doc in enumerate(docs, 1):
    print(f"\nChunk {i}:")
    print(doc.page_content)