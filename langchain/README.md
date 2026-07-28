# LangChain Practice

This folder contains hands-on LangChain examples and notebooks for building LLM applications.

## What is inside

- [AI_Agent/](AI_Agent/) - agent-style examples and experiments.
- [CX_Chains/](CX_Chains/) - sequential, parallel, and conditional chain examples.
- [CX_DocLoaders/](CX_DocLoaders/) - loaders for CSV, text, PDF, and web data.
- [CX_Models/](CX_Models/) - model and embedding examples for OpenAI, Gemini, Groq, and Hugging Face.
- [CX_OutputParsers/](CX_OutputParsers/) - output parsing patterns with JSON and Pydantic.
- [CX_Prompts/](CX_Prompts/) - prompt templates and chat prompt helpers.
- [CX_Retrievers/](CX_Retrievers/) - retrieval examples and experiments.
- [CX_Runnable/](CX_Runnable/) - runnable composition examples.
- [CX_StructuredOutput/](CX_StructuredOutput/) - structured output examples.
- [CX_TextSplitting/](CX_TextSplitting/) - text splitting utilities.
- [CX_Tools/](CX_Tools/) - tool-based examples.
- [CX_VectorStore/](CX_VectorStore/) - Chroma-based vector store examples and a local database folder.
- [FILES/](FILES/) - sample data files used by the examples.
- [YtChatApp/](YtChatApp/) - a YouTube chat application example.
- [simpleLlmCall.py](simpleLlmCall.py) and [test.ipynb](test.ipynb) - quick starter files.

## Setup

- Create a virtual environment and install dependencies with `pip install -r requirements.txt`.
- Many examples need API keys in a `.env` file.
- The local vector store under [CX_VectorStore/my_chroma_db/](CX_VectorStore/my_chroma_db/) is used by notebook examples.
