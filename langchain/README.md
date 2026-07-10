# LangChain Practice

A hands-on learning repository for building AI applications with LangChain, LangGraph, and various LLM providers.

This project contains practical examples, notebooks, and small experiments covering prompts, chains, document loaders, embeddings, vector stores, retrievers, tools, structured outputs, and runnable workflows.

## What is included

- Chains and orchestration:
  - sequential, parallel, and conditional chains
  - runnable pipelines and branching logic
- Prompt engineering:
  - prompt templates
  - chat prompt examples
  - chat history handling
- Document processing:
  - text, CSV, PDF, directory, and web loaders
- Models and embeddings:
  - OpenAI, Google Gemini, Groq, and Hugging Face examples
  - embedding generation and similarity search
- Vector storage and retrieval:
  - Chroma-based vector store examples
  - retriever experiments
- Structured outputs and tools:
  - JSON and Pydantic output parsing
  - tool usage examples

## Tech stack

- Python
- LangChain
- LangGraph
- Chroma
- OpenAI
- Google Generative AI
- Groq
- Hugging Face
- Jupyter Notebooks

## Setup

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Add your API keys to a `.env` file
   ```env
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Run an example

```bash
python simpleLlmCall.py
```

## Project structure

- AI_Agent/ - agent-related examples
- CX_Chains/ - chain implementations
- CX_DocLoaders/ - document loading examples
- CX_Models/ - model integrations
- CX_Prompts/ - prompt templates and chat prompt logic
- CX_Retrievers/ - retriever examples
- CX_Runnable/ - runnable workflow examples
- CX_StructuredOutput/ - structured output parsers
- CX_TextSplitting/ - text chunking examples
- CX_Tools/ - tool-based examples
- CX_VectorStore/ - vector database examples

## Notes

This repository is intended for learning, experimentation, and practice with modern LLM application development.
