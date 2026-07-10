# LangChain Practice

This folder is the LangChain practice project. It contains notebooks, example scripts, and helper modules used for experimenting with prompts, chains, document loaders, embeddings, vector stores, retrievers, and structured outputs.

Quick start

1. Create and activate a virtual environment
   - Windows (PowerShell): `.\.venv\Scripts\Activate.ps1` after creating the venv
   - Windows (cmd): `.\.venv\Scripts\activate.bat`
   - Unix/macOS: `source .venv/bin/activate`

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the simple example

```bash
python simpleLlmCall.py
```

What is included

- Chains and orchestration: `CX_Chains/` (sequential, parallel, conditional)
- Prompt engineering and chat helpers: `CX_Prompts/` (templates, chat bot)
- Document loaders: `CX_DocLoaders/` (CSV, text, PDF, web)
- Models and embeddings: `CX_Models/` (OpenAI, Google Gemini, Groq, Hugging Face)
- Vector store examples: `CX_VectorStore/` (Chroma examples and a local chroma DB)
- Retrievers and runnable workflows: `CX_Retrievers/`, `CX_Runnable/`
- Structured output parsers: `CX_StructuredOutput/` (Pydantic/typed outputs)
- Text splitting utilities: `CX_TextSplitting/`
- Tools & notebooks: `CX_Tools/`, various `.ipynb` notebooks
- Example scripts: `simpleLlmCall.py`, `test.ipynb`

Notes

- Many examples require API keys (OpenAI, Google Gen AI, Groq, Hugging Face). Add them to a `.env` file in this folder or in the repository root before running.
- The `CX_VectorStore/my_chroma_db/` directory contains local Chroma DB files used by notebook examples — these are binary and can be large.
