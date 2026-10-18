# Corrective Retrieval-Augmented Generation (CRAG)

A Corrective RAG implementation using LangGraph, LangChain, FAISS, Groq, SerpAPI, TypeSafe AI's JEV, and Streamlit. It evaluates retrieved documents and uses query rewriting and web search when local knowledge is insufficient.

## Workflow

```text
                         User Query
                             |
                             v
                         Retrieval
                             |
                             v
                    JEV Retrieval Evaluator
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
         CORRECT          AMBIGUOUS        INCORRECT
            |                |                |
            v                v                v
      Knowledge          Knowledge        Rewrite Query
      Refinement         Refinement           |
            |                |                |
            |                v                |
            |          Rewrite Query          |
            |                |                |
            |                +-------+--------+
            |                        |
            |                        v
            |                    Web Search
            |                        |
            |                        v
            |                Knowledge Refinement
            |                        |
            +-----------+------------+
                        |
                        v
                    Generation
                        |
                        v
                      Answer
```

## Tech Stack

- **Python** — core implementation
- **LangGraph** — workflow orchestration and SQLite checkpointing
- **LangChain** — LLM, embedding, and retrieval integrations
- **TypeSafe AI JEV** — structured evaluation of retrieved context
- **FAISS** — local vector similarity search
- **Hugging Face** — `all-MiniLM-L6-v2` embeddings
- **Groq** — LLM inference for query rewriting, knowledge refinement, and answer generation
- **SerpAPI** — external web search
- **Streamlit** — interactive chat interface
- **SQLite** — persistent conversation checkpoints

## Features

- Evaluates retrieved context as **correct, ambiguous, or incorrect** using JEV.
- Refines relevant local knowledge before answer generation.
- Rewrites queries before external web searches.
- Falls back to web search when local retrieval is ambiguous or incorrect.
- Generates answers from refined knowledge.
- Displays retrieval decisions and a CRAG trace.
- Persists conversations using LangGraph's SQLite checkpointer.
- Supports multiple conversations through the Streamlit interface.

## Setup

1. Install dependencies using your preferred environment manager.

2. Create a `.env` file in the project root using `.env.example` as a template:

   ```bash
   cp .env.example .env
   ```

3. Configure the required API keys and settings in `.env`:

   ```env
   GROQ_API_KEY=your_groq_api_key
   SERPAPI_API_KEY=your_serpapi_api_key
   TYPESAFE_API_KEY=your_typesafe_api_key
   GROQ_MODEL=llama-3.3-70b-versatile
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   TOP_K=4
   ```

4. Add your `.txt` source documents to `data/dataset/`.

5. Build the FAISS index:

   ```bash
   python3 scripts/ingest.py
   ```

6. Launch the application:

   ```bash
   streamlit run app.py
   ```

## Project Structure

```text
crag/
├── app.py
├── graph.py
├── retrieval.py
├── web_search.py
├── config.py
├── scripts/
│   └── ingest.py
├── data/
│   ├── dataset/
│   ├── faiss/
│   └── checkpoints.db
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Notes

- Re-run ingestion after adding or modifying source documents.
- FAISS indexes and SQLite checkpoints are generated locally.
- Keep API keys in `.env` and never commit secrets to version control.
- The embedding model runs locally after its initial download.
