# Corrective Retrieval-Augmented Generation (CRAG)

A CRAG implementation using LangGraph, LangChain, FAISS, Groq, SerpAPI, and Streamlit. It evaluates retrieved documents and uses query rewriting and web search when local knowledge is insufficient.

## Workflow

```text
                         User Query
                              |
                              v
                          Retrieval
                              |
                              v
                     Retrieval Evaluator
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

- **Python**
- **LangGraph** – workflow orchestration and SQLite checkpointing
- **LangChain** – LLM and retrieval integrations
- **FAISS** – local vector search
- **Hugging Face** – `all-MiniLM-L6-v2` embeddings
- **Groq** – LLM inference
- **SerpAPI** – web search
- **Streamlit** – chat interface

## Setup

1. Install dependencies using your preferred environment manager.

2. Create a `.env` file in the project root:

```bash
cat .env.example .env
```

3. Add your `.txt` documents to `data/documents/`.

4. Build the FAISS index:

```bash
python3 scripts/ingest.py
```

5. Launch the application:

```bash
streamlit run app.py
```

## Features

- Evaluates local retrieval as **correct, ambiguous, or incorrect**.
- Refines retrieved knowledge before generation.
- Rewrites queries before external web searches.
- Uses web search when local retrieval is ambiguous or incorrect.
- Displays retrieval decisions and a CRAG trace.
- Persists conversations using LangGraph's SQLite checkpointer.

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
│   ├── documents/
│   └── faiss/
├── pyproject.toml
└── README.md
```

## Notes

- Re-run ingestion after adding or changing source documents.
- FAISS indexes and SQLite checkpoints are generated locally.
