# Corrective Retrieval-Augmented Generation (CRAG)

A Corrective Retrieval-Augmented Generation (CRAG) implementation built with LangGraph, LangChain, FAISS, Groq, SerpAPI, TypeSafe AI's JEV, and Streamlit.

The system evaluates locally retrieved documents, refines knowledge at the individual-strip level, and uses query rewriting and external web search when local retrieval is ambiguous or incorrect. It then generates answers using knowledge permitted by the retrieval decision.

## Workflow

```text
                         User Query
                             |
                             v
                    Local FAISS Retrieval
                             |
                             v
                    JEV Retrieval Evaluator
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
           CORRECT       AMBIGUOUS       INCORRECT
              |              |              |
              v              v              |
       Refine Internal  Refine Internal     |
         Knowledge          Knowledge       |
              |              |              |
              |              +------+-------+
              |                     |
              |                     v
              |               Rewrite Query
              |                     |
              |                     v
              |                 Web Search
              |                     |
              |                     v
              |              Refine External
              |                Knowledge
              |                     |
              +----------+----------+
                         |
                         v
                    Answer Generation
                         |
                         v
                       Answer
```

### Retrieval and generation strategy

| Retrieval grade | Processing                                                                    | Knowledge used for generation           |
| --------------- | ----------------------------------------------------------------------------- | --------------------------------------- |
| `correct`       | Refine local knowledge                                                        | Refined internal knowledge              |
| `ambiguous`     | Refine local knowledge, rewrite query, search the web, and refine web results | Refined internal and external knowledge |
| `incorrect`     | Rewrite query, search the web, and refine web results                         | Refined external knowledge only         |

## Knowledge Refinement

The refinement process operates on retrieved knowledge rather than performing retrieval itself.

1. **Decomposition:** Split the retrieved context into small knowledge strips, generally containing one or two sentences.
2. **Relevance grading:** Use TypeSafe AI's JEV to evaluate each strip independently against the original question.
3. **Filtering:** Retain relevant strips and reject irrelevant or unhelpful information.
4. **Recomposition:** Use the LLM to combine the accepted strips into concise, coherent knowledge while preserving facts and available source attribution.

The same refinement process is used for internal documents and external web-search results.

## Tech Stack

- **Python** — core implementation.
- **LangGraph** — workflow orchestration, conditional routing, and conversation checkpointing.
- **LangChain** — LLM, embedding, and retrieval integrations.
- **TypeSafe AI JEV** — structured evaluation of retrieved documents and individual knowledge strips.
- **FAISS** — local vector similarity search.
- **Hugging Face** — local `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- **Groq** — LLM inference for query rewriting, knowledge refinement, and answer generation.
- **SerpAPI** — external web search.
- **Streamlit** — interactive chat interface.
- **SQLite** — persistent conversation checkpoints.

## Features

- Evaluates retrieved documents as `correct`, `ambiguous`, or `incorrect` using JEV.
- Refines internal and external knowledge using independently graded knowledge strips.
- Filters irrelevant information before answer generation.
- Rewrites queries before external web searches.
- Uses internal knowledge for correct retrieval, combines internal and external knowledge for ambiguous retrieval, and uses external knowledge only for incorrect retrieval.
- Displays retrieval decisions and a CRAG execution trace.
- Persists conversations using LangGraph's SQLite checkpointer.
- Supports multiple conversations through the Streamlit interface.
- Uses a local FAISS index and locally executed embedding model.

## Setup

### 1. Install dependencies

This project uses `uv` for Python dependency management.

```bash
uv sync
```

Activate the project environment if you prefer running commands directly:

```bash
source .venv/bin/activate
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Add your API keys and settings:

```env
GROQ_API_KEY=your_groq_api_key
SERPAPI_API_KEY=your_serpapi_api_key
TYPESAFE_API_KEY=your_typesafe_api_key

GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4
```

Use the variable names expected by your `config.py` and `.env.example`. Never commit real API keys.

### 3. Add source documents

Place your `.txt` source documents in:

```text
data/dataset/
```

The current dataset includes job descriptions from:

- Konecranes
- PTC
- Atlas Copco

### 4. Build the FAISS index

Run ingestion after adding or modifying source documents:

```bash
python3 scripts/ingest.py
```

This creates or refreshes the local vector index in `data/faiss/`.

### 5. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## Project Structure

```text
crag/
├── app.py                  # Streamlit chat interface
├── graph.py                # CRAG workflow and knowledge refinement
├── retrieval.py             # FAISS retrieval and embeddings
├── web_search.py            # SerpAPI web search
├── config.py                # Configuration and environment variables
├── scripts/
│   └── ingest.py            # Document ingestion and index creation
├── data/
│   ├── dataset/             # Source text documents
│   ├── faiss/               # Local vector index
│   └── checkpoints.db       # SQLite conversation checkpoints
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Validation

The following routing scenarios have been tested successfully:

- **Correct retrieval:** Local knowledge is refined and used to generate the answer.
- **Ambiguous retrieval:** Internal knowledge is refined, a rewritten query is searched externally, and both knowledge sources are available for generation.
- **Incorrect retrieval:** The graph uses web search and external knowledge for generation without relying on irrelevant local documents.

## Notes

- Re-run ingestion after adding or modifying source documents.
- The FAISS index and SQLite checkpoint database are generated locally.
- Keep API keys in `.env` and exclude that file from version control.
- The embedding model runs locally after its initial download.
- External search and LLM operations require the relevant API access and may be subject to provider limits.

## Future Scope

- **Refinement fallback:** If all internal knowledge strips are rejected, route to web search rather than treating `INSUFFICIENT INTERNAL KNOWLEDGE` as useful evidence.
- **Source attribution:** Preserve web URLs alongside accepted knowledge strips and support source citations in generated answers.
- **Refinement observability:** Track the number of strips extracted, accepted, and rejected, and expose these metrics in the CRAG trace.
- **Evaluation:** Build a test set to measure retrieval grading, routing accuracy, knowledge relevance, and answer quality across different questions and documents.
