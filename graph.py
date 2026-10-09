from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import (
    CHECKPOINT_DB,
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K,
)
from retrieval import get_retriever
from web_search import search_web
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

class CRAGState(TypedDict, total=False):
    question: str
    rewritten_query: str
    retrieved_docs: list[Document]
    retrieval_grade: Literal["correct", "ambiguous", "incorrect"]
    refined_knowledge: str
    web_results: str
    answer: str


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0,
)

retriever = get_retriever(TOP_K)


def retrieve(state: CRAGState) -> CRAGState:
    docs = retriever.invoke(state["question"])

    return {
        "retrieved_docs": docs,
    }


def evaluate_retrieval(
    state: CRAGState,
) -> CRAGState:
    question = state["question"]
    docs = state["retrieved_docs"]

    context = "\n\n".join(
        f"DOCUMENT {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    prompt = f"""
You are a retrieval evaluator in a Corrective RAG system.

Evaluate whether the retrieved documents are useful for answering
the user's question.

Question:
{question}

Retrieved documents:
{context}

Classify the retrieval into exactly ONE category:

CORRECT:
The documents contain sufficient and directly relevant information
to answer the question.

AMBIGUOUS:
The documents contain some potentially relevant information, but
they are incomplete, unclear, or only partially related to the question.

INCORRECT:
The documents are irrelevant or do not provide useful information
for answering the question.

Return ONLY one word:
CORRECT
AMBIGUOUS
INCORRECT
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    grade = response.content.strip().upper()

    if grade not in {"CORRECT", "AMBIGUOUS", "INCORRECT"}:
        grade = "AMBIGUOUS"

    return {
        "retrieval_grade": grade.lower(),
    }


def route_after_evaluation(
    state: CRAGState,
) -> Literal["correct", "ambiguous", "incorrect"]:

    return state["retrieval_grade"]


def refine_knowledge(state: CRAGState) -> CRAGState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are the knowledge refinement component of a Corrective RAG system.

Question:
{question}

Retrieved knowledge:
{context}

Refine this knowledge for downstream answer generation.

Your task:
1. Identify information relevant to the question.
2. Remove irrelevant information.
3. Remove redundant information.
4. Preserve useful factual details.
5. Do not invent information.
6. Produce a concise, coherent knowledge base.

Return only the refined knowledge.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "refined_knowledge": response.content,
    }


def web_search_node(state: CRAGState) -> CRAGState:
    query = state.get("rewritten_query") or state["question"]

    results = search_web(query)

    return {
        "web_results": results,
    }

def rewrite_query(state: CRAGState) -> CRAGState:
    question = state["question"]

    prompt = f"""
        You are a query rewriting component in a Corrective RAG system.

        Rewrite the user's question into a concise, search-engine-friendly
        query that will retrieve reliable and relevant information.

        Requirements:
        - Preserve the original intent.
        - Include important keywords and entities.
        - Resolve ambiguity when possible without inventing facts.
        - Do not answer the question.
        - Return only the rewritten search query.

        Original question:
        {question}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "rewritten_query": response.content.strip(),
    }

def refine_web_knowledge(state: CRAGState) -> CRAGState:
    question = state["question"]
    web_results = state.get("web_results", "")

    prompt = f"""
        You are the knowledge refinement component of a Corrective RAG system.

        Question:
        {question}

        External web knowledge:
        {web_results}

        Extract only information that is useful for answering the question.

        Requirements:
        1. Remove irrelevant search-result information.
        2. Remove redundancy.
        3. Prefer factual information.
        4. Do not invent facts.
        5. Do not answer the question directly.
        6. Return concise knowledge that can be given to the final generator.

        Return only the refined knowledge.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "refined_knowledge": response.content,
    }


def generate(state: CRAGState) -> CRAGState:
    question = state["question"]
    knowledge = state.get("refined_knowledge", "")

    prompt = f"""
You are the final answer generator in a Corrective RAG system.

Question:
{question}

Verified knowledge:
{knowledge}

Answer the question using only the provided knowledge.

Rules:
- Do not invent facts.
- If the knowledge is insufficient, clearly say so.
- Be concise and directly answer the question.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "answer": response.content,
    }


def build_graph():
    workflow = StateGraph(CRAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("evaluate_retrieval", evaluate_retrieval)
    workflow.add_node("refine_knowledge", refine_knowledge)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("refine_web_knowledge", refine_web_knowledge)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate_retrieval")

    workflow.add_conditional_edges(
        "evaluate_retrieval",
        route_after_evaluation,
        {
            "correct": "refine_knowledge",
            "ambiguous": "refine_knowledge",
            "incorrect": "rewrite_query",
        },
    )

    workflow.add_conditional_edges(
        "refine_knowledge",
        lambda state: state["retrieval_grade"],
        {
            "correct": "generate",
            "ambiguous": "rewrite_query",
        },
    )
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "refine_web_knowledge")
    workflow.add_edge("refine_web_knowledge", "generate")
    workflow.add_edge("generate", END)

    connection = sqlite3.connect(
        CHECKPOINT_DB,
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)

    return workflow.compile(
        checkpointer=checkpointer,
    )


graph = build_graph()