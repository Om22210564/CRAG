from typing import Literal
from typesafe_sdk import Choice, TypeSafeClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
import re

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

    internal_knowledge: str
    web_results: str
    external_knowledge: str

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


# def evaluate_retrieval(
#     state: CRAGState,
# ) -> CRAGState:
#     question = state["question"]
#     docs = state["retrieved_docs"]

#     context = "\n\n".join(
#         f"DOCUMENT {i + 1}:\n{doc.page_content}"
#         for i, doc in enumerate(docs)
#     )

#     prompt = f"""
# You are a retrieval evaluator in a Corrective RAG system.

# Evaluate whether the retrieved documents are useful for answering
# the user's question.

# Question:
# {question}

# Retrieved documents:
# {context}

# Classify the retrieval into exactly ONE category:

# CORRECT:
# The documents contain sufficient and directly relevant information
# to answer the question.

# AMBIGUOUS:
# The documents contain some potentially relevant information, but
# they are incomplete, unclear, or only partially related to the question.

# INCORRECT:
# The documents are irrelevant or do not provide useful information
# for answering the question.

# Return ONLY one word:
# CORRECT
# AMBIGUOUS
# INCORRECT
# """

#     response = llm.invoke([HumanMessage(content=prompt)])

#     grade = response.content.strip().upper()

#     if grade not in {"CORRECT", "AMBIGUOUS", "INCORRECT"}:
#         grade = "AMBIGUOUS"

#     return {
#         "retrieval_grade": grade.lower(),
#     }

def evaluate_retrieval(state: CRAGState) -> CRAGState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    # No retrieved documents means there is no local evidence to assess.
    if not docs:
        return {
            "retrieval_grade": "incorrect",
        }

    context = "\n\n".join(
        f"DOCUMENT {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    with TypeSafeClient() as client:
        response = client.system_one(
            state={
                "question": question,
                "retrieved_documents": context,
            },
            questions={
                "retrieval_grade": Choice(
                    instructions=(
                        "Evaluate how relevant and sufficient the "
                        "retrieved documents are for answering the question. "
                        "Judge the documents against the question, not "
                        "whether the documents are generally informative."
                    ),
                    criteria={
                        "correct": (
                            "The retrieved documents contain directly "
                            "relevant, sufficient evidence to answer "
                            "the question."
                        ),
                        "ambiguous": (
                            "The documents contain partially relevant "
                            "evidence, but important details are missing, "
                            "the evidence is unclear, or relevance is mixed."
                        ),
                        "incorrect": (
                            "The documents are irrelevant or provide "
                            "no useful evidence for answering the question."
                        ),
                    },
                ),
            },
        )

    grade = response.choices["retrieval_grade"].choice.lower()

    if grade not in {"correct", "ambiguous", "incorrect"}:
        raise ValueError(f"Unexpected JEV retrieval grade: {grade}")

    return {
        "retrieval_grade": grade,
    }


def route_after_evaluation(
    state: CRAGState,
) -> Literal["correct", "ambiguous", "incorrect"]:

    return state["retrieval_grade"]


def split_into_knowledge_strips(
    text: str,
    max_sentences: int = 2,
) -> list[str]:
    """Split context into small, meaningful knowledge strips."""
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9“"])',
        text.strip(),
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    strips = []
    for i in range(0, len(sentences), max_sentences):
        strip = " ".join(sentences[i:i + max_sentences])
        strips.append(strip)

    # Remove exact duplicates while preserving order.
    unique_strips = []
    seen = set()

    for strip in strips:
        normalized = " ".join(strip.lower().split())
        if normalized not in seen:
            seen.add(normalized)
            unique_strips.append(strip)

    return unique_strips


def grade_knowledge_strips(
    question: str,
    strips: list[str],
) -> list[str]:
    """Grade each knowledge strip independently using JEV."""
    if not strips:
        return []

    # Use separate evaluation questions for each strip.
    state = {"question": question}
    questions = {}

    for i, strip in enumerate(strips):
        state[f"strip_{i}"] = strip

        questions[f"grade_{i}"] = Choice(
            instructions=(
                f"Evaluate the relevance of strip_{i} in the state "
                "to the user's question. Judge this strip independently. "
                "Do not assume it is relevant merely because other strips "
                "are relevant."
            ),
            criteria={
                "relevant": (
                    "The strip contains factual information that directly "
                    "helps answer the user's question."
                ),
                "irrelevant": (
                    "The strip is unrelated, redundant without useful "
                    "information, or does not help answer the question."
                ),
            },
        )

    with TypeSafeClient() as client:
        response = client.system_one(
            state=state,
            questions=questions,
        )

    accepted = []

    for i, strip in enumerate(strips):
        grade = response.choices[f"grade_{i}"].choice.lower()

        if grade == "relevant":
            accepted.append(strip)
        elif grade != "irrelevant":
            raise ValueError(
                f"Unexpected JEV grade for strip {i}: {grade}"
            )

    return accepted


def refine_context(
    question: str,
    context: str,
    source: str,
) -> str:
    """Decompose, grade, filter, and recompose a knowledge source."""
    if not context.strip():
        return ""

    strips = split_into_knowledge_strips(context)

    relevant_strips = grade_knowledge_strips(
        question,
        strips,
    )

    if not relevant_strips:
        return f"INSUFFICIENT {source.upper()} KNOWLEDGE"

    accepted_context = "\n\n".join(relevant_strips)

    prompt = f"""
You are the {source} knowledge recomposition component of a
Corrective RAG system.

User question:
{question}

Individually graded, relevant knowledge strips:
{accepted_context}

Recompose these strips into concise, coherent knowledge.

Requirements:
- Preserve factual details, numbers, conditions, and qualifications.
- Remove redundancy without losing useful information.
- Do not introduce facts that are not present in the strips.
- Preserve source URLs and attribution when present.
- Do not answer the user's question directly.

Return only the recomposed knowledge.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content.strip()


# def refine_knowledge(state: CRAGState) -> CRAGState:
#     question = state["question"]
#     docs = state.get("retrieved_docs", [])

#     context = "\n\n".join(
#         doc.page_content
#         for doc in docs
#     )

#     prompt = f"""
# You are the knowledge refinement component of a Corrective RAG system.

# Question:
# {question}

# Retrieved knowledge:
# {context}

# Refine this knowledge for downstream answer generation.

# Your task:
# 1. Identify information relevant to the question.
# 2. Remove irrelevant information.
# 3. Remove redundant information.
# 4. Preserve useful factual details.
# 5. Do not invent information.
# 6. Produce a concise, coherent knowledge base.

# Return only the refined knowledge.
# """

#     response = llm.invoke([HumanMessage(content=prompt)])

#     return {
#         "refined_knowledge": response.content,
#     }

# def refine_knowledge(state: CRAGState) -> CRAGState:
#     question = state["question"]
#     docs = state.get("retrieved_docs", [])

#     if not docs:
#         return {"internal_knowledge": ""}

#     context = "\n\n".join(
#         doc.page_content for doc in docs
#     )

#     prompt = f"""
# You are the internal knowledge refinement component of a
# Corrective RAG system.

# User question:
# {question}

# Retrieved documents:
# {context}

# Perform these steps:

# 1. DECOMPOSE
#    Split the documents into small, meaningful knowledge strips.
#    Each strip should generally contain one or two sentences.

# 2. GRADE
#    Independently assess each strip against the user question.
#    Label each strip RELEVANT or IRRELEVANT.
#    Judge its usefulness for answering this specific question.

# 3. FILTER
#    Discard irrelevant, redundant, and unsupported information.
#    Preserve specific facts, values, conditions, and qualifications.

# 4. RECOMPOSE
#    Combine the remaining relevant strips into concise,
#    coherent knowledge for answer generation.

# Do not invent facts or fill gaps with outside knowledge.
# Do not answer the user's question directly.

# Return only the recomposed internal knowledge.
# If no useful information remains, return:
# INSUFFICIENT INTERNAL KNOWLEDGE
# """

#     response = llm.invoke([HumanMessage(content=prompt)])

#     return {
#         "internal_knowledge": response.content.strip(),
#     }

def refine_knowledge(state: CRAGState) -> CRAGState:
    question = state["question"]

    context = "\n\n".join(
        doc.page_content
        for doc in state.get("retrieved_docs", [])
    )

    refined = refine_context(
        question=question,
        context=context,
        source="internal",
    )

    return {
        "internal_knowledge": refined,
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

# def refine_web_knowledge(state: CRAGState) -> CRAGState:
#     question = state["question"]
#     web_results = state.get("web_results", "")

#     prompt = f"""
#         You are the knowledge refinement component of a Corrective RAG system.

#         Question:
#         {question}

#         External web knowledge:
#         {web_results}

#         Extract only information that is useful for answering the question.

#         Requirements:
#         1. Remove irrelevant search-result information.
#         2. Remove redundancy.
#         3. Prefer factual information.
#         4. Do not invent facts.
#         5. Do not answer the question directly.
#         6. Return concise knowledge that can be given to the final generator.

#         Return only the refined knowledge.
#     """

#     response = llm.invoke([HumanMessage(content=prompt)])

#     return {
#         "refined_knowledge": response.content,
#     }
# def refine_web_knowledge(state: CRAGState) -> CRAGState:
#     question = state["question"]
#     web_results = state.get("web_results", "")

#     if not web_results.strip():
#         return {"external_knowledge": ""}

#     prompt = f"""
# You are the external knowledge refinement component of a
# Corrective RAG system.

# User question:
# {question}

# External web search results:
# {web_results}

# Perform these steps:

# 1. DECOMPOSE
#    Split the search results into small, meaningful knowledge strips.

# 2. GRADE
#    Independently assess each strip for relevance to the question.
#    Distinguish useful evidence from unrelated search-result content.

# 3. FILTER
#    Discard irrelevant or redundant strips.
#    Do not treat unsupported claims as established facts.

# 4. RECOMPOSE
#    Combine the useful information into concise, coherent knowledge.

# Preserve source URLs or source identifiers when available.
# Do not invent facts, sources, or citations.
# Do not answer the question directly.

# Return only the recomposed external knowledge.
# If no useful information remains, return:
# INSUFFICIENT EXTERNAL KNOWLEDGE
# """

#     response = llm.invoke([HumanMessage(content=prompt)])

#     return {
#         "external_knowledge": response.content.strip(),
#     }


def refine_web_knowledge(state: CRAGState) -> CRAGState:
    refined = refine_context(
        question=state["question"],
        context=state.get("web_results", ""),
        source="external",
    )

    return {
        "external_knowledge": refined,
    }



# def generate(state: CRAGState) -> CRAGState:
#     question = state["question"]
#     knowledge = state.get("refined_knowledge", "")

#     prompt = f"""
# You are the final answer generator in a Corrective RAG system.

# Question:
# {question}

# Verified knowledge:
# {knowledge}

# Answer the question using only the provided knowledge.

# Rules:
# - Do not invent facts.
# - If the knowledge is insufficient, clearly say so.
# - Be concise and directly answer the question.
# """

#     response = llm.invoke([HumanMessage(content=prompt)])

#     return {
#         "answer": response.content,
#     }
def generate(state: CRAGState) -> CRAGState:
    question = state["question"]
    grade = state["retrieval_grade"]

    internal = state.get("internal_knowledge", "")
    external = state.get("external_knowledge", "")

    if grade == "correct":
        knowledge = f"""
INTERNAL KNOWLEDGE:
{internal}
"""
    elif grade == "ambiguous":
        knowledge = f"""
INTERNAL KNOWLEDGE:
{internal}

EXTERNAL KNOWLEDGE:
{external}
"""
    else:
        knowledge = f"""
EXTERNAL KNOWLEDGE:
{external}
"""

    prompt = f"""
You are the final answer generator in a Corrective RAG system.

User question:
{question}

Permitted knowledge for this answer:
{knowledge}

Instructions:
- Answer using only the permitted knowledge above.
- Do not use irrelevant local documents when retrieval was incorrect.
- Do not invent facts or fill evidence gaps with assumptions.
- If the knowledge is insufficient, clearly state what is missing.
- If internal and external evidence conflict, acknowledge the conflict.
- Keep the answer concise and directly relevant to the question.
- Do not claim that a fact is verified unless the evidence supports it.

Answer:
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "answer": response.content.strip(),
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