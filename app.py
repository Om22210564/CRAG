
import sqlite3
import uuid

import streamlit as st

from config import CHECKPOINT_DB
from graph import graph


st.set_page_config(
    page_title="Corrective RAG",
    page_icon="🔎",
    layout="wide",
)


# ---------- Thread and history helpers ----------

def generate_thread_id() -> str:
    return str(uuid.uuid4())



def retrieve_all_threads() -> list[str]:
    """Load persisted thread IDs from the LangGraph SQLite database."""
    if not CHECKPOINT_DB.exists():
        return []

    with sqlite3.connect(CHECKPOINT_DB) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE checkpoint_ns = ''
            """
        ).fetchall()

    return [row[0] for row in rows]



def load_conversation(thread_id: str) -> list[dict]:
    """Reconstruct completed question-answer pairs from graph checkpoints."""
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    snapshots = list(graph.get_state_history(config))
    snapshots.reverse()

    messages = []

    for snapshot in snapshots:
        # Only include completed graph executions.
        if snapshot.next:
            continue

        values = snapshot.values

        question = values.get("question")
        answer = values.get("answer")

        if not question or not answer:
            continue

        messages.append(
            {"role": "user", "content": question}
        )
        messages.append(
            {"role": "assistant", "content": answer}
        )

    return messages


def get_thread_title(thread_id: str) -> str:
    """Use the first question as the conversation's sidebar label."""
    messages = load_conversation(thread_id)

    if messages:
        title = messages[0]["content"].replace("\n", " ").strip()
        return title[:35] + ("…" if len(title) > 35 else "")

    return f"Chat {thread_id[:8]}"


# ---------- Session initialization ----------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()


def start_new_chat():
    st.session_state.thread_id = generate_thread_id()


def open_chat(thread_id: str):
    st.session_state.thread_id = thread_id


# ---------- Sidebar ----------

with st.sidebar:
    st.title("🔎 Corrective RAG")
    st.caption("Retrieval with corrective web search")

    if st.button("＋ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()
    st.subheader("Your conversations")

    thread_ids = retrieve_all_threads()

    for thread_id in thread_ids:
        label = get_thread_title(thread_id)

        if st.button(
            label,
            key=f"thread_{thread_id}",
            use_container_width=True,
            type=(
                "primary"
                if thread_id == st.session_state.thread_id
                else "secondary"
            ),
        ):
            open_chat(thread_id)
            st.rerun()

    st.divider()
    st.caption("LangGraph · FAISS · Groq · SerpAPI")


# ---------- Main chat ----------

st.header("Chat")
st.caption("Ask a question about your documents or the wider web.")

current_thread = st.session_state.thread_id
history = load_conversation(current_thread)

# Restore the selected conversation whenever the UI reruns.
for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            pass


question = st.chat_input("Ask a question...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    config = {
        "configurable": {
            "thread_id": current_thread,
        }
    }

    with st.chat_message("assistant"):
        with st.spinner("Running CRAG..."):
            try:
                result = graph.invoke(
                    {"question": question},
                    config=config,
                )
            except Exception as exc:
                st.error(f"CRAG execution failed: {exc}")
                st.stop()

        answer = result.get("answer", "No answer generated.")
        grade = result.get("retrieval_grade", "unknown")

        st.markdown(answer)

        with st.expander("CRAG Trace"):
            st.write(f"**Retrieval decision:** {grade.upper()}")

            if result.get("rewritten_query"):
                st.write("**Rewritten search query**")
                st.code(result["rewritten_query"])

            if result.get("retrieved_docs"):
                st.write("**Retrieved documents**")

                for index, doc in enumerate(
                    result["retrieved_docs"],
                    start=1,
                ):
                    source = doc.metadata.get("source", "Unknown source")

                    st.markdown(f"**Document {index}:** `{source}`")
                    st.text(doc.page_content[:500])

            if result.get("web_results"):
                st.write("**Web search results**")
                st.text(result["web_results"][:2000])

            if result.get("refined_knowledge"):
                st.write("**Refined knowledge**")
                st.text(result["refined_knowledge"])

        if grade == "correct":
            st.caption("Retrieval: Correct")
        elif grade == "ambiguous":
            st.caption("Retrieval: Ambiguous → Web Search")
        elif grade == "incorrect":
            st.caption("Retrieval: Incorrect → Web Search")
