import uuid

import streamlit as st

from graph import graph


st.set_page_config(
    page_title="Corrective RAG",
    page_icon="🔎",
    layout="centered",
)


def initialize_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


initialize_session()


st.title("Corrective RAG")



with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.thread_id)

    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


question = st.chat_input("Ask a question...")


if question:
    st.chat_message("user").write(question)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
        }
    }

    with st.chat_message("assistant"):
        with st.spinner("Running CRAG..."):
            result = graph.invoke(
                {"question": question},
                config=config,
            )

        grade = result.get("retrieval_grade", "unknown")
        answer = result.get("answer", "No answer generated.")

        st.write(answer)

        st.divider()

        if grade == "correct":
            st.caption("Retrieval: Correct")
        elif grade == "ambiguous":
            st.caption("Retrieval: Ambiguous → Web Search")
        elif grade == "incorrect":
            st.caption("Retrieval: Incorrect → Web Search")