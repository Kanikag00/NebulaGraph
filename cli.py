import streamlit as st
from api import NebulaGraphQA

st.set_page_config(page_title="Graph QA System", layout="wide")

st.title("🧠 Nebula Graph Q&A")

@st.cache_resource
def load_qa():
    return NebulaGraphQA()

qa_system = load_qa()

question = st.text_input("Ask your question:")

if st.button("Submit") and question:
    with st.spinner("Processing..."):
        try:
            result = qa_system.ask(question)

            st.subheader("📌 Generated NGQL Query")
            st.code(result["query"], language="sql")

            st.subheader("📊 Raw Database Result")
            st.json(result["raw_result"])

            st.subheader("✅ Final Answer")
            st.success(result["answer"])

        except Exception as e:
            st.error(f"Error: {str(e)}")