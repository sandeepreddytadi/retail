import streamlit as st
from agents.intent_agent import classify_intent
from agents.retrieval_agent import retrieve
from agents.response_agent import generate_answer
from agents.evaluation_agent import evaluate_relevance
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="AI Retail Assistant (Gemini)", layout="centered")
st.title("🛍️ AI Retail Assistant — Google Gemini LLM")

st.markdown("This app uses **Gemini 1.5 Flash** for intent, response, and evaluation.")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask about products, return policies, or store info:")

col1, col2 = st.columns(2)
with col1:
    k = st.slider("Retrieve top-k chunks", 1, 5, 3)
with col2:
    show_ctx = st.checkbox("Show retrieved context", value=True)

if st.button("Ask") and query.strip():
    with st.spinner("Classifying intent..."):
        intent = classify_intent(query)
    with st.spinner("Retrieving relevant data..."):
        contexts = retrieve(intent, query, k=k)
    with st.spinner("Generating answer..."):
        answer = generate_answer(query, contexts)
    with st.spinner("Evaluating relevance..."):
        eval_result = evaluate_relevance(query, answer, " ".join(contexts))

    st.session_state.history.append({
        "query": query, "intent": intent,
        "answer": answer, "contexts": contexts,
        "eval": eval_result
    })

for item in reversed(st.session_state.history):
    st.markdown(f"**Q:** {item['query']}")
    st.markdown(f"**Intent:** `{item['intent']}`")
    st.markdown(f"**A:** {item['answer']}")
    st.markdown(f"**Relevance:** {item['eval']['score']} — {item['eval']['explain']}")
    if show_ctx:
        st.markdown("**Context snippets:**")
        for c in item["contexts"]:
            st.write(c[:800])
    st.markdown("---")
