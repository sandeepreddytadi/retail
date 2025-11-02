# app.py
import streamlit as st
from agents.intent_agent import classify_intent
from agents.retrieval_agent import retrieve
from agents.response_agent import generate_answer
from agents.evaluation_agent import evaluate_relevance

st.set_page_config(page_title="AI Retail Assistant (Gemini)", layout="centered")
st.title("🛍️ AI Retail Assistant — Google Gemini LLM")
st.markdown("This app uses **Gemini** for classification and conversational responses. Retail questions use RAG.")

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
        result = classify_intent(query)

    # Greeting: LLM responded directly
    if result["type"] == "greeting":
        intent = "general"
        contexts = []
        answer = result["response"]
        eval_result = {"score": 100, "explain": "Handled as greeting by Gemini."}

    # Out of scope (non-retail)
    elif result["type"] == "out_of_scope":
        intent = "out_of_scope"
        contexts = []
        answer = "I'm a retail assistant — I answer retail-related questions. Please ask about products, returns, or stores."
        eval_result = {"score": 100, "explain": "Out-of-scope query; user redirected to retail topics."}

    # Retail intent -> RAG pipeline
    else:
        intent = result["response"]  # faq/product/policy/store
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

# Display history
for item in reversed(st.session_state.history):
    st.markdown(f"**Q:** {item['query']}")
    st.markdown(f"**Intent:** `{item['intent']}`")
    st.markdown(f"**A:** {item['answer']}")
    st.markdown(f"**Relevance:** {item['eval'].get('score','-')} — {item['eval'].get('explain','-')}")
    if show_ctx and item["contexts"]:
        st.markdown("**Context snippets:**")
        for c in item["contexts"]:
            st.write(c[:800])
    st.markdown("---")
