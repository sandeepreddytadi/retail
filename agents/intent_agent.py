from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def classify_intent(query: str) -> str:
    prompt = f"""
You are a retail assistant intent classifier.
Classify the query into one of these intents:
[faq, product, policy, store]
Respond with **only** the label, nothing else.

Query: {query}
"""
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    if intent not in ["faq", "product", "policy", "store"]:
        # fallback
        if any(w in query.lower() for w in ["return", "refund", "exchange"]): return "policy"
        if any(w in query.lower() for w in ["price", "battery", "camera", "model"]): return "product"
        if any(w in query.lower() for w in ["store", "location", "hours", "contact"]): return "store"
        return "faq"
    return intent
