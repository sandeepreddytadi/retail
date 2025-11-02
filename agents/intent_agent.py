# agents/intent_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
import re

# Use a friendly temperature for greetings so responses are natural.
_llm_greeting = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
# Use a more deterministic model for classification (low temperature).
_llm_classifier = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def _is_general_greeting(text: str) -> bool:
    text = text.strip().lower()
    general_keywords = [
        "hi", "hello", "hey", "good morning", "good evening",
        "how are you", "what's up", "what’s up", "thanks", "thank you",
        "your name", "who are you", "bye", "goodbye"
    ]
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in general_keywords)

def classify_intent(query: str) -> dict:
    """
    Returns one of:
      {"type": "greeting", "response": "<LLM greeting text>"}
      {"type": "intent", "response": "<faq|product|policy|store>"}
      {"type": "out_of_scope", "response": "<reason>"}  # non-retail queries
    """
    q = (query or "").strip()
    if not q:
        return {"type": "out_of_scope", "response": "empty"}

    # 1) If query looks like greeting/small talk -> let Gemini respond conversationally
    if _is_general_greeting(q):
        prompt = f"""
You are a friendly retail AI assistant. The user said:
\"{q}\"
Respond naturally, briefly, and warmly. If the user asks something unrelated to retail, keep your reply short and friendly.
"""
        resp = _llm_greeting.invoke(prompt)
        return {"type": "greeting", "response": resp.content.strip()}

    # 2) Otherwise ask the LLM to classify intent into retail categories
    prompt = f"""
You are an intent classifier for a retail assistant.
Classify the query into exactly one of these labels: [faq, product, policy, store, out_of_scope]
- faq: general frequently asked questions about shopping / support
- product: product-specific questions (price, specs, availability)
- policy: return/refund/exchange/guarantee
- store: store location, hours, contact
- out_of_scope: things not related to retail or shop (politics, medical advice, coding help)
Respond with ONLY the single label (no explanation).
Query: {q}
"""
    resp = _llm_classifier.invoke(prompt)
    label = resp.content.strip().lower()

    # Normalize label
    if label not in ["faq", "product", "policy", "store", "out_of_scope"]:
        # fallback heuristics
        ql = q.lower()
        if any(w in ql for w in ["return", "refund", "exchange", "warranty", "guarantee"]):
            label = "policy"
        elif any(w in ql for w in ["price", "cost", "battery", "camera", "model", "specs", "specifications", "availability", "stock"]):
            label = "product"
        elif any(w in ql for w in ["store", "location", "hours", "open", "closed", "address", "contact", "phone"]):
            label = "store"
        elif any(w in ql for w in ["joke", "politics", "medical", "program", "code", "how to code", "movie review"]):
            label = "out_of_scope"
        else:
            # default to faq if unsure
            label = "faq"

    if label == "out_of_scope":
        return {"type": "out_of_scope", "response": "non-retail"}

    return {"type": "intent", "response": label}
