# agents/response_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def generate_answer(query: str, contexts: list) -> str:
    context = "\n\n".join(contexts) if contexts else ""
    prompt = f"""
You are a helpful retail AI assistant. Use the provided context to answer concisely.

Context:
{context}

User question:
{query}

Instructions:
- If the context contains an answer, answer concisely and cite (briefly) where you got the info if possible.
- If the context does not contain an answer, respond exactly:
"I couldn’t find that in our data — please check with the nearest store."

Keep the answer short (1-4 sentences).
"""
    resp = _llm.invoke(prompt)
    return resp.content.strip()
