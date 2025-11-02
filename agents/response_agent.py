from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def generate_answer(query: str, contexts: list) -> str:
    context = "\n\n".join(contexts) if contexts else ""
    prompt = f"""
You are a helpful retail AI assistant. Use the provided context to answer concisely.

Context:
{context}

User question:
{query}

If the context doesn't contain an answer, respond:
"I couldn’t find that in our data — please check with the nearest store."
"""
    resp = llm.invoke(prompt)
    return resp.content.strip()
