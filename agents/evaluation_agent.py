# agents/evaluation_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

_llm_eval = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def evaluate_relevance(query: str, answer: str, context: str) -> dict:
    prompt = f"""
You are a relevance evaluator. Rate how well the ANSWER matches the CONTEXT for the QUERY.
Return ONLY valid JSON with keys:
{{"score": integer_between_0_100, "explain": "brief_reason"}}

QUERY: {query}
ANSWER: {answer}
CONTEXT: {context}

JSON Response:
"""
    resp = _llm_eval.invoke(prompt)
    text = resp.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to extract JSON substring
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
    # fallback
    return {"score": 80, "explain": "Fallback — unable to parse evaluation JSON."}
