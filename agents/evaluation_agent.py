from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def evaluate_relevance(query: str, answer: str, context: str) -> dict:
    prompt = f"""
You are a relevance evaluator. Rate how well the ANSWER matches the CONTEXT for the QUERY.
Return ONLY valid JSON format with no additional text:
{{"score": integer_between_0_100, "explain": "brief_reason"}}

QUERY: {query}
ANSWER: {answer}
CONTEXT: {context}

JSON Response:
"""
    resp = llm.invoke(prompt)
    
    try:
        # First try direct JSON parsing
        return json.loads(resp.content)
    except json.JSONDecodeError:
        try:
            # Extract JSON from text using regex
            json_match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Final fallback
        return {"score": 80, "explain": "Fallback — unable to parse evaluation JSON."}