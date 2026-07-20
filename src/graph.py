import re
import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI

import config
from config import OPENROUTER_API_KEY
from retrieval import vector_search

# OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

REFUSAL_RESPONSE = "I am sorry, but I do not have enough information to answer your question."


class AgentState(TypedDict):
    question: str
    retrieved_docs: list
    rewritten_question: str
    answer: str
    retries: int
    refused: bool
    cited_ids: list[str]


# Step 1: Retrieve
def retrieve(state: AgentState):
    print("📄 Retrieving documents...")

    # In baseline or if no rewritten query yet, use original question.
    query = state.get("rewritten_question") or state["question"]

    # Call vector search
    docs = vector_search(query, k=5)

    state["retrieved_docs"] = docs
    print(f"Retrieved {len(docs)} documents for query: '{query}'")

    return state


# Step 2: Grade
def grade(state: AgentState):
    # If RAG_MODE is baseline, skip grading
    if config.RAG_MODE == "baseline":
        print("✅ Skipping grading (Baseline mode)")
        state["refused"] = False
        return state

    print("🔍 Grading retrieved documents in batch...")
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        print("No documents retrieved, setting to rewrite/refuse.")
        state["refused"] = True
        return state

    # Format all docs for batch grading
    docs_text = ""
    for idx, doc in enumerate(docs):
        docs_text += f"Document [{idx}]:\n{doc.get('text', '')}\n---\n"

    prompt = f"""You are a relevance grader. Grade if each of the following retrieved documents is relevant to the user question.
User Question: {question}

{docs_text}
Respond with a JSON list of indices of the relevant documents, for example: [0, 2].
If no documents are relevant, return an empty list: [].
Respond ONLY with the JSON list (no text, no explanation, no formatting other than valid JSON)."""

    relevant_docs = []
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        print(f"Grader raw output: {content}")
        
        # Clean response in case it returned code block formatting like ```json ... ```
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            content = match.group(0)
            
        relevant_indices = json.loads(content)
        for idx in relevant_indices:
            try:
                i = int(idx)
                if 0 <= i < len(docs):
                    relevant_docs.append(docs[i])
            except (ValueError, TypeError):
                pass
    except Exception as e:
        print(f"Error in batch grading: {e}. Falling back to keeping all documents.")
        # Fallback to keeping all documents
        relevant_docs = docs

    print(f"Grading complete: {len(relevant_docs)} / {len(docs)} docs found relevant.")

    # Store only relevant documents
    state["retrieved_docs"] = relevant_docs

    if not relevant_docs:
        state["refused"] = True
    else:
        state["refused"] = False

    return state


# Step 3: Rewrite
def rewrite(state: AgentState):
    print("✍️ Rewriting query...")
    question = state["question"]

    prompt = f"""You are a query rewriter. Rewrite the following user question to make it better optimized for a vector search engine that searches Airbnb listings.
Original Question: {question}
Output only the rewritten query string, with no extra explanation or introduction."""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        rewritten = response.choices[0].message.content.strip()
        state["rewritten_question"] = rewritten
        print(f"Rewritten query: '{rewritten}'")
    except Exception as e:
        print(f"Error rewriting query: {e}")
        state["rewritten_question"] = question

    # Increment retries
    state["retries"] = state.get("retries", 0) + 1
    return state


# Step 4: Generate
def generate(state: AgentState):
    print("🤖 Generating answer...")

    # Check if refused during grading
    if config.RAG_MODE == "improved" and state.get("refused", False):
        state["answer"] = REFUSAL_RESPONSE
        state["cited_ids"] = []
        return state

    docs = state.get("retrieved_docs", [])
    if not docs:
        state["answer"] = REFUSAL_RESPONSE
        state["cited_ids"] = []
        state["refused"] = True
        return state

    # Construct context
    context = ""
    for doc in docs:
        context += f"""
Listing ID: {doc.get("listing_id")}
Name: {doc.get("name", "")}
Text Chunk: {doc.get("text", "")}
---
"""

    prompt = f"""You are an Airbnb assistant. Answer the user's question using ONLY the provided context.
If the required information is not available in the context, you must refuse to answer by returning exactly: "{REFUSAL_RESPONSE}"

You must cite the source Listing IDs of any listings you use to answer. Format your citation exactly as [Listing ID: <id>].
For example, if you use listing 10006546, include [Listing ID: 10006546] in your text.

User Question: {state["question"]}

Context:
{context}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful, precise Airbnb assistant that strictly answers only based on the provided context."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        state["answer"] = answer

        # Extract citations
        cited_ids = []
        for doc in docs:
            lid = str(doc.get("listing_id"))
            if lid in answer:
                cited_ids.append(lid)

        parsed_ids = re.findall(r"\[Listing ID:\s*(\d+)\]", answer)
        for pid in parsed_ids:
            if pid not in cited_ids and any(str(d.get("listing_id")) == pid for d in docs):
                cited_ids.append(pid)

        state["cited_ids"] = cited_ids

        # If answer is refusal response, set refused to True
        if REFUSAL_RESPONSE.lower() in answer.lower():
            state["refused"] = True
            state["cited_ids"] = []

    except Exception as e:
        print(f"Error generating answer: {e}")
        state["answer"] = REFUSAL_RESPONSE
        state["cited_ids"] = []
        state["refused"] = True

    return state


def decide_to_generate(state: AgentState):
    if config.RAG_MODE == "baseline":
        return "generate"

    if not state.get("retrieved_docs", []):
        if state.get("retries", 0) < 1:
            print("No relevant documents found. Routing to rewrite_query...")
            return "rewrite"
        else:
            print("No relevant documents found after max retries. Routing to generate (which will refuse)...")
            return "generate"
    else:
        print("Relevant documents found. Routing to generate...")
        return "generate"


# Build Graph
builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")

builder.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "question": "Find me an apartment in Portugal",
            "retrieved_docs": [],
            "rewritten_question": "",
            "answer": "",
            "retries": 0,
            "refused": False,
            "cited_ids": [],
        }
    )

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(result["answer"])
    print("Citations:", result.get("cited_ids"))