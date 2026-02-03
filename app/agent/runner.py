from __future__ import annotations

import json
from typing import Any, Dict, List

from app.llm.client import chat
from app.rag.vectorstore import FaissStore
from app.agent.tools import tool_search_docs, tool_summarize, tool_extract_action_items


TOOLS_SPEC = """
You can call tools by returning JSON in this format ONLY:
{"tool":"search_docs","args":{"query":"...","top_k":5}}
{"tool":"summarize","args":{"text":"..."}}
{"tool":"extract_action_items","args":{"text":"..."}}

If you have enough info, return:
{"tool":"final","args":{"answer":"..."}}
"""


async def run_agent(store: FaissStore, user_question: str, max_steps: int = 3) -> Dict[str, Any]:
    memory: List[Dict[str, Any]] = []
    question = user_question

    for step in range(max_steps):
        # Provide memory as context
        messages = [
            {"role": "system", "content": "You are an agent. Decide the next tool call. " + TOOLS_SPEC},
            {"role": "user", "content": f"USER_QUESTION: {question}\n\nMEMORY_JSON:\n{json.dumps(memory)[:6000]}"},
        ]

        plan = (await chat(messages)).strip()

        # Try to parse tool call JSON
        start = plan.find("{")
        end = plan.rfind("}")
        if start != -1 and end != -1 and end > start:
            plan = plan[start : end + 1]

        try:
            call = json.loads(plan)
        except Exception:
            # If model didn't follow tool JSON, fallback: do search then final
            evidence = await tool_search_docs(store, query=question, top_k=5)
            memory.append({"tool": "search_docs", "result": evidence})
            return {"answer": "I couldn't plan tool calls reliably. Here is evidence-based info.", "memory": memory}

        tool = call.get("tool")
        args = call.get("args", {})

        if tool == "final":
            return {"answer": args.get("answer", ""), "memory": memory}

        if tool == "search_docs":
            result = await tool_search_docs(store, **args)
        elif tool == "summarize":
            result = await tool_summarize(**args)
        elif tool == "extract_action_items":
            result = await tool_extract_action_items(**args)
        else:
            result = {"error": f"Unknown tool: {tool}", "raw_call": call}

        memory.append({"tool": tool, "args": args, "result": result})

    # If we exhausted steps, produce final answer from memory
    messages = [
        {"role": "system", "content": "Answer the user using only the MEMORY_JSON. If missing, say you don't know."},
        {"role": "user", "content": f"USER_QUESTION: {question}\n\nMEMORY_JSON:\n{json.dumps(memory)[:8000]}"},
    ]
    final = await chat(messages)
    return {"answer": final, "memory": memory}

