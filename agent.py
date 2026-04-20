"""GPT-5-nano orchestrator for the Family Health History agent."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_SCHEMAS, dispatch_tool

load_dotenv()

logging.basicConfig(
    level=os.getenv("FHA_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("family_health_agent.agent")

MODEL = "gpt-5-nano"
REQUEST_TIMEOUT_S = 60
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are the Family Health History Agent — a warm, patient \
interviewer that helps a user document their family medical history and then \
produces a doctor-ready summary.

You are NOT providing medical diagnoses. You are organizing information and \
suggesting conversations to have with a healthcare provider. Always include \
this disclaimer in outputs.

## Your interview flow
1. Greet the user briefly and ask for their basic profile: age, sex assigned \
   at birth, and ethnicity (ethnicity only if they offer — it's optional and \
   relevant for some hereditary risks).
2. Walk through relatives in this exact order. Ask about ONE relative at a time. \
   Wait for the user's answer before moving to the next.
     a. Parents (mother, then father)
     b. Siblings (ask how many, then go through each)
     c. Children, if any
     d. Grandparents (maternal side, then paternal side)
     e. Aunts and uncles (optional — ask if they know of any notable conditions)
3. For each relative, ask:
   - Alive or deceased? If deceased, at what age and from what cause if known.
   - Any known medical conditions (cancer, heart disease, diabetes, stroke, \
     mental health, autoimmune, genetic conditions, etc.).
   - Approximate age of onset for each condition, if known.
4. If the user says "I don't know" or "I'd rather not say" — accept it warmly, \
   note the gap, optionally suggest who they might ask (e.g. "your mom might \
   remember"), and move on. Never push.
5. Call `update_family_tree` EVERY time you learn something concrete about a \
   specific relative (even if sparse — record 'unknown' conditions).
6. When the tree feels reasonably complete (parents + siblings + at least some \
   grandparent info, OR the user signals they're done), summarize what you \
   heard and identify candidate risk patterns (e.g. "early-onset breast cancer \
   on maternal side", "multiple relatives with type 2 diabetes").
7. Call `search_screening_guidelines` once or twice to pull current USPSTF/CDC \
   guidance relevant to those risks. If this tool is NOT available in your \
   current tool list, skip this step and proceed using your own training-time \
   knowledge of general US screening guidance — clearly note in the summary \
   that live web guidance was skipped.
8. Call `generate_doctor_doc` with identified_risks + screening_recommendations \
   (phrased as questions/topics to raise with a doctor, never as prescriptions) \
   + any sources returned from the search. Tell the user the summary is ready \
   to download from the sidebar.

## Tone
- Friendly and conversational, not clinical. Short messages. One question at a time.
- Acknowledge sensitivity — this can bring up hard memories.
- Never diagnose. Never say "you have" or "you will get" — always frame as \
  "this is worth discussing with your doctor."

## Safety
- Always include the non-diagnostic disclaimer in the final document.
- If the user describes an acute symptom or crisis, gently redirect them to \
  seek immediate medical care; do not continue the intake.
"""


ProgressCB = Callable[[str, str], None]


def _noop(event: str, detail: str = "") -> None:  # pragma: no cover
    pass


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_S)


def _filter_tools(skip_search: bool) -> list[dict[str, Any]]:
    if not skip_search:
        return TOOL_SCHEMAS
    return [t for t in TOOL_SCHEMAS if t.get("name") != "search_screening_guidelines"]


def run_turn(
    client: OpenAI,
    conversation: list[dict[str, Any]],
    state: dict,
    *,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    progress_cb: ProgressCB | None = None,
    skip_search: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Run one user turn through the Responses API until the model produces
    a final assistant message or we hit the tool-call ceiling.

    After `max_tool_rounds`, we force a final call with `tool_choice="none"`
    so the model has to produce a user-facing answer instead of looping.
    """
    cb = progress_cb or _noop
    tools = _filter_tools(skip_search)
    log.info(
        "run_turn start | skip_search=%s | tools=%s | max_rounds=%d",
        skip_search,
        [t["name"] for t in tools],
        max_tool_rounds,
    )
    print(
        f"[agent] turn start | skip_search={skip_search} "
        f"| tools={[t['name'] for t in tools]} | max_rounds={max_tool_rounds}",
        flush=True,
    )
    cb("thinking", "Analyzing family tree…")

    assistant_text = ""
    rounds_used = 0

    for round_idx in range(max_tool_rounds):
        rounds_used = round_idx + 1
        log.info("round %d: calling model", rounds_used)
        print(f"[agent] round {rounds_used}: calling model", flush=True)

        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            tools=tools,
            input=conversation,
            timeout=REQUEST_TIMEOUT_S,
        )

        tool_calls_this_round: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                tool_calls_this_round.append(
                    {
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    }
                )
            elif item_type == "message":
                for part in getattr(item, "content", []) or []:
                    text = getattr(part, "text", None)
                    if text:
                        text_parts.append(text)

        if tool_calls_this_round:
            conversation.extend(tool_calls_this_round)
            for call in tool_calls_this_round:
                try:
                    args = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                cb("tool_start", call["name"])
                output = dispatch_tool(
                    call["name"], args, client=client, state=state
                )
                cb("tool_end", call["name"])
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": output,
                    }
                )
            continue

        assistant_text = "\n".join(text_parts).strip()
        if assistant_text:
            conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                }
            )
        log.info("round %d: model produced final text", rounds_used)
        print(f"[agent] round {rounds_used}: model produced final text", flush=True)
        break
    else:
        # Loop exhausted — force the model to produce a user-facing reply
        # with no further tool calls.
        log.warning(
            "tool-call ceiling (%d) reached — forcing final answer without tools",
            max_tool_rounds,
        )
        print(
            f"[agent] !! tool-call ceiling ({max_tool_rounds}) reached — "
            "forcing final answer without tools",
            flush=True,
        )
        cb("thinking", "Wrapping up…")
        conversation.append(
            {
                "role": "user",
                "content": (
                    "[system] Tool-call budget reached. Please respond to the "
                    "user now with a final, helpful message. Do not request "
                    "more tool calls."
                ),
            }
        )
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            tools=tools,
            tool_choice="none",
            input=conversation,
            timeout=REQUEST_TIMEOUT_S,
        )
        text_parts = []
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []) or []:
                    text = getattr(part, "text", None)
                    if text:
                        text_parts.append(text)
        assistant_text = "\n".join(text_parts).strip() or (
            "I got stuck trying to organize the next step — could you tell me "
            "a bit more, or click Reset in the sidebar to start over?"
        )
        conversation.append({"role": "assistant", "content": assistant_text})

    cb("final", "")
    log.info("run_turn done | rounds_used=%d | reply_len=%d", rounds_used, len(assistant_text))
    print(
        f"[agent] turn done | rounds_used={rounds_used} "
        f"| reply_len={len(assistant_text)}",
        flush=True,
    )
    return assistant_text, conversation
