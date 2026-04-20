"""Tool functions the Family Health History agent can call."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from openai import OpenAI

log = logging.getLogger("family_health_agent.tools")

DISCLAIMER = (
    "**Important:** This document is not a medical diagnosis. It is an organized "
    "summary of family health history intended to help you have a more productive "
    "conversation with a licensed healthcare provider. Screening suggestions are "
    "general and based on publicly available guidelines; your doctor should "
    "personalize them for you."
)

RELATIVE_TYPES = {
    "mother", "father", "sister", "brother", "son", "daughter",
    "maternal_grandmother", "maternal_grandfather",
    "paternal_grandmother", "paternal_grandfather",
    "maternal_aunt", "maternal_uncle",
    "paternal_aunt", "paternal_uncle",
    "other",
}


def update_family_tree(
    family_tree: dict,
    relative_type: str,
    name_or_label: str,
    status: str,
    conditions: list[str] | None = None,
    age_of_onset: str | None = None,
    notes: str | None = None,
) -> dict:
    """Add or update one relative in the family tree dict held in session state.

    Returns a small confirmation payload for the model.
    """
    key = relative_type.strip().lower().replace(" ", "_")
    entry = {
        "relative_type": key,
        "label": name_or_label or key,
        "status": status,
        "conditions": conditions or [],
        "age_of_onset": age_of_onset,
        "notes": notes,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    family_tree.setdefault("relatives", []).append(entry)
    return {
        "ok": True,
        "stored": entry,
        "total_relatives": len(family_tree["relatives"]),
    }


def search_screening_guidelines(client: OpenAI, query: str) -> dict:
    """Use OpenAI's built-in web_search tool (Responses API) to look up
    current USPSTF / CDC screening guidance.
    """
    try:
        resp = client.with_options(timeout=60).responses.create(
            model="gpt-5-nano",
            tools=[{"type": "web_search"}],
            input=(
                "Find current US screening guidelines (prefer USPSTF, CDC, "
                "ACS, or major US medical society) relevant to: "
                f"{query}\n\n"
                "Return a concise bulleted list with source names and URLs "
                "where possible. Note year of guideline."
            ),
        )
        text = getattr(resp, "output_text", None) or ""
        sources: list[dict] = []
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                for ann in getattr(part, "annotations", []) or []:
                    url = getattr(ann, "url", None)
                    title = getattr(ann, "title", None)
                    if url:
                        sources.append({"title": title or url, "url": url})
        return {"ok": True, "summary": text, "sources": sources}
    except Exception as e:
        return {"ok": False, "error": str(e), "summary": "", "sources": []}


def generate_doctor_doc(
    user_profile: dict,
    family_tree: dict,
    identified_risks: list[str],
    screening_recommendations: list[str],
    sources: list[dict] | None = None,
) -> str:
    """Return a markdown document the user can download."""
    now = datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# Family Health History — Doctor Visit Summary")
    lines.append(f"_Generated {now}_\n")
    lines.append(DISCLAIMER + "\n")

    lines.append("## 1. Patient profile")
    if user_profile:
        for k, v in user_profile.items():
            if v:
                lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
    else:
        lines.append("- _Not provided_")
    lines.append("")

    lines.append("## 2. Family tree")
    relatives = family_tree.get("relatives", []) if family_tree else []
    if relatives:
        lines.append("| Relative | Label | Status | Conditions | Age of onset | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for r in relatives:
            conds = ", ".join(r.get("conditions") or []) or "—"
            lines.append(
                f"| {r.get('relative_type','')} "
                f"| {r.get('label','')} "
                f"| {r.get('status','')} "
                f"| {conds} "
                f"| {r.get('age_of_onset') or '—'} "
                f"| {r.get('notes') or ''} |"
            )
    else:
        lines.append("_No relatives recorded._")
    lines.append("")

    lines.append("## 3. Identified risk factors")
    if identified_risks:
        for r in identified_risks:
            lines.append(f"- {r}")
    else:
        lines.append("- None identified from the information provided.")
    lines.append("")

    lines.append("## 4. Suggested conversations with your doctor")
    if screening_recommendations:
        for s in screening_recommendations:
            lines.append(f"- {s}")
    else:
        lines.append("- Discuss general preventive screening appropriate for your age and sex.")
    lines.append("")

    lines.append("## 5. Sources")
    if sources:
        for s in sources:
            title = s.get("title") or s.get("url")
            url = s.get("url") or ""
            lines.append(f"- [{title}]({url})")
    else:
        lines.append("- USPSTF: https://www.uspreventiveservicestaskforce.org/")
        lines.append("- CDC: https://www.cdc.gov/")
    lines.append("")

    lines.append("---")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# ---------- OpenAI function-calling schema ----------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "update_family_tree",
        "description": (
            "Record or update one relative in the user's family tree. "
            "Call this every time you learn something new about a specific relative."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "relative_type": {
                    "type": "string",
                    "description": (
                        "One of: mother, father, sister, brother, son, daughter, "
                        "maternal_grandmother, maternal_grandfather, paternal_grandmother, "
                        "paternal_grandfather, maternal_aunt, maternal_uncle, paternal_aunt, "
                        "paternal_uncle, other."
                    ),
                },
                "name_or_label": {
                    "type": "string",
                    "description": "A short label for the relative (e.g. 'Mom', 'Uncle Joe', 'oldest sister').",
                },
                "status": {
                    "type": "string",
                    "description": "alive, deceased, or unknown. If deceased, include age/cause in notes.",
                },
                "conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Known medical conditions for this relative.",
                },
                "age_of_onset": {
                    "type": "string",
                    "description": "Approximate age when condition(s) started, if known. Free text ok.",
                },
                "notes": {
                    "type": "string",
                    "description": "Anything else worth noting (cause of death, uncertainty, etc.).",
                },
            },
            "required": ["relative_type", "name_or_label", "status"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_screening_guidelines",
        "description": (
            "Search the web for current US screening guidelines (USPSTF, CDC, ACS, etc.) "
            "relevant to the risks identified in the family history. Call this once you "
            "have a reasonably complete family tree and have identified candidate risks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language search query, e.g. 'breast cancer screening when mother diagnosed at 45'.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "generate_doctor_doc",
        "description": (
            "Generate the final downloadable markdown summary once the user is ready. "
            "Include identified risks and screening recommendations phrased as conversations "
            "to have with a doctor — never as diagnoses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "identified_risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Risk factors you identified from the family tree.",
                },
                "screening_recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggested topics / screenings to discuss with their doctor.",
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "required": ["title", "url"],
                        "additionalProperties": False,
                    },
                    "description": "Citations pulled from the web_search step.",
                },
            },
            "required": ["identified_risks", "screening_recommendations"],
            "additionalProperties": False,
        },
    },
]


def dispatch_tool(
    name: str,
    arguments: dict,
    *,
    client: OpenAI,
    state: dict,
) -> str:
    """Execute a tool by name. Returns a JSON string for the model."""
    log.info("[tool] ▶ %s args=%s", name, _preview(arguments))
    print(f"[tool] ▶ {name} args={_preview(arguments)}", flush=True)

    if name == "update_family_tree":
        result = update_family_tree(state["family_tree"], **arguments)
    elif name == "search_screening_guidelines":
        result = search_screening_guidelines(client, arguments["query"])
        state["last_search"] = result
        if result.get("sources"):
            seen = {s["url"] for s in state.get("sources", [])}
            for s in result["sources"]:
                if s["url"] not in seen:
                    state.setdefault("sources", []).append(s)
                    seen.add(s["url"])
    elif name == "generate_doctor_doc":
        md = generate_doctor_doc(
            user_profile=state.get("user_profile", {}),
            family_tree=state.get("family_tree", {}),
            identified_risks=arguments.get("identified_risks", []),
            screening_recommendations=arguments.get("screening_recommendations", []),
            sources=arguments.get("sources") or state.get("sources", []),
        )
        state["doctor_doc_md"] = md
        result = {"ok": True, "bytes": len(md)}
    else:
        result = {"ok": False, "error": f"unknown tool: {name}"}

    log.info("[tool] ◀ %s result=%s", name, _preview(result))
    print(f"[tool] ◀ {name} result={_preview(result)}", flush=True)
    return json.dumps(result)


def _preview(obj: Any, limit: int = 240) -> str:
    """Short stringified preview for logging."""
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"
