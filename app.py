"""Streamlit UI for the Family Health History Agent."""
from __future__ import annotations

import json

import streamlit as st

from agent import build_client, run_turn

st.set_page_config(
    page_title="Family Health History Agent",
    page_icon="🩺",
    layout="wide",
)


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])      # visible chat log: [{role, content}]
    ss.setdefault("conversation", [])  # Responses API input items
    ss.setdefault(
        "agent_state",
        {
            "user_profile": {},
            "family_tree": {"relatives": []},
            "sources": [],
            "last_search": None,
            "doctor_doc_md": None,
        },
    )
    ss.setdefault("phase", "intro")
    ss.setdefault("skip_search", False)


def _reset() -> None:
    for k in ("messages", "conversation", "agent_state", "phase", "skip_search"):
        st.session_state.pop(k, None)
    _init_state()


def _infer_phase(agent_state: dict) -> str:
    if agent_state.get("doctor_doc_md"):
        return "summary_ready"
    relatives = agent_state.get("family_tree", {}).get("relatives", [])
    if agent_state.get("last_search"):
        return "reviewing_guidelines"
    if len(relatives) >= 4:
        return "wrapping_up"
    if relatives:
        return "gathering_relatives"
    return "intro"


PHASE_LABELS = {
    "intro": "1. Introductions & your profile",
    "gathering_relatives": "2. Walking through relatives",
    "wrapping_up": "3. Summarizing risks",
    "reviewing_guidelines": "4. Looking up screening guidelines",
    "summary_ready": "5. Summary ready to download ✅",
}


def _render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.header("Family Health Agent")

        st.subheader("Settings")
        ss.skip_search = st.toggle(
            "Skip web search",
            value=ss.skip_search,
            help=(
                "When on, the agent won't call the live web_search tool for "
                "screening guidelines and will rely on its own knowledge. Use "
                "this as a fallback if the search step is slow or failing."
            ),
        )
        st.divider()

        phase = _infer_phase(ss.agent_state)
        ss.phase = phase
        st.subheader("Progress")
        for key, label in PHASE_LABELS.items():
            marker = "▶" if key == phase else ("✓" if _phase_order(key) < _phase_order(phase) else "○")
            st.write(f"{marker} {label}")

        st.divider()
        st.subheader("Family tree (live)")
        relatives = ss.agent_state["family_tree"].get("relatives", [])
        if relatives:
            rows = [
                {
                    "Relative": r.get("relative_type", ""),
                    "Label": r.get("label", ""),
                    "Status": r.get("status", ""),
                    "Conditions": ", ".join(r.get("conditions") or []) or "—",
                    "Onset": r.get("age_of_onset") or "—",
                }
                for r in relatives
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
            with st.expander("Raw JSON"):
                st.json(ss.agent_state["family_tree"])
        else:
            st.caption("No relatives recorded yet — chat with the agent to start.")

        st.divider()
        st.subheader("Download")
        doc = ss.agent_state.get("doctor_doc_md")
        if doc:
            st.download_button(
                "⬇ Download summary (Markdown)",
                data=doc,
                file_name="family_health_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.caption("The download button appears once the agent generates your summary.")

        st.divider()
        if st.button("🔄 Reset conversation", use_container_width=True):
            _reset()
            st.rerun()


def _phase_order(p: str) -> int:
    order = list(PHASE_LABELS.keys())
    return order.index(p) if p in order else -1


def _render_chat() -> None:
    ss = st.session_state
    st.title("🩺 Family Health History Agent")
    st.caption(
        "A guided conversation to document your family health history and "
        "produce a doctor-ready summary. Not a medical diagnosis."
    )

    if not ss.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I'll help you put together a clear picture of your family's "
                "health history — the kind of thing that's really useful to bring "
                "to a doctor's appointment.\n\n"
                "A quick heads-up: **I'm not a doctor and I won't diagnose anything.** "
                "I'm just going to ask some questions, organize your answers, and at "
                "the end generate a summary you can download.\n\n"
                "Ready? Let's start with **you** — could you tell me your age and "
                "sex assigned at birth?"
            )

    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Type your reply…")
    if not user_input:
        return

    ss.messages.append({"role": "user", "content": user_input})
    ss.conversation.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        client = build_client()
    except RuntimeError as e:
        with st.chat_message("assistant"):
            st.error(str(e))
        return

    tool_labels = {
        "update_family_tree": "Updating family tree…",
        "search_screening_guidelines": "Searching screening guidelines…",
        "generate_doctor_doc": "Generating summary…",
    }

    with st.chat_message("assistant"):
        status = st.status("Analyzing family tree…", expanded=False)

        def progress_cb(event: str, detail: str = "") -> None:
            if event == "thinking":
                status.update(label=detail or "Analyzing family tree…")
            elif event == "tool_start":
                label = tool_labels.get(detail, f"Running {detail}…")
                status.update(label=label)
                status.write(f"→ {label}")
            elif event == "tool_end":
                status.write(f"✓ {detail} done")
            elif event == "final":
                status.update(label="Done", state="complete")

        reply = ""
        try:
            reply, _ = run_turn(
                client,
                ss.conversation,
                ss.agent_state,
                progress_cb=progress_cb,
                skip_search=ss.skip_search,
            )
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Something went wrong talking to the model: {e}")
            return

        if reply:
            st.markdown(reply)
            ss.messages.append({"role": "assistant", "content": reply})
        else:
            st.markdown("_(The agent didn't produce a reply — try again.)_")


def main() -> None:
    _init_state()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
