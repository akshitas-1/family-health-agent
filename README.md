# Family Health History Agent

A Streamlit + OpenAI agentic app that interviews you about your family's health
history and produces a downloadable, doctor-ready markdown summary with
personalized screening topics to discuss.

> ⚠️ This tool does **not** provide medical diagnoses. It organizes information
> and suggests conversations to have with a licensed healthcare provider.

## Features

- Multi-turn guided interview (profile → parents → siblings → children → grandparents → aunts/uncles)
- GPT-5-nano orchestrator with function calling (OpenAI Responses API)
- Three tools the agent calls itself:
  1. `update_family_tree` — records each relative as you go
  2. `search_screening_guidelines` — pulls current USPSTF / CDC guidance via the Responses API's built-in `web_search` tool
  3. `generate_doctor_doc` — writes a markdown summary you can download
- Sidebar with live progress, live family tree, download button, and reset

## Project layout

```
family-health-agent/
├── app.py              # Streamlit UI + session state
├── agent.py            # GPT-5-nano orchestrator (Responses API + tool loop)
├── tools.py            # The three tools + JSON schemas
├── requirements.txt
├── .env.example
└── README.md
```

## Run it locally

1. **Clone / open the folder**

   ```bash
   cd family-health-agent
   ```

2. **Create a virtualenv and install deps**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Add your OpenAI API key**

   ```bash
   cp .env.example .env
   # edit .env and paste your key:
   # OPENAI_API_KEY=sk-...
   ```

   Your key must have access to `gpt-5-nano` and to the Responses API
   `web_search` tool.

4. **Launch Streamlit**

   ```bash
   streamlit run app.py
   ```

   The app opens in your browser at http://localhost:8501.

## Using the app

1. The agent greets you and asks for your age + sex assigned at birth.
2. Answer one relative at a time. You can always say "I don't know" or
   "I'd rather not say" — the agent will log the gap and move on.
3. Watch the sidebar: the family tree fills in live as you talk, and the
   progress indicator shows which phase you're in.
4. When the tree is reasonably complete, the agent will summarize candidate
   risk patterns, look up current screening guidelines, and offer to
   generate your summary.
5. Hit **⬇ Download summary (Markdown)** in the sidebar and bring the file
   to your next doctor's appointment.

## Troubleshooting

- **`OPENAI_API_KEY is not set`** → make sure `.env` exists in the project root
  with a line `OPENAI_API_KEY=sk-...`, and that you launched `streamlit` from
  that same directory.
- **Tool call errors / web_search failures** → your API key may not have
  Responses API or web-search tool access. The app still works without it; it
  just won't cite external guidelines.
- **Reset** → click **🔄 Reset conversation** in the sidebar to start over.

## Safety

Every generated summary includes the non-diagnostic disclaimer. The system
prompt also instructs the model to redirect to urgent care if the user
describes an acute symptom.
