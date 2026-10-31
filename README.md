# OpenPoke 🌴 — Jev-powered retrieval fork

**This fork adds a System 1 / System 2 retrieval pipeline that fixes the Execution Agent Overload problem** the OpenPoke team flagged as their biggest unsolved gap. 

Baseline OpenPoke renders every execution agent (hundreds of them, after a few weeks of use) into the Interaction Agent's prompt on every turn. This fork filters that roster down to ~10 via a four-stage pipeline (plus a fifth that fires only on new-agent spawns), using TypeSafe's [Jev](https://typesafe.ai) for the classification work and OpenAI embeddings for semantic recall. On a 13-case benchmark, routing accuracy goes from **69.2% → 92.3%** and the visible roster shrinks from **25 → 4.6** agents per turn.

### What's new in this fork

**New package** `server/services/retrieval/`:
- `router.py` — Jev-based intent + domain classifier
- `embeddings.py` — OpenAI semantic search over cached per-agent embeddings
- `reranker.py` — Jev Noul rerank over top-K cosine hits
- `verifier.py` — Jev duplicate check before new agent spawning
- `descriptions.py` — Haiku-generated one-line agent descriptions
- `jev_client.py` — TypeSafe SDK wrapper

**Modified** OpenPoke internals:
- `server/services/execution/roster.py` — upgraded from `list[str]` to `AgentRecord` with descriptions, timestamps, usage counts, cached embeddings. Backward-compat with the original array-of-strings `roster.json`.
- `server/agents/interaction_agent/agent.py` — `_render_active_agents` now returns a filtered 10-agent roster instead of all N
- `server/agents/interaction_agent/tools.py` — spawn path runs the duplicate check before adding to roster

**New eval harness** at `server/eval/`:
- 13 hand-written benchmark cases across 6 categories (clean follow-ups, semantic follow-ups, disambiguation, stale, spawn-only, chat-only)
- Deterministic scoring over the exact `<active_agents>` block the IA would see
- Run: `python -m server.eval.run` for the ablation table, or `python -m server.eval.charts --rerun` to regenerate the presentation charts.

### Extra setup for the hybrid pipeline

Three API keys needed in `.env` on top of OpenPoke's standard config:

```bash
TYPESAFE_API_KEY=   # https://typesafe.ai — powers router + rerank + verifier
OPENAI_API_KEY=     # https://platform.openai.com — semantic embeddings
# OPENROUTER_API_KEY is already required by base OpenPoke; this fork also uses
# it for the Haiku description generator.
```

Without `TYPESAFE_API_KEY`, the pipeline falls back to baseline-ish behaviour (hot cache + lexical embedding fallback). All stages degrade gracefully.

---

OpenPoke is a simplified, open-source take on [Interaction Company’s](https://interaction.co/about) [Poke](https://poke.com/) assistant—built to show how a multi-agent orchestration stack can feel genuinely useful. It keeps the handful of things Poke is great at (email triage, reminders, and persistent agents) while staying easy to spin up locally.

- Multi-agent FastAPI backend that mirrors Poke's interaction/execution split, powered by [OpenRouter](https://openrouter.ai/).
- Gmail tooling via [Composio](https://composio.dev/) for drafting/replying/forwarding without leaving chat.
- Trigger scheduler and background watchers for reminders and "important email" alerts.
- Next.js web UI that proxies everything through the shared `.env`, so plugging in API keys is the only setup.

## Requirements
- Python 3.10+
- Node.js 18+
- npm 9+

## Quickstart
1. **Clone and enter the repo.**
   ```bash
   git clone https://github.com/shlokkhemani/OpenPoke
   cd OpenPoke
   ```
2. **Create a shared env file.** Copy the template and open it in your editor:
   ```bash
   cp .env.example .env
   ```
3. **Get your API keys and add them to `.env`:**
   
   **OpenRouter (Required)**
   - Create an account at [openrouter.ai](https://openrouter.ai/)
   - Generate an API key
   - Replace `your_openrouter_api_key_here` with your actual key in `.env`
   
   **Composio (Required for Gmail)**
   - Sign in at [composio.dev](https://composio.dev/)
   - Create an API key
   - Set up Gmail integration and get your auth config ID
   - Replace `your_composio_api_key_here` and `your_gmail_auth_config_id_here` in `.env`
4. **(Required) Create and activate a Python 3.10+ virtualenv:**
   ```bash
   # Ensure you're using Python 3.10+
   python3.10 -m venv .venv
   source .venv/bin/activate
   
   # Verify Python version (should show 3.10+)
   python --version
   ```
   On Windows (PowerShell):
   ```powershell
   # Use Python 3.10+ (adjust path as needed)
   python3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # Verify Python version
   python --version
   ```

5. **Install backend dependencies:**
   ```bash
   pip install -r server/requirements.txt
   ```
6. **Install frontend dependencies:**
   ```bash
   npm install --prefix web
   ```
7. **Start the FastAPI server:**
   ```bash
   python -m server.server --reload
   ```
8. **Start the Next.js app (new terminal):**
   ```bash
   npm run dev --prefix web
   ```
9. **Connect Gmail for email workflows.** With both services running, open [http://localhost:3000](http://localhost:3000), head to *Settings → Gmail*, and complete the Composio OAuth flow. This step is required for email drafting, replies, and the important-email monitor.

The web app proxies API calls to the Python server using the values in `.env`, so keeping both processes running is required for end-to-end flows.

## Project Layout
- `server/` – FastAPI application and agents
- `web/` – Next.js app
- `server/data/` – runtime data (ignored by git)

## License
MIT — see [LICENSE](LICENSE).
