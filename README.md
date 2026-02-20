![FAIV Screenshot](faiv.png)

# FAIV — Five-Pillar AI Council Framework

> *In memory of Derek — uncle, father figure, and the person who first taught me that there are pillars of humanity that, if mastered, lead to a happier life. His belief was simple: focus on the pillars that matter, become a student of each, and you'll always find your way. FAIV turns that philosophy into something tangible — a council that deliberates the way he encouraged me to think.*

---

FAIV is a decision-intelligence framework that routes user queries through a council of 25 AI personas organized into five philosophical pillars. Rather than returning a single AI opinion, FAIV simulates a genuine multi-perspective debate — each council member argues from their unique worldview, challenges each other directly, and produces a synthesized consensus grounded in the actual deliberation.

The core idea: better decisions emerge from structured disagreement. FAIV doesn't optimize for a single "correct" answer — it forges answers through authentic tension between wisdom, strategy, creativity, foresight, and moral accountability. Users don't just receive answers; they participate in the deliberation itself, injecting their own arguments into the council's debate and watching as 25 deeply characterized personas respond, adapt, and sometimes change their minds.

**Live:** [https://faiv.ai](https://faiv.ai)

---

## How It Works

### The Five Pillars

| Pillar | Focus | Example Members |
|--------|-------|-----------------|
| **Wisdom** | Philosophy, ethics, historical precedent | Kyre (Skeptical Philosopher), Solyn (Ethical Historian) |
| **Strategy** | Tactical planning, risk assessment, game theory | Iom (Strategic Tactician), Neris (Economic Strategist) |
| **Expansion** | Growth, creativity, unconventional thinking | Zyra (Disruptive Innovator), Kael (Visionary Architect) |
| **Future** | Emerging trends, long-term impact, technology | Aen (Futurist Technologist), Torin (Speculative Theorist) |
| **Integrity** | Moral grounding, accountability, fairness | Cery (Moral Compass), Elyn (Justice Advocate) |

Each pillar contains 5 members (25 total), defined in the **Identity Codex** (`faiv_app/identity_codex.py`). Every member has 16 identity fields including principles, faith, vices, social dynamics, alliances, and conflicts — creating deeply distinct personas.

### The Deliberation Process

1. **Member Selection** — For each query, 1 random representative is chosen from each pillar (5 total for FAIV mode), or 1 from a single pillar if queried directly.
2. **Full Character Injection** — All 16 identity fields per member are injected into the system prompt so the model deeply embodies each character.
3. **Authentic Debate** — Members must give concrete, specific answers, directly challenge each other by name, and reference their principles. No abstract philosophizing.
4. **Consensus Synthesis** — The final consensus reflects what was actually debated, including confidence score, justification, and any dissenting opinions.

### Interactive Re-Deliberation

Users can engage directly with the council:
- Each speaker's statement appears as a **clickable tile**
- Click a tile to expand a reply field — agree, disagree, or redirect
- Submitting a reply triggers a **re-deliberation** where the same council members reconvene and incorporate the user's input
- Replied-to tiles show a "replied" badge that links to the user's reply, and the reply links back to the tile

---

## Architecture

```
faiv/
├── faiv_app/                    # Backend (Python/FastAPI)
│   ├── core.py                  # API server, deliberation engine, prompt construction
│   ├── identity_codex.py        # 25-member Identity Codex (5 pillars x 5 members)
│   ├── utils.py                 # Utility functions
│   └── __init__.py
├── faiv-console/                # Frontend (React)
│   ├── src/
│   │   └── components/
│   │       ├── FAIVConsole.js   # Main UI component
│   │       └── FAIVConsole.css  # Retro terminal styling
│   ├── public/
│   │   ├── favicon.png          # FAIV favicon
│   │   ├── index.html
│   │   └── manifest.json
│   └── package.json
├── passenger_wsgi.py            # cPanel/Passenger WSGI entry point
├── requirements.txt             # Python dependencies
├── .env                         # OpenAI API key (not committed)
└── .gitignore
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns API status, Redis status, model info |
| `POST` | `/query/` | Submit a query for council deliberation |
| `POST` | `/redeliberate/` | Re-deliberate with user interjection (same council reconvenes) |
| `POST` | `/reset/` | Reset a session's conversation history |

### Tech Stack

- **Backend:** FastAPI, OpenAI GPT-4o, Redis (optional, with in-memory fallback), Pydantic
- **Frontend:** React 19, vanilla CSS (retro terminal aesthetic)
- **Hosting:** cPanel with Phusion Passenger (ASGI-to-WSGI via `a2wsgi`)
- **Sessions:** Redis or in-memory — conversation history persists across queries within a session

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Backend

```sh
cd faiv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-proj-your-key-here
```

Start the API server:
```sh
uvicorn faiv_app.core:fastapi_app --host 127.0.0.1 --port 8000 --reload
```

Verify: `curl http://127.0.0.1:8000/health`

### Frontend

```sh
cd faiv-console
npm install
npm start
```

Opens at `http://localhost:3000`. The frontend calls `http://127.0.0.1:8000` by default (configured via `REACT_APP_API_BASE_URL`).

---

## Production Deployment (cPanel)

### Backend

1. Upload the project to the server (e.g., `/home/user/api/`)
2. Create a Python app in cPanel pointing to the project root
3. Install dependencies in the virtualenv:
   ```sh
   pip install -r requirements.txt
   pip install a2wsgi python-dotenv
   ```
4. Create `.env` with your OpenAI key
5. The `passenger_wsgi.py` bridges FastAPI (ASGI) to Passenger (WSGI) via `a2wsgi`
6. Update CORS origins in `core.py` to include your production domain
7. Restart Passenger: `touch tmp/restart.txt`

### Frontend

Build with the production API URL:
```sh
cd faiv-console
REACT_APP_API_BASE_URL=https://api.yourdomain.com npm run build
```

Deploy the `build/` contents to your web document root. Add an `.htaccess` for client-side routing:
```apache
RewriteEngine On
RewriteBase /
RewriteRule ^index\.html$ - [L]
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule . /index.html [L]
```

---

## The Identity Codex

Each of the 25 council members is defined by 16 fields:

| Field | Purpose |
|-------|---------|
| `claimed-title` | Their self-identified role (e.g., "The Skeptical Philosopher") |
| `role` | Professional archetype |
| `principles` | Core beliefs that drive their reasoning |
| `aligns_with` | Members they naturally agree with |
| `conflicts_with` | Members they naturally challenge |
| `contribution` | What they uniquely bring to deliberation |
| `faith` | Spiritual/philosophical worldview |
| `fight-for` | The cause they champion above all |
| `social-level` | Introvert/extrovert scale (1-10) |
| `favorite-activity` | What energizes them |
| `finality` | Their view on permanence and endings |
| `chosen-memory` | A defining personal memory |
| `vice-of-choice` | Their human indulgence |
| `one-piece-of-wisdom` | Their signature insight |
| `real-world-analogy` | A real-world figure they resemble |
| `example-influence` | How they'd respond in a sample scenario |

These fields are injected into the system prompt so that GPT-4o can embody each character with depth and consistency.

---

## Design Philosophy

FAIV is built on the premise that better decisions emerge from structured disagreement. Rather than a single AI voice optimizing for agreeableness, FAIV creates productive tension between five philosophical orientations:

- **Wisdom** grounds decisions in historical precedent and ethical reflection
- **Strategy** pressure-tests feasibility and risk
- **Expansion** pushes boundaries and challenges assumptions
- **Future** considers long-term trajectories and emerging patterns
- **Integrity** holds the process accountable to fairness and moral standards

The result is a consensus that has been stress-tested from multiple angles — not just an answer, but an answer that has survived genuine debate.

### What Makes FAIV Different

Most AI tools give you one voice with one opinion. FAIV gives you five perspectives in active conflict — and then invites you to join the argument. The interactive re-deliberation system means you aren't a passive consumer of AI output. You're a participant in a structured debate, pushing back on reasoning you disagree with and watching the council adapt in real time.

Each council member isn't a shallow "mode" or persona label. They're defined by 16 deep identity fields — their faith, their vices, their defining memories, the causes they fight for, the members they align with and conflict against. When Kyre (The Skeptical Philosopher) challenges Iom (The Calculated Risk-Taker), it's not templated disagreement — it's a genuine clash of worldviews informed by the full depth of their character.

The five pillars themselves reflect a fundamental belief: that no single lens on reality is sufficient. Wisdom without strategy is impractical. Strategy without integrity is dangerous. Expansion without foresight is reckless. Every decision benefits from being examined through all five, and FAIV makes that examination automatic, authentic, and interactive.

---

## License

MIT
