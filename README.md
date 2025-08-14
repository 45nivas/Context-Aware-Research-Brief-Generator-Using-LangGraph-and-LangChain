![Architecture](docs/architecture.png)

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-orange)
![Render](https://img.shields.io/badge/Deployed%20on-Render-purple)

<a href="https://context-aware-research-brief-generator-f2se.onrender.com/docs" target="_blank">
    <img src="https://img.shields.io/badge/View%20API%20Docs-Click%20Here-brightgreen?style=for-the-badge">
</a>

# 🚀 Objective & Problem Statement

**❓ Problem:**
Researchers spend a lot of time manually collecting, analyzing, and summarizing information. This is slow, inconsistent, and prone to bias.

**💡 Solution:**
An AI-powered research assistant that automatically creates **structured, evidence-linked research briefs**.
- 🧩 Uses **LangGraph** for workflow orchestration
- 🤖 Uses **LangChain** for LLM and tool interaction
- 🔄 Keeps context across sessions for follow-up questions

---

# ✨ Features
- 📄 Automated, structured research brief generation.
- 🔍 Multi-source web search with fallback logic.
- 🤖 Multi-model LLM orchestration using LangGraph + LangChain.
- 🧠 Session context retention for follow-up queries.
- 📊 Evidence-linked references with timestamps.

---

# 🎯 Live Demo
Try it here: [**API Documentation on Render**](https://context-aware-research-brief-generator-f2se.onrender.com/docs)

---

# 🛠️ How It Works (Workflow)

1. 🟢 **Start Request**
2. 🧠 **Context Summarization** (only if follow-up)
3. 📝 **Planning** (generate steps)
4. 🔍 **Search** (multi-source with fallback between SerpAPI & DuckDuckGo)
5. 📄 **Content Fetching** (full text, skips if no results)
6. ✍️ **Per-Source Summarization**
7. 🏗️ **Synthesis** (final brief creation)
8. ✅ **Post-Processing** (validation & save)
9. 🏁 **End Result**

---

# ⚙️ Models & Tools

- ⚡ **Google Gemini 1.5 Flash** — Large context, fast, best for planning & synthesis
- 🧠 **OpenAI GPT-3.5-Turbo** — Fallback, faster for summarization
- 🖥️ **Ollama (local)** — Unlimited use for offline dev
- 🔎 **Search** — SerpAPI (free tier) + DuckDuckGo (free fallback)

---

# 🗂️ Data Models (Pydantic Validation)

- **BriefRequest:** topic, depth, follow_up, user_id
- **FinalBrief:** title, executive summary, key findings, detailed analysis, implications, limitations, references, metadata

---

# 🏃 How to Run

**Local:**
```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt

# Add .env file:
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
SERPAPI_API_KEY=...
LANGCHAIN_API_KEY=...

uvicorn app.api:app --reload
```

Access: [http://127.0.0.1:8000](http://127.0.0.1:8000)

**Deploy on Render:**
- Service: Web Service
- Build: `pip install -r requirements.txt`
- Start: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.api:app`
- Env Vars: Add same keys in Render settings

---

# 📬 API Example

**POST /brief**
```json
{
  "topic": "Impact of AI on education",
  "depth": 2,
  "follow_up": false,
  "user_id": "test-user-1"
}
```

**Example Output**
```json
{
  "title": "Research Brief: Impact of AI on education",
  "executive_summary": "Synthesis failed.",
  "key_findings": [],
  "detailed_analysis": "",
  "implications": "",
  "limitations": "",
  "references": [
    {
      "title": "Error",
      "url": "",
      "access_date": "2025-08-14T18:07:38.947329",
      "relevance_note": "Synthesis failed"
    }
  ],
  "metadata": {
    "creation_timestamp": "2025-08-14T18:07:38.952476",
    "research_duration": 47,
    "total_sources_found": 5,
    "sources_used": 5,
    "confidence_score": 0.1,
    "depth_level": 2,
    "token_usage": {}
  }
}
```

---

# 🛠 Tech Stack
- **Backend:** FastAPI, Python 3.13
- **AI Orchestration:** LangGraph, LangChain
- **LLMs:** Google Gemini 1.5 Flash, OpenAI GPT-3.5 Turbo, Ollama
- **Search APIs:** SerpAPI, DuckDuckGo
- **Database:** SQLite (SQLAlchemy ORM)
- **Deployment:** Render

---

# 🛡️ Error Handling
- Fallback from Gemini → GPT-3.5 in case of API rate limit errors.
- Graceful degradation: returns structured error messages with metadata.

---

# 💸 Cost & Performance
- **Cost:** $0 (free tiers)
- **Latency:** 60–90 sec per request (may increase with API rate limits)

---

# 🚧 Current Limitations & Future Plans
**Limitations:**
- ⏳ Free-tier API rate limits
- 🔍 Search quality depends on public search engines
- 🚫 Some sites block content fetching

**Future Improvements:**
- 📚 Add academic databases (arXiv, Google Scholar)
- 🕵️‍♂️ Use headless browser for complex sites
- 🌐 Multi-language & translation support

---

# 👨‍💻 Author
Developed by **Lakshmi Nivas**  
GitHub: [45nivas](https://github.com/45nivas)
