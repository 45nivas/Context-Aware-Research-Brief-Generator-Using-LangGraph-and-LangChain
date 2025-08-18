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

# 🧩 Graph Architecture

**Visual Representation:**
![Architecture](docs/architecture.png)

**Workflow Nodes:**
1. Context Summarization
2. Planning
3. Search
4. Content Fetching
5. Per-Source Summarization
6. Synthesis
7. Post-Processing

Each node is implemented as an async function, validated with Pydantic models, and orchestrated using LangGraph.

---


# 🧠 Model & Tool Selection Rationale

- **Google Gemini 1.5 Flash:** Large context, fast, best for planning & synthesis (free tier, quota limited)
- **OpenRouter:** Main LLM provider, supports multiple open models, custom integration for LangChain/LangGraph
- **SerpAPI:** High-quality Google search results (free tier, 100 searches/month)
- **DuckDuckGo:** Unlimited, privacy-focused fallback search

Selection is based on maximizing free access, reliability, and diversity of sources. Fallback logic ensures graceful degradation if quotas are exceeded.

---

# 🗂️ Schema Definitions & Validation Strategy

All workflow states and outputs are validated using Pydantic models:

- **BriefRequest:** topic, depth, follow_up, user_id
- **FinalBrief:** title, executive summary, key findings, detailed analysis, implications, limitations, references, metadata
- **SourceSummary:** title, summary, relevance_score, main_topics, word_count
- **Reference:** title, url, access_date, relevance_note

Validation is enforced at every node transition, ensuring structured outputs and robust error handling.

---

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
- 🌐 **OpenRouter** — Main LLM provider, supports open models, custom integration
- 🔎 **Search** — SerpAPI (free tier) + DuckDuckGo (free fallback)

---

# 🗂️ Data Models (Pydantic Validation)

- **BriefRequest:** topic, depth, follow_up, user_id
- **FinalBrief:** title, executive summary, key findings, detailed analysis, implications, limitations, references, metadata

---



# 🚀 Deployment Instructions

**Local:**
```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt

# Add .env file:
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
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


# 📬 Example Requests & Outputs

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
- **LLMs:** Google Gemini 1.5 Flash, OpenRouter
- **Search APIs:** SerpAPI, DuckDuckGo
- **Database:** SQLite (SQLAlchemy ORM)
- **Deployment:** Render

---


# 🛡️ Error Handling
- Fallback from Gemini → OpenRouter in case of API rate limit errors.
- Graceful degradation: returns structured error messages with metadata.

---


# 💸 Cost & Latency Benchmarks

- **Cost:** $0 (free tiers for all APIs and models)
- **LLM Quotas:** Gemini (50 requests/day), SerpAPI (100 searches/month)
- **Latency:** 60–90 sec per request (may increase with API rate limits or quota exhaustion)
- **Local Mode:** Ollama provides unlimited, fast inference for development and fallback

---
- **Cost:** $0 (free tiers)
- **Latency:** 60–90 sec per request (may increase with API rate limits)

---



# 🚧 Limitations & Areas for Improvement
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


---


## 🚀 Live API Endpoint & CLI

Your project is deployed and publicly accessible at:

```
https://<your-render-service-url>
```

To test the `/brief` endpoint, use the following curl command:

```bash
curl -X 'POST' \
  'https://<your-render-service-url>/brief' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "topic": "Impact of renewable energy on global economy",
    "depth": 2,
    "follow_up": false,
    "user_id": "test-user"
  }'
```

Replace `<your-render-service-url>` with your actual Render deployment URL.

**CLI Usage:**
```bash
python main.py brief --topic "Impact of renewable energy on global economy" --depth 2 --user_id "test-user"
```

---


## 🔎 Observability and Tracing

This project uses [LangSmith](https://smith.langchain.com/) for workflow tracing and observability.

You can view a public trace of a sample research brief workflow here:

```
https://smith.langchain.com/public/<your-trace-id>
```

Replace `<your-trace-id>` with your actual public LangSmith trace link.

---

## ✅ Assignment Compliance Checklist

- [x] Uses **LangGraph** for workflow orchestration
- [x] Uses **LangChain** for LLM/tool abstraction
- [x] Supports **two LLM providers**: Google Gemini & OpenRouter
- [x] Custom OpenRouter integration for LangChain/LangGraph
- [x] Pydantic schema validation at every node
- [x] API and CLI interfaces for brief generation
- [x] Context summarization and session retention
- [x] Multi-source search with fallback logic
- [x] Observability via LangSmith tracing
- [x] Comprehensive unit, integration, and end-to-end tests (`/tests` folder)
- [x] Deployment instructions for local and Render
