# 1. Objective & Problem Statement

**Problem:**  
Researchers spend a lot of time manually collecting, analyzing, and summarizing information. This is slow, inconsistent, and prone to bias.  

**Solution:**  
An AI-powered research assistant that automatically creates **structured, evidence-linked research briefs**.  
- Uses **LangGraph** for workflow orchestration  
- Uses **LangChain** for LLM and tool interaction  
- Keeps context across sessions for follow-up questions  

---

# 2. How It Works (Workflow)

1. **Start Request**  
2. **Context Summarization** (only if follow-up)  
3. **Planning** (generate steps)  
4. **Search** (multi-source with fallback between SerpAPI & DuckDuckGo)  
5. **Content Fetching** (full text, skips if no results)  
6. **Per-Source Summarization**  
7. **Synthesis** (final brief creation)  
8. **Post-Processing** (validation & save)  
9. **End Result**  

---

# 3. Models & Tools

- **Google Gemini 1.5 Flash** → Large context, fast, best for planning & synthesis  
- **OpenAI GPT-3.5-Turbo** → Fallback, faster for summarization  
- **Ollama (local)** → Unlimited use for offline dev  
- **Search** → SerpAPI (free tier) + DuckDuckGo (free fallback)  

---

# 4. Data Models (Pydantic Validation)

**BriefRequest:** topic, depth, follow_up, user_id  
**FinalBrief:** title, executive summary, key findings, detailed analysis, implications, limitations, references, metadata  

---

# 5. How to Run

**Local:**
```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt


#  Add .env file:
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
SERPAPI_API_KEY=...
LANGCHAIN_API_KEY=...
Run:

uvicorn app.api:app --reload


Access: http://127.0.0.1:8000

Deploy on Render:

Service: Web Service

Build: pip install -r requirements.txt

Start: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.api:app

Env Vars: Add same keys in Render settings

6. API Example

POST /brief

{
  "topic": "Impact of AI on education",
  "depth": 2,
  "follow_up": false,
  "user_id": "test-user-1"
}


Response includes:

Title

Executive Summary

Key Findings (list)

Detailed Analysis

Implications

Limitations

References (with URLs)

Metadata (timestamps, sources, confidence score, etc.)

7. Cost & Performance

Cost: $0 (free tiers)

Latency: 60–90 sec per request (may increase with API rate limits)

8. Current Limitations & Future Plans

Limitations:

Free-tier API rate limits

Search quality depends on public search engines

Some sites block content fetching

Future Improvements:

Add academic databases (arXiv, Google Scholar)

Use headless browser for complex sites

Multi-language & translation support


---

If you paste this into your README.md and commit, GitHub will render it with **clear headings, spacing, and bullet points** — no more dull wall of text.  

If you want, I can also **add GitHub-style emoji icons** to each section so it pops visually in dark mode. That will make it stand out even more.
