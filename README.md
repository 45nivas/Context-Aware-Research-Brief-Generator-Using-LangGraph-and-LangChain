Context-Aware Research Brief Generator

1. Objective & Problem Statement

Problem Statement

Research professionals and analysts spend significant time manually collecting, analyzing, and synthesizing information from multiple sources to create comprehensive research briefs. This process is time-consuming, prone to human bias, and often lacks consistency.

Objective

This project is an AI-powered research assistant designed to solve this problem by generating structured, evidence-linked research briefs automatically. It uses LangGraph for workflow orchestration and LangChain for interacting with LLMs and external tools. A key feature is its ability to support follow-up queries by maintaining and summarizing user context across sessions.

2. Architecture Overview

The application is built around a LangGraph state machine that executes a series of nodes to generate the research brief.

Workflow Graph Structure

┌─────────────────────┐
│   START REQUEST     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Context             │◄─── Skips if not a follow-up
│ Summarization       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Planning            │
│ (Generate Steps)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Search              │◄─── Falls back between providers
│ (Multi-source)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Content Fetching    │◄─── Skips if no results
│ (Full Text)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Per-Source          │
│ Summarization       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Synthesis           │
│ (Final Brief)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Post-Processing     │
│ (Validation & Save) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    END RESULT       │
└─────────────────────┘


3. Model and Tool Selection Rationale

This project is designed to be run using free-tier and locally available models to ensure accessibility and minimize costs.

Primary Model (Google Gemini 1.5 Flash): Chosen for its large context window, fast response times, and strong performance on structured data generation. It is ideal for the complex planning and synthesis tasks.

Secondary Model (OpenAI GPT-3.5-Turbo): Used as a fallback and for less complex tasks like context_summarization and per_source_summarization due to its speed and the availability of a free tier.

Local Model (Ollama): Supported for local development, offering unlimited usage for users who have it installed.

Search Tools (SerpAPI & DuckDuckGo): A combination of a free-tier API (SerpAPI) and a completely free tool (DuckDuckGo) provides robust search capabilities. The system automatically falls back from SerpAPI to DuckDuckGo to ensure high availability.

4. Schema Definitions & Validation

All data structures, from the initial request to the final brief, are defined and validated using Pydantic models to ensure type safety and data integrity. LLM outputs are parsed and validated against these schemas, with automatic retries for invalid outputs.

Core Data Models:

class BriefRequest(BaseModel):
    topic: str = Field(..., min_length=10, max_length=500)
    depth: DepthLevel = Field(default=DepthLevel.STANDARD)
    follow_up: bool = Field(default=False)
    user_id: str = Field(..., min_length=1, max_length=100)

class FinalBrief(BaseModel):
    title: str
    executive_summary: str
    key_findings: List[str]
    detailed_analysis: str
    implications: str
    limitations: str
    references: List[Reference]
    metadata: BriefMetadata


5. How to Run and Deploy

Local Development

Clone the repository and install dependencies:

git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt


Set up environment variables:
Create a .env file and add your API keys:

GOOGLE_API_KEY="your_google_api_key"
OPENAI_API_KEY="your_openai_api_key"
SERPAPI_API_KEY="your_serpapi_api_key"
LANGCHAIN_API_KEY="your_langsmith_api_key"


Run the server:

uvicorn app.api:app --reload


The API will be available at http://127.0.0.1:8000.

Production Deployment (Render)

Service Type: Web Service

Build Command: pip install -r requirements.txt

Start Command: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.api:app

Environment Variables: Add the same key-value pairs from your .env file to the Render environment settings.

6. Example API Usage

Endpoint: POST /brief

Request Body:

{
  "topic": "Impact of AI on education",
  "depth": 2,
  "follow_up": false,
  "user_id": "test-user-1"
}


Successful Response (200 OK):

{
  "title": "The Impact of AI on Education: A Research Brief",
  "executive_summary": "The integration of artificial intelligence (AI) in education is a rapidly evolving field, showing both promise and limitations...",
  "key_findings": [
    "AI tools are enhancing classroom engagement according to U.S. educators.",
    "AI is personalizing and improving the efficiency of studying for students.",
    "The impact of AI on university education is currently modest and varies significantly.",
    "AI tools are being utilized by educators to create interactive learning experiences.",
    "There is a need for clearer guidance on the ethical and effective use of AI in education."
  ],
  "detailed_analysis": "The available sources present a mixed picture of AI's impact on education...",
  "implications": "The findings suggest several key implications for the future of AI in education...",
  "limitations": "This research brief is limited by the available source material...",
  "references": [
    {
      "title": "Gen Z educators embrace AI tools more often than Gen X",
      "url": "[https://www.eschoolnews.com/digital-learning/2025/07/16/gen-z-educators-embrace-ai-tools-more-often-than-gen-x/](https://www.eschoolnews.com/digital-learning/2025/07/16/gen-z-educators-embrace-ai-tools-more-often-than-gen-x/)",
      "access_date": "2024-07-27T12:00:00Z",
      "relevance_note": "Provides insights into generational differences in AI adoption among educators and highlights the need for clearer guidelines."
    }
  ],
  "metadata": {
    "creation_timestamp": "2025-08-14T15:14:10.810320",
    "research_duration": 83,
    "total_sources_found": 5,
    "sources_used": 5,
    "confidence_score": 0.7,
    "depth_level": 2,
    "token_usage": {}
  }
}


7. Cost & Latency Benchmarks

Cost: As this project is configured to use the free tiers of its external services, the estimated operational cost is $0.

Latency: Performance is subject to the rate limits of the free-tier APIs (e.g., Google Gemini's 15 requests/minute). A standard request typically takes 60-90 seconds. During periods of heavy use, latency can increase as the application's built-in retry logic waits for the rate limits to reset.

8. Limitations & Future Improvements

Current Limitations

Rate Limiting: The system's speed is constrained by the free-tier API rate limits.

Search Quality: Output quality is dependent on the relevance of results from public search engines.

Content Fetching: Can fail for websites with strong anti-scraping measures.

Future Improvements

Integrate with academic databases (e.g., arXiv, Google Scholar) for higher-quality sources.

Implement a more sophisticated content fetching mechanism, such as a headless browser, to handle dynamic websites.

Add support for multi-language research and translation.