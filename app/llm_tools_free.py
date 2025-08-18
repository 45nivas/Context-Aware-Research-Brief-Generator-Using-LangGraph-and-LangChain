"""
FREE LLM and search tools for the Research Brief Generator.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import httpx
import logging
import re
from typing import List, Dict, Any, Optional, Union

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from ddgs import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from app.models import SourceSummary

logger = logging.getLogger(__name__)

class OpenRouterLLM:
    """Custom LLM wrapper for OpenRouter API."""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def ainvoke(self, messages: List[Union[HumanMessage, SystemMessage]]) -> str:
        prompt = "\n".join([m.content for m in messages])
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Custom OpenRouter API call failed: {e}")
            return f"Error: {e}"

class FreeLLMManager:
    """Manages free LLM models including OpenRouter and Gemini."""
    def __init__(self):
        self.token_counts = {}
        self._primary_llm = None
        self._secondary_llm = None
        self._check_availability()
    
    def _check_availability(self):
        self.available_services = []
        if os.getenv("OPENROUTER_API_KEY"):
            self.available_services.append("openrouter")
            logger.info("✅ OpenRouter is available.")
        else:
            logger.warning("❌ OpenRouter not available (missing OPENROUTER_API_KEY).")

        if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            self.available_services.append("gemini")
            logger.info("✅ Google Gemini is available.")
        else:
            logger.warning("❌ Google Gemini not available (missing GOOGLE_API_KEY or package).")
            
    def reset_token_usage(self):
        self.token_counts = {"total_tokens": 0}

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_counts

    @property
    def primary_llm(self) -> Optional[OpenRouterLLM]:
        if self._primary_llm is None and "openrouter" in self.available_services:
            self._primary_llm = OpenRouterLLM(api_key=os.getenv("OPENROUTER_API_KEY"), model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct"))
        return self._primary_llm

    @property
    def secondary_llm(self) -> Optional[BaseLanguageModel]:
        if self._secondary_llm is None and "gemini" in self.available_services:
            self._secondary_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        return self._secondary_llm
    
    async def generate_with_fallback(self, messages: List[Union[HumanMessage, SystemMessage]]) -> str:
        prompt_tokens = sum(len(m.content) for m in messages) // 4 
        models_to_try = [m for m in [self.primary_llm, self.secondary_llm] if m is not None]
        if not models_to_try:
            raise Exception("No LLM models are available.")
        for model in models_to_try:
            try:
                response = await model.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                if content and "Error:" not in content:
                    completion_tokens = len(content) // 4
                    self.token_counts["total_tokens"] = self.token_counts.get("total_tokens", 0) + prompt_tokens + completion_tokens
                    return content.strip()
            except Exception as e:
                logger.warning(f"Model {type(model).__name__} failed: {e}")
        raise Exception("All available LLM models failed.")

class FreeSearchManager:
    """Manages free search tools."""
    async def search_with_fallback(self, query: str, max_results: int = 5) -> List[dict]:
        if not DUCKDUCKGO_AVAILABLE:
            logger.error("DuckDuckGo search is not installed. Cannot perform search.")
            return []
        try:
            # CORRECT WAY: Instantiate DDGS and run the synchronous text method in a thread
            ddgs = DDGS()
            results = await asyncio.to_thread(ddgs.text, query, max_results=max_results)
            
            # The ddgs library returns 'href', but our models expect 'url'
            formatted_results = []
            for res in results:
                formatted_results.append({
                    "title": res.get("title"),
                    "url": res.get("href"), # Rename key for compatibility
                    "snippet": res.get("body")
                })
            return formatted_results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

class ContentFetcher:
    """Fetches and cleans content from web URLs."""
    async def fetch_content(self, url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=30.0, headers={'User-Agent': 'Mozilla/5.0'}) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.text
                content = re.sub(r'<script.*?</script>|<style.*?</style>', '', content, flags=re.DOTALL)
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                return content
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return None
            
    async def close(self):
        pass # This method exists to prevent shutdown errors

llm_manager = FreeLLMManager()
search_manager = FreeSearchManager()
content_fetcher = ContentFetcher()

async def generate_text_with_fallback(messages: List[Union[HumanMessage, SystemMessage]]) -> str:
    return await llm_manager.generate_with_fallback(messages)

async def search_web_free(query: str) -> List[dict]:
    return await search_manager.search_with_fallback(query)

async def fetch_content_free(url: str) -> str:
    content = await content_fetcher.fetch_content(url)
    return content or ""

async def summarize_source_free(content: str, query: str) -> SourceSummary:
    try:
        prompt = f"Summarize the following content in the context of the research query: '{query}'\n\nContent:\n{content[:8000]}"
        messages = [HumanMessage(content=prompt)]
        summary_text = await llm_manager.generate_with_fallback(messages)
        return SourceSummary(summary=summary_text, key_quotes=[], relevance_score=0.8, main_topics=[], word_count=len(summary_text.split()))
    except Exception as e:
        logger.error(f"Source summarization failed: {e}")
        return SourceSummary(summary=f"Error: {e}", key_quotes=[], relevance_score=0.1, main_topics=[], word_count=0)