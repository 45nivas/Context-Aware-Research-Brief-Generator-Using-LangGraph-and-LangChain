from dotenv import load_dotenv
load_dotenv()
"""
FREE LLM and search tools for the Research Brief Generator.

This module provides integrations with FREE alternatives:
- OpenRouter (via a custom wrapper)
- Google Gemini (free tier)
- Ollama (local/free)
- DuckDuckGo Search (completely free)
- SerpAPI (free tier)
"""

import os
import asyncio
import aiohttp
import httpx
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import logging
import re

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# --- Check for LangChain package availability ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("❌ Google GenAI not available. Install with: pip install langchain-google-genai")

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("❌ Ollama not available. Install with: pip install langchain-community")

try:
    from ddgs import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    print("❌ DuckDuckGo search not available. Install with: pip install ddgs")

try:
    from langchain_community.utilities import SerpAPIWrapper
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    print("⚠️ SerpAPI not available. Install with: pip install google-search-results")

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

# --- Custom OpenRouter Class ---
class OpenRouterLLM:
    """Custom LLM wrapper for OpenRouter API."""
    def __init__(self, api_key: str, model: str = "openrouter/openchat-3.5-0106", temperature: float = 0.1, max_tokens: int = 8192, timeout: int = 120):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def ainvoke(self, messages: List[Union[HumanMessage, SystemMessage]]) -> str:
        """Asynchronously invokes the OpenRouter API."""
        prompt = "\n".join([m.content for m in messages])
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Custom OpenRouter API call failed: {e}")
            return f"Error: {e}"

# --- LLM and Search Managers ---
class FreeLLMManager:
    """Manages free LLM models including OpenRouter (custom), Gemini, and Ollama."""
    
    def __init__(self):
        self._primary_llm = None
        self._secondary_llm = None
        self._local_llm = None
        self._check_availability()
    
    def _check_availability(self):
        """Check which LLM services are available."""
        self.available_services = []

        # Debug: Print loaded environment variables for LLMs
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        logger.info(f"[DEBUG] OPENROUTER_API_KEY: {openrouter_key}")
        logger.info(f"[DEBUG] GOOGLE_API_KEY: {google_key}")

        if openrouter_key:
            self.available_services.append("openrouter")
            logger.info("✅ OpenRouter available (using custom wrapper)")
        else:
            logger.warning("❌ OpenRouter not available (missing OPENROUTER_API_KEY)")

        if GOOGLE_AVAILABLE and google_key:
            self.available_services.append("gemini")
            logger.info("✅ Google Gemini available")
        else:
            logger.warning("❌ Google Gemini not available (missing API key or package)")

        if OLLAMA_AVAILABLE and not os.getenv("RENDER"): # RENDER env var used to detect cloud deployment
            try:
                import requests
                # A simple check to see if the Ollama server is running
                response = requests.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), timeout=3)
                if response.status_code == 200:
                    self.available_services.append("ollama")
                    logger.info("✅ Ollama available")
            except Exception:
                logger.warning("❌ Ollama not available (could not connect to server)")

        if not self.available_services:
            logger.error("❌ No LLM services available! Please set up OpenRouter, Gemini, or Ollama.")
    
    @property
    def primary_llm(self) -> Optional[OpenRouterLLM]:
        """Get the primary LLM (OpenRouter custom wrapper)."""
        if self._primary_llm is None and "openrouter" in self.available_services:
            try:
                self._primary_llm = OpenRouterLLM(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    model=os.getenv("OPENROUTER_MODEL", "openrouter/openchat-3.5-0106"),
                )
            except Exception as e:
                logger.error(f"Failed to initialize custom OpenRouterLLM: {e}")
        return self._primary_llm

    @property
    def secondary_llm(self) -> Optional[BaseLanguageModel]:
        """Get the secondary LLM (Google Gemini)."""
        if self._secondary_llm is None and "gemini" in self.available_services:
            try:
                self._secondary_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.1,
                    max_tokens=8192,
                    timeout=120,
                    max_retries=3
                )
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
        return self._secondary_llm

    @property
    def local_llm(self) -> Optional[BaseLanguageModel]:
        """Get the local Ollama LLM (local only)."""
        if self._local_llm is None and "ollama" in self.available_services:
            try:
                self._local_llm = Ollama(
                    model=os.getenv("LOCAL_MODEL", "llama3.1:8b"),
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    temperature=0.1,
                    timeout=300
                )
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
        return self._local_llm
    
    async def generate_with_fallback(self, messages: List[Union[HumanMessage, SystemMessage]]) -> str:
        """Generate text with fallback between available models."""
        models_to_try = [m for m in [self.primary_llm, self.secondary_llm, self.local_llm] if m is not None]

        if not models_to_try:
            raise Exception("No LLM models available to try.")

        for model in models_to_try:
            try:
                response = await model.ainvoke(messages)
                # Handle both LangChain's BaseMessage content and our custom str response
                content = response.content if hasattr(response, 'content') else str(response)
                if content and content.strip() and not content.startswith("Error:"):
                    return content.strip()
            except Exception as e:
                logger.warning(f"Model {type(model).__name__} failed: {e}")
        raise Exception("All available models failed to generate a valid response.")

class FreeSearchManager:
    """Manages free search tools with fallback."""
    
    def __init__(self):
        self.available_services = []
        if DUCKDUCKGO_AVAILABLE:
            self.available_services.append("duckduckgo")
            logger.info("✅ DuckDuckGo search available.")
        if SERPAPI_AVAILABLE and os.getenv("SERPAPI_API_KEY"):
            self.available_services.append("serpapi")
            logger.info("✅ SerpAPI search available.")

    async def search_with_fallback(self, query: str, max_results: int = 5) -> List[dict]:
        """Search with fallback from SerpAPI to DuckDuckGo."""
        if "serpapi" in self.available_services:
            try:
                serpapi_wrapper = SerpAPIWrapper()
                results = await asyncio.to_thread(serpapi_wrapper.run, query)
                # Basic parsing, assuming results is a string that can be evaluated or a list
                if isinstance(results, str):
                    results = eval(results)
                if isinstance(results, list) and results:
                    return [{"title": r.get("title"), "href": r.get("link"), "body": r.get("snippet")} for r in results[:max_results]]
            except Exception as e:
                logger.warning(f"SerpAPI search failed: {e}. Falling back to DuckDuckGo.")

        if "duckduckgo" in self.available_services:
            try:
                async with DDGS() as ddgs:
                    results = [r async for r in ddgs.text(query, max_results=max_results)]
                    return results
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")
        
        return []

class ContentFetcher:
    """Fetches and cleans content from web URLs."""
    
    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetches and cleans HTML content from a URL."""
        try:
            async with httpx.AsyncClient(timeout=30.0, headers={'User-Agent': 'Mozilla/5.0'}) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.text
                # Basic cleaning
                content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
                content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                return content
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return None

    async def close(self):
        """A placeholder for cleanup, currently does nothing."""
        # If you were using a persistent session, you would close it here.
        # For now, this method just needs to exist.
        pass

# --- Instantiated Managers and Helper Functions ---
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
# You need to add this missing function to your file

from .models import SourceSummary # Make sure this import is at the top of your file

async def summarize_source_free(content: str, query: str) -> SourceSummary:
    """
    Summarizes the given content in relation to a query using the fallback LLM.
    """
    try:
        prompt = f"Summarize the following content in the context of the research query: '{query}'\n\nContent:\n{content[:8000]}"
        messages = [HumanMessage(content=prompt)]
        
        # Use the main fallback generator to get the summary
        summary_text = await llm_manager.generate_with_fallback(messages)
        
        return SourceSummary(
            summary=summary_text,
            key_quotes=[], # You can enhance this later if needed
            relevance_score=0.8,
            main_topics=query.split()[:5],
            word_count=len(summary_text.split())
        )
    except Exception as e:
        logger.error(f"Source summarization failed: {e}")
        return SourceSummary(
            summary=f"Error during summarization: {e}",
            key_quotes=[],
            relevance_score=0.1,
            main_topics=[],
            word_count=0
        )
    
    # ...existing code...