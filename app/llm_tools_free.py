"""
FREE LLM and search tools for the Research Brief Generator.

This module provides integrations with FREE alternatives:
- Google Gemini (free tier: 15 requests/minute, 1 million tokens/month)
- Ollama (local/free: unlimited usage)
- DuckDuckGo Search (completely free)
- SerpAPI (free tier: 100 searches/month)
"""

import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import logging

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("❌ OpenAI not available. Install with: pip install langchain-openai")

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
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from .config import get_config
from .models import SearchResult, SourceSummary

logger = logging.getLogger(__name__)

class FreeLLMManager:
    """Manages free LLM models including Gemini and Ollama."""
    
    def __init__(self):
        self.config = get_config()
        self._primary_llm = None
        self._secondary_llm = None
        self._local_llm = None
        self._check_availability()
    
    def _check_availability(self):
        """Check which LLM services are available."""
        self.available_services = []

        if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            self.available_services.append("gemini")
            logger.info("✅ Google Gemini available")
        else:
            logger.warning("❌ Google Gemini not available (missing API key or package)")

        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.available_services.append("openai")
            logger.info("✅ OpenAI available")
        else:
            logger.warning("❌ OpenAI not available (missing API key or package)")

        if OLLAMA_AVAILABLE and not os.getenv("RENDER"):
            try:
                import requests
                response = requests.get("http://localhost:11434/api/version", timeout=5)
                if response.status_code == 200:
                    self.available_services.append("ollama")
                    logger.info("✅ Ollama available")
            except:
                logger.warning("❌ Ollama not available")

        if not self.available_services:
            logger.error("❌ No LLM services available! Please set up Gemini, OpenAI, or Ollama")
    
    @property
    def primary_llm(self) -> Optional[BaseLanguageModel]:
        """Get the primary LLM (Google Gemini)."""
        if self._primary_llm is None and "gemini" in self.available_services:
            try:
                self._primary_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.1,
                    max_tokens=8192,
                    timeout=120,
                    max_retries=3
                )
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
        return self._primary_llm

    @property
    def secondary_llm(self) -> Optional[BaseLanguageModel]:
        """Get the secondary LLM (OpenAI)."""
        if self._secondary_llm is None and "openai" in self.available_services:
            try:
                self._secondary_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0.2,
                    max_tokens=4096,
                    timeout=120,
                    max_retries=3
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
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
                    timeout=300,
                )
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
        return self._local_llm
    
    async def generate_with_fallback(self, messages: List[Union[HumanMessage, SystemMessage]], prefer_local: bool = False) -> str:
        """Generate text with fallback between models."""
        models_to_try = []
        if prefer_local and self.local_llm: models_to_try.append(self.local_llm)
        if self.primary_llm: models_to_try.append(self.primary_llm)
        if self.secondary_llm: models_to_try.append(self.secondary_llm)
        if self.local_llm and not prefer_local: models_to_try.append(self.local_llm)

        models_to_try = [m for m in models_to_try if m is not None]
        if not models_to_try:
            raise Exception("No LLM models available.")

        for model in models_to_try:
            try:
                response = await model.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                if content and content.strip():
                    return content.strip()
            except Exception as e:
                logger.warning(f"Model {type(model).__name__} failed: {e}")
        raise Exception("All available models failed to generate a response.")
    
    def reset_token_usage(self): pass
    def get_token_usage(self): return {}

class FreeSearchManager:
    """Manages free search tools."""
    
    def __init__(self):
        self.config = get_config()
        self._duckduckgo_search = None
        self._serpapi_search = None
        self._check_availability()
    
    def _check_availability(self):
        self.available_services = []
        if DUCKDUCKGO_AVAILABLE: self.available_services.append("duckduckgo")
        if SERPAPI_AVAILABLE and os.getenv("SERPAPI_API_KEY"): self.available_services.append("serpapi")
    
    @property
    def duckduckgo_search(self) -> Optional[DDGS]:
        if self._duckduckgo_search is None and "duckduckgo" in self.available_services:
            self._duckduckgo_search = DDGS()
        return self._duckduckgo_search
    
    @property
    def serpapi_search(self) -> Optional[SerpAPIWrapper]:
        if self._serpapi_search is None and "serpapi" in self.available_services:
            self._serpapi_search = SerpAPIWrapper()
        return self._serpapi_search
    
    async def search_with_fallback(self, query: str) -> List[SearchResult]:
        search_results = []
        if self.serpapi_search:
            try:
                results = await asyncio.to_thread(self.serpapi_search.run, query)
                search_results.extend(self._parse_serpapi_results(results, query))
                if search_results: return search_results
            except Exception as e:
                logger.warning(f"SerpAPI search failed: {e}")
        
        if self.duckduckgo_search:
            try:
                # FIX: Use the synchronous `text` method in a separate thread as `atext` might not exist
                results = await asyncio.to_thread(self.duckduckgo_search.text, query, max_results=5)
                search_results.extend(self._parse_duckduckgo_results(results, query))
                return search_results
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")
        
        return []
    
    def _parse_serpapi_results(self, results: Any, query: str) -> List[SearchResult]:
        search_results = []
        try:
            # FIX: Robustly handle different possible return types from SerpAPI
            data_to_parse = None
            if isinstance(results, str):
                if results.strip().startswith('{'):
                    data_to_parse = json.loads(results).get("organic_results", [])
                elif results.strip().startswith('['):
                    try:
                        data_to_parse = eval(results)
                    except:
                        logger.warning(f"SerpAPI returned a string that could not be evaluated: {results}")
                        return []
            elif isinstance(results, dict):
                data_to_parse = results.get("organic_results", [])
            elif isinstance(results, list):
                data_to_parse = results

            if not data_to_parse:
                return []

            for item in data_to_parse[:5]:
                if isinstance(item, dict):
                    search_results.append(SearchResult(
                        title=item.get("title", ""), url=item.get("link", ""),
                        snippet=item.get("snippet", ""), source="serpapi",
                        relevance_score=0.8, search_query=query, timestamp=datetime.now()
                    ))
                else:
                    search_results.append(SearchResult(
                        title=query, url="", snippet=str(item), source="serpapi",
                        relevance_score=0.7, search_query=query, timestamp=datetime.now()
                    ))
        except Exception as e:
            logger.error(f"Error parsing SerpAPI results: {e}")
        return search_results
    
    def _parse_duckduckgo_results(self, results: List[dict], query: str) -> List[SearchResult]:
        search_results = []
        try:
            for result in results[:5]:
                if result.get('title') and result.get('href'):
                    search_results.append(SearchResult(
                        title=result.get('title', ''), url=result.get('href', ''),
                        snippet=result.get('body', '').strip(), source="duckduckgo",
                        relevance_score=0.7, search_query=query, timestamp=datetime.now()
                    ))
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo results: {e}")
        return search_results

class ContentFetcher:
    """Fetches full content from web sources."""
    def __init__(self):
        self.session = None
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30.0),
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
    
    async def fetch_content(self, url: str) -> Optional[str]:
        try:
            await self._ensure_session()
            async with self.session.get(url) as response:
                response.raise_for_status()
                content = await response.text()
                import re
                content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
                content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                return content
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return None
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _arun(self, url: str) -> str:
        content = await self.fetch_content(url)
        return content or ""

llm_manager = FreeLLMManager()
search_manager = FreeSearchManager()
content_fetcher = ContentFetcher()

async def generate_text_with_fallback(messages: List[Union[HumanMessage, SystemMessage]], **kwargs) -> str:
    return await llm_manager.generate_with_fallback(messages)

async def search_web_free(query: str) -> List[SearchResult]:
    return await search_manager.search_with_fallback(query)

async def fetch_content_free(url: str) -> str:
    return await content_fetcher._arun(url)

async def summarize_source_free(content: str, query: str) -> SourceSummary:
    try:
        llm = llm_manager.primary_llm
        prompt = f"Summarize the following content in relation to the query: '{query}'\n\nContent:\n{content[:4000]}..."
        messages = [HumanMessage(content=prompt)]
        summary = await llm_manager.generate_with_fallback(messages)
        return SourceSummary(
            summary=summary, key_quotes=[], relevance_score=0.8,
            main_topics=query.split()[:5], word_count=len(summary.split())
        )
    except Exception as e:
        logger.error(f"Source summarization failed: {e}")
        return SourceSummary(summary=f"Error: {e}", key_quotes=[], relevance_score=0.1, main_topics=[], word_count=0)
