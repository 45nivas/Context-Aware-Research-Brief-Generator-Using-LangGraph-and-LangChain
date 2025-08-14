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
        
        # Check Google Gemini
        if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            self.available_services.append("gemini")
            logger.info("✅ Google Gemini available")
        else:
            logger.warning("❌ Google Gemini not available (missing API key or package)")
        
        # Check Ollama
        if OLLAMA_AVAILABLE:
            try:
                # Test Ollama connection
                import requests
                response = requests.get("http://localhost:11434/api/version", timeout=5)
                if response.status_code == 200:
                    self.available_services.append("ollama")
                    logger.info("✅ Ollama available")
                else:
                    logger.warning("❌ Ollama not running (start with: ollama serve)")
            except:
                logger.warning("❌ Ollama not available (install from https://ollama.ai/)")
        
        if not self.available_services:
            logger.error("❌ No LLM services available! Please set up Gemini API key or install Ollama")
    
    @property
    def primary_llm(self) -> Optional[BaseLanguageModel]:
        """Get the primary LLM (Google Gemini)."""
        if self._primary_llm is None and "gemini" in self.available_services:
            try:
                self._primary_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",  # Free tier model
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.1,
                    max_tokens=8192,
                    timeout=120,
                    max_retries=3
                )
                logger.info("Initialized Google Gemini Flash")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                return self.local_llm
        return self._primary_llm or self.local_llm
    
    @property
    def secondary_llm(self) -> Optional[BaseLanguageModel]:
        """Get the secondary LLM (Google Gemini Pro or Ollama)."""
        if self._secondary_llm is None and "gemini" in self.available_services:
            try:
                self._secondary_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",  # More capable model
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.3,
                    max_tokens=8192,
                    timeout=180,
                    max_retries=3
                )
                logger.info("Initialized Google Gemini Pro")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Pro: {e}")
                return self.local_llm
        return self._secondary_llm or self.local_llm
    
    @property
    def local_llm(self) -> Optional[BaseLanguageModel]:
        """Get the local Ollama LLM."""
        if self._local_llm is None and "ollama" in self.available_services:
            try:
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                local_model = os.getenv("LOCAL_MODEL", "llama3.1:8b")
                
                self._local_llm = Ollama(
                    model=local_model,
                    base_url=ollama_base_url,
                    temperature=0.1,
                    timeout=300,  # 5 minutes for local processing
                )
                logger.info(f"Initialized Ollama: {local_model}")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
        return self._local_llm
    
    async def generate_with_fallback(self, 
                                   messages: List[Union[HumanMessage, SystemMessage]], 
                                   prefer_local: bool = False) -> str:
        """Generate text with fallback between models."""
        models_to_try = []
        
        # Determine order based on preference and availability
        if prefer_local and self.local_llm:
            models_to_try = [self.local_llm, self.primary_llm, self.secondary_llm]
        else:
            models_to_try = [self.primary_llm, self.secondary_llm, self.local_llm]
        
        # Filter out None models
        models_to_try = [m for m in models_to_try if m is not None]
        
        if not models_to_try:
            raise Exception("No LLM models available. Please set up Gemini API key or install Ollama.")
        
        for i, model in enumerate(models_to_try):
            try:
                logger.info(f"Attempting generation with model {i+1}/{len(models_to_try)}")
                
                # Generate response
                response = await model.ainvoke(messages)
                
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                if content and content.strip():
                    logger.info(f"Successfully generated response with model {i+1}")
                    return content.strip()
                    
            except Exception as e:
                logger.warning(f"Model {i+1} failed: {e}")
                if i == len(models_to_try) - 1:  # Last model
                    raise Exception(f"All models failed. Last error: {e}")
                continue
        
        raise Exception("No models available for text generation")
    
    def reset_token_usage(self):
        """Reset token usage tracking. Stub implementation for compatibility."""
        # This is a stub method for compatibility with the original LLMManager
        # Free models don't have token tracking, so this is a no-op
        logger.debug("reset_token_usage called (no-op for free models)")
        pass


class FreeSearchManager:
    """Manages free search tools including DuckDuckGo and SerpAPI free tier."""
    
    def __init__(self):
        self.config = get_config()
        self._duckduckgo_search = None
        self._serpapi_search = None
        self._check_availability()
    
    def _check_availability(self):
        """Check which search services are available."""
        self.available_services = []
        
        # Check DuckDuckGo
        if DUCKDUCKGO_AVAILABLE:
            self.available_services.append("duckduckgo")
            logger.info("✅ DuckDuckGo search available")
        else:
            logger.warning("❌ DuckDuckGo search not available")
        
        # Check SerpAPI
        if SERPAPI_AVAILABLE and os.getenv("SERPAPI_API_KEY"):
            self.available_services.append("serpapi")
            logger.info("✅ SerpAPI available (free tier)")
        else:
            logger.warning("❌ SerpAPI not available")
        
        if not self.available_services:
            logger.error("❌ No search services available!")
    
    @property
    def duckduckgo_search(self) -> Optional[DDGS]:
        """Get DuckDuckGo search tool (completely free)."""
        if self._duckduckgo_search is None and "duckduckgo" in self.available_services:
            try:
                self._duckduckgo_search = DDGS()
                logger.info("Initialized DuckDuckGo search")
            except Exception as e:
                logger.error(f"Failed to initialize DuckDuckGo: {e}")
        return self._duckduckgo_search
    
    @property
    def serpapi_search(self) -> Optional[SerpAPIWrapper]:
        """Get SerpAPI search tool (free tier: 100 searches/month)."""
        if self._serpapi_search is None and "serpapi" in self.available_services:
            try:
                self._serpapi_search = SerpAPIWrapper(
                    serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
                    params={
                        "engine": "google",
                        "gl": "us",
                        "hl": "en",
                        "num": 5  # Limit for free tier
                    }
                )
                logger.info("Initialized SerpAPI search")
            except Exception as e:
                logger.error(f"Failed to initialize SerpAPI: {e}")
        return self._serpapi_search
    
    async def search_with_fallback(self, query: str) -> List[SearchResult]:
        """Search with fallback between available search tools."""
        search_results = []
        
        # Try SerpAPI first if available (better quality results)
        if self.serpapi_search:
            try:
                logger.info("Searching with SerpAPI")
                results = self.serpapi_search.run(query)
                search_results.extend(self._parse_serpapi_results(results, query))
                if search_results:
                    logger.info(f"SerpAPI returned {len(search_results)} results")
                    return search_results
            except Exception as e:
                logger.warning(f"SerpAPI search failed: {e}")
        
        # Fallback to DuckDuckGo (always available)
        if self.duckduckgo_search:
            try:
                logger.info("Searching with DuckDuckGo")
                results = list(self.duckduckgo_search.text(query, max_results=5))
                search_results.extend(self._parse_duckduckgo_results(results, query))
                logger.info(f"DuckDuckGo returned {len(search_results)} results")
                return search_results
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")
        
        # If no search works, return empty results with warning
        logger.warning("All search methods failed, returning empty results")
        return []
    
    def _parse_serpapi_results(self, results: str, query: str) -> List[SearchResult]:
        """Parse SerpAPI results."""
        search_results = []
        try:
            # SerpAPI returns JSON string
            data = json.loads(results) if isinstance(results, str) else results
            
            organic_results = data.get("organic_results", [])
            for result in organic_results[:5]:  # Limit for free tier
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    source="serpapi",
                    relevance_score=0.8,  # High relevance for SerpAPI
                    search_query=query,
                    timestamp=datetime.now()
                )
                search_results.append(search_result)
        except Exception as e:
            logger.error(f"Error parsing SerpAPI results: {e}")
        
        return search_results
    
    def _parse_duckduckgo_results(self, results: List[dict], query: str) -> List[SearchResult]:
        """Parse DuckDuckGo results from the new ddgs package format."""
        search_results = []
        try:
            # New ddgs package returns list of dictionaries like:
            # [{'title': '...', 'href': '...', 'body': '...'}, ...]
            
            for result in results[:5]:  # Limit results
                title = result.get('title', '')
                url = result.get('href', '')
                snippet = result.get('body', '')
                
                if title and url:
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet.strip(),
                        source="duckduckgo",
                        relevance_score=0.7,  # Good relevance for DDG
                        search_query=query,
                        timestamp=datetime.now()
                    )
                    search_results.append(search_result)
                        
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo results: {e}")
        
        return search_results


# Initialize global managers
llm_manager = FreeLLMManager()
search_manager = FreeSearchManager()


# Tool implementations
class FreeWebSearchTool(BaseTool):
    """Free web search tool using DuckDuckGo and SerpAPI free tier."""
    
    name = "free_web_search"
    description = "Search the web for current information using free search APIs"
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[SearchResult]:
        """Execute web search synchronously."""
        return asyncio.run(self._arun(query, run_manager))
    
    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[SearchResult]:
        """Execute web search asynchronously."""
        try:
            return await search_manager.search_with_fallback(query)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []


class FreeContentFetchTool(BaseTool):
    """Free content fetching tool for retrieving webpage content."""
    
    name = "free_content_fetch"
    description = "Fetch and extract content from web pages"
    
    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Fetch content synchronously."""
        return asyncio.run(self._arun(url, run_manager))
    
    async def _arun(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Fetch content asynchronously."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Basic content cleaning
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            
                            # Get text content
                            text = soup.get_text()
                            
                            # Clean up whitespace
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk)
                            
                            # Limit content length
                            max_content_length = 8000  # Smaller for free tier
                            if len(text) > max_content_length:
                                text = text[:max_content_length] + "..."
                            
                            return text
                        except ImportError:
                            # Fallback without BeautifulSoup
                            return content[:8000] + "..."
                    else:
                        logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return f"Failed to fetch content: HTTP {response.status}"
                        
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return f"Error fetching content: {str(e)}"


# Tool instances
web_search_tool = FreeWebSearchTool()
content_fetch_tool = FreeContentFetchTool()


# Helper functions
async def get_llm_for_task(task_type: str = "general") -> BaseLanguageModel:
    """Get the appropriate LLM for a specific task type."""
    if task_type in ["summarization", "extraction"]:
        # Use faster model for simple tasks
        return llm_manager.primary_llm
    elif task_type in ["analysis", "synthesis", "complex"]:
        # Use more capable model for complex tasks
        return llm_manager.secondary_llm or llm_manager.primary_llm
    elif task_type == "local":
        # Use local model when internet is not available or for privacy
        return llm_manager.local_llm or llm_manager.primary_llm
    else:
        return llm_manager.primary_llm


async def generate_text_with_fallback(messages: List[Union[HumanMessage, SystemMessage]], 
                                    task_type: str = "general",
                                    prefer_local: bool = False) -> str:
    """Generate text with automatic fallback between models."""
    try:
        return await llm_manager.generate_with_fallback(messages, prefer_local=prefer_local)
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise


async def search_web_free(query: str) -> List[SearchResult]:
    """Search the web using free search tools."""
    try:
        return await search_manager.search_with_fallback(query)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


async def fetch_content_free(url: str) -> str:
    """Fetch web content using free tools."""
    try:
        return await content_fetch_tool._arun(url)
    except Exception as e:
        logger.error(f"Content fetch failed: {e}")
        return ""


async def summarize_source_free(content: str, query: str) -> SourceSummary:
    """Summarize source content using free LLMs."""
    try:
        llm = await get_llm_for_task("summarization")
        
        prompt = f"""
        Please summarize the following content in relation to the query: "{query}"
        
        Focus on:
        1. Key information relevant to the query
        2. Important facts and statistics
        3. Notable quotes or statements
        4. Main conclusions or findings
        
        Content to summarize:
        {content[:4000]}...
        
        Provide a concise but comprehensive summary in 2-3 paragraphs.
        """
        
        messages = [HumanMessage(content=prompt)]
        summary = await llm_manager.generate_with_fallback(messages)
        
        # Extract key quotes (simple extraction)
        key_quotes = []
        lines = content.split('\n')
        for line in lines:
            if '"' in line and len(line.strip()) > 20 and len(line.strip()) < 200:
                key_quotes.append(line.strip())
            if len(key_quotes) >= 3:
                break
        
        return SourceSummary(
            summary=summary,
            key_quotes=key_quotes,
            relevance_score=0.8,
            main_topics=query.split()[:5],  # Simple topic extraction
            word_count=len(summary.split())
        )
        
    except Exception as e:
        logger.error(f"Source summarization failed: {e}")
        return SourceSummary(
            summary=f"Error summarizing content: {str(e)}",
            key_quotes=[],
            relevance_score=0.1,
            main_topics=[],
            word_count=0
        )


def check_free_services_status():
    """Check the status of all free services."""
    status = {
        "llm_services": [],
        "search_services": [],
        "recommendations": []
    }
    
    # Check LLM services
    if llm_manager.primary_llm:
        status["llm_services"].append("✅ Google Gemini (Primary)")
    if llm_manager.secondary_llm:
        status["llm_services"].append("✅ Google Gemini Pro (Secondary)")
    if llm_manager.local_llm:
        status["llm_services"].append("✅ Ollama (Local)")
    
    # Check search services
    if search_manager.duckduckgo_search:
        status["search_services"].append("✅ DuckDuckGo (Free)")
    if search_manager.serpapi_search:
        status["search_services"].append("✅ SerpAPI (Free Tier)")
    
    # Recommendations
    if not status["llm_services"]:
        status["recommendations"].append("🔧 Set up Google Gemini API key or install Ollama")
    if not status["search_services"]:
        status["recommendations"].append("🔧 Install duckduckgo-search package")
    
    return status


class ContentFetcher:
    """Fetches full content from web sources using free methods."""
    
    def __init__(self):
        """Initialize the content fetcher."""
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30.0),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
    
    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch full content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Content string or None if failed
        """
        try:
            await self._ensure_session()
            
            async with self.session.get(url) as response:
                response.raise_for_status()
                content = await response.text()
                
                # Basic HTML tag removal for text content
                import re
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                return content
                
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return None
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _arun(self, url: str) -> str:
        """LangChain compatible method."""
        content = await self.fetch_content(url)
        return content or ""


# Global instances for easy importing
llm_manager = FreeLLMManager()
search_manager = FreeSearchManager()
content_fetcher = ContentFetcher()
