"""
LangChain LLM implementations and tool integrations.
Provides abstractions for different model providers and search tools.
"""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler

from app.config import config
from app.models import SearchResult, SourceContent


class TokenUsageTracker(BaseCallbackHandler):
    """Callback handler to track token usage across LLM calls."""
    
    def __init__(self):
        self.token_usage = {}
        self.total_cost = 0.0
    
    def on_llm_end(self, response, **kwargs):
        """Track token usage when LLM call ends."""
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            model_name = response.llm_output.get('model_name', 'unknown')
            
            if model_name not in self.token_usage:
                self.token_usage[model_name] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            
            self.token_usage[model_name]['prompt_tokens'] += usage.get('prompt_tokens', 0)
            self.token_usage[model_name]['completion_tokens'] += usage.get('completion_tokens', 0)
            self.token_usage[model_name]['total_tokens'] += usage.get('total_tokens', 0)


class LLMManager:
    """Manages different LLM instances and their configurations."""
    
    def __init__(self):
        self.token_tracker = TokenUsageTracker()
        
        # Initialize OpenAI models
        self.primary_llm = ChatOpenAI(
            model=config.primary_model.model_name,
            temperature=config.primary_model.temperature,
            max_tokens=config.primary_model.max_tokens,
            openai_api_key=config.primary_model.api_key,
            callbacks=[self.token_tracker]
        )
        
        # Initialize Anthropic model
        self.secondary_llm = ChatAnthropic(
            model=config.secondary_model.model_name,
            temperature=config.secondary_model.temperature,
            max_tokens=config.secondary_model.max_tokens,
            anthropic_api_key=config.secondary_model.api_key,
            callbacks=[self.token_tracker]
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model.model_name,
            openai_api_key=config.embedding_model.api_key
        )
    
    def get_llm(self, model_type: str):
        """
        Get LLM instance by type.
        
        Args:
            model_type: 'primary' or 'secondary'
            
        Returns:
            LLM instance
        """
        if model_type == "primary":
            return self.primary_llm
        elif model_type == "secondary":
            return self.secondary_llm
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_token_usage(self) -> Dict[str, Dict[str, int]]:
        """Get current token usage statistics."""
        return self.token_tracker.token_usage.copy()
    
    def reset_token_usage(self):
        """Reset token usage statistics."""
        self.token_tracker.token_usage = {}
        self.token_tracker.total_cost = 0.0


class SearchManager:
    """Manages search tools and operations."""
    
    def __init__(self):
        # Initialize Tavily search
        self.tavily_search = TavilySearchResults(
            api_key=config.search.tavily_api_key,
            max_results=config.search.max_results
        )
        
        # Initialize SerpAPI
        self.serp_search = SerpAPIWrapper(
            serpapi_api_key=config.search.serpapi_api_key
        )
    
    async def search_tavily(self, query: str, max_results: int = None) -> List[SearchResult]:
        """
        Search using Tavily API.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        max_results = max_results or config.search.max_results
        
        try:
            # Run search in thread pool since Tavily is synchronous
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.tavily_search.run(query)
            )
            
            search_results = []
            for i, result in enumerate(results[:max_results]):
                search_results.append(SearchResult(
                    title=result.get('title', 'No title'),
                    url=result.get('url', ''),
                    snippet=result.get('content', result.get('snippet', '')),
                    relevance_score=max(0.1, 1.0 - (i * 0.1))  # Simple relevance scoring
                ))
            
            return search_results
        
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    async def search_serp(self, query: str, max_results: int = None) -> List[SearchResult]:
        """
        Search using SerpAPI.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        max_results = max_results or config.search.max_results
        
        try:
            # Run search in thread pool since SerpAPI is synchronous
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.serp_search.run(query)
            )
            
            search_results = []
            if isinstance(results, str):
                # Parse the results if they come as a string
                import json
                try:
                    results_data = json.loads(results)
                    organic_results = results_data.get('organic', [])
                except:
                    return []
            else:
                organic_results = results.get('organic', []) if isinstance(results, dict) else []
            
            for i, result in enumerate(organic_results[:max_results]):
                search_results.append(SearchResult(
                    title=result.get('title', 'No title'),
                    url=result.get('link', ''),
                    snippet=result.get('snippet', ''),
                    relevance_score=max(0.1, 1.0 - (i * 0.1))
                ))
            
            return search_results
        
        except Exception as e:
            print(f"SerpAPI search error: {e}")
            return []
    
    async def combined_search(self, query: str, max_results: int = None) -> List[SearchResult]:
        """
        Perform combined search using multiple providers.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            Combined and deduplicated search results
        """
        max_results = max_results or config.search.max_results
        
        # Run searches concurrently
        tavily_task = self.search_tavily(query, max_results // 2)
        serp_task = self.search_serp(query, max_results // 2)
        
        tavily_results, serp_results = await asyncio.gather(
            tavily_task, serp_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(tavily_results, Exception):
            tavily_results = []
        if isinstance(serp_results, Exception):
            serp_results = []
        
        # Combine and deduplicate results
        all_results = tavily_results + serp_results
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Sort by relevance score and return top results
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return unique_results[:max_results]


class ContentFetcher:
    """Fetches full content from web sources."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
    
    async def fetch_content(self, url: str) -> Optional[SourceContent]:
        """
        Fetch full content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            SourceContent object or None if failed
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Simple text extraction (in production, use proper HTML parsing)
            content = response.text
            
            # Basic HTML tag removal for text content
            import re
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Limit content length
            max_length = config.node_configs["content_fetching"]["max_content_length"]
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            return SourceContent(
                url=url,
                title=self._extract_title(response.text),
                content=content,
                fetch_timestamp=datetime.utcnow(),
                content_length=len(content)
            )
        
        except Exception as e:
            print(f"Failed to fetch content from {url}: {e}")
            return None
    
    def _extract_title(self, html: str) -> str:
        """Extract title from HTML content."""
        import re
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        return title_match.group(1).strip() if title_match else "Untitled"
    
    async def fetch_multiple(self, urls: List[str]) -> List[SourceContent]:
        """
        Fetch content from multiple URLs concurrently.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            List of SourceContent objects (successful fetches only)
        """
        max_concurrent = config.node_configs["content_fetching"]["concurrent_fetches"]
        
        # Process URLs in batches to avoid overwhelming servers
        results = []
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i:i + max_concurrent]
            batch_tasks = [self.fetch_content(url) for url in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, SourceContent):
                    results.append(result)
        
        return results
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Global instances
llm_manager = LLMManager()
search_manager = SearchManager()
content_fetcher = ContentFetcher()
