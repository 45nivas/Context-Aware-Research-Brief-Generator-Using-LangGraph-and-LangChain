"""
FREE LLM and search tools for the Research Brief Generator (fixed + Tavily integrated).
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
    ChatGoogleGenerativeAI = None

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = None

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from app.models import SourceSummary

logger = logging.getLogger(__name__)

# -------------------------
# OpenRouter Wrapper
# -------------------------
class OpenRouterLLM:
    """OpenRouter LLM implementation compatible with LangChain."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    async def ainvoke(self, messages: List[Union[HumanMessage, SystemMessage]]) -> object:
        """Async invocation compatible with LangChain interface."""
        try:
            # Convert messages to prompt
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}")
                elif isinstance(msg, HumanMessage):
                    prompt_parts.append(f"Human: {msg.content}")
                else:
                    prompt_parts.append(str(msg.content))
            
            prompt = "\n".join(prompt_parts)
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Return object with content attribute for compatibility
                class LLMResponse:
                    def __init__(self, content):
                        self.content = content
                
                return LLMResponse(content)
                
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            class ErrorResponse:
                def __init__(self, error):
                    self.content = f"Error: {error}"
            return ErrorResponse(str(e))

    @property
    def _llm_type(self) -> str:
        return "openrouter"


# -------------------------
# Free LLM Manager
# -------------------------
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
            self._primary_llm = OpenRouterLLM(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
            )
        return self._primary_llm

    @property
    def secondary_llm(self) -> Optional[BaseLanguageModel]:
        if self._secondary_llm is None and "gemini" in self.available_services:
            self._secondary_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        return self._secondary_llm
    
    async def generate_with_fallback(self, messages: List[Union[HumanMessage, SystemMessage]]) -> str:
        prompt_tokens = sum(len(m.content) for m in messages) // 4 
        models_to_try = [m for m in [self.primary_llm, self.secondary_llm] if m is not None]
        if not models_to_try:
            raise Exception("No LLM models are available.")
        for model in models_to_try:
            try:
                response = await model.ainvoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                if content and "Error:" not in content:
                    completion_tokens = len(content) // 4
                    self.token_counts["total_tokens"] = self.token_counts.get("total_tokens", 0) + prompt_tokens + completion_tokens
                    return content.strip()
            except Exception as e:
                logger.warning(f"Model {type(model).__name__} failed: {e}", exc_info=True)
        raise Exception("All available LLM models failed.")


# -------------------------
# Free Search Manager (Tavily first, fallback to Bing/Wikipedia)
# -------------------------
class FreeSearchManager:
    """Manages free search tools using Tavily first, fallback to Wikipedia/Bing scraping."""
    
    def __init__(self):
        self.tavily_client = None
        if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
            try:
                self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily client: {e}")
    
    async def _resolve_redirect_url(self, client: httpx.AsyncClient, url: str) -> str:
        """Resolve redirect URLs to get the actual destination."""
        try:
            # For Bing redirect URLs, try to extract the actual URL from the parameters
            if 'bing.com/ck/a' in url and 'u=a1' in url:
                # Extract the base64-encoded URL
                import urllib.parse
                parsed = urllib.parse.urlparse(url)
                params = urllib.parse.parse_qs(parsed.query)
                if 'u' in params:
                    encoded_url = params['u'][0]
                    if encoded_url.startswith('a1'):
                        # Remove 'a1' prefix and decode
                        try:
                            import base64
                            decoded_bytes = base64.b64decode(encoded_url[2:] + '==')  # Add padding
                            decoded_url = decoded_bytes.decode('utf-8')
                            return decoded_url
                        except:
                            pass
            
            # For other redirects, make a HEAD request to follow redirects
            try:
                response = await client.head(url, follow_redirects=True, timeout=5.0)
                return str(response.url)
            except:
                # If HEAD fails, try GET with early termination
                try:
                    response = await client.get(url, follow_redirects=True, timeout=3.0)
                    return str(response.url)
                except:
                    pass
        except Exception:
            pass
        
        # Return original URL if resolution fails
        return url
    
    async def search_with_fallback(self, query: str, max_results: int = 5) -> List[dict]:
        # ✅ First try Tavily with enhanced academic focus
        if self.tavily_client:
            try:
                logger.info("🔎 Using Tavily for search...")
                
                # Create academic-focused query that excludes marketing sites
                academic_query = f'"{query}" (site:edu OR site:org OR "research paper" OR "academic study" OR "journal article") -site:impact.com -marketing -advertising -affiliate'
                
                results = self.tavily_client.search(
                    query=academic_query, 
                    max_results=max_results,
                    search_depth="advanced",
                    include_domains=["edu", "org", "gov", "scholar.google.com", "researchgate.net", "arxiv.org"],
                    exclude_domains=["impact.com", "marketing.com", "advertising.com", "affiliate.com"]
                )
                
                normalized = []
                for r in results.get('results', []):
                    url = r.get("url", "")
                    # Double-check exclusions
                    if not any(blocked in url.lower() for blocked in ['impact.com', 'marketing', 'advertising', 'affiliate']):
                        normalized.append({
                            "title": r.get("title", "Untitled Result"),
                            "href": url,
                            "body": r.get("content", "")
                        })
                
                if normalized:
                    logger.info(f"✅ Tavily found {len(normalized)} academic sources")
                    return normalized
                    
            except Exception as e:
                logger.error(f"Tavily search failed: {e}")

        # ⚠️ Enhanced fallback strategy for better academic content
        return await self._enhanced_academic_search(query, max_results)

    async def _enhanced_academic_search(self, query: str, max_results: int) -> List[dict]:

        # ⚠️ Fallback to Wikipedia API parsing
        logger.warning("⚠️ Tavily unavailable — falling back to Wikipedia API.")
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&limit={max_results}&format=json&search={query}"
        results = []
        async with httpx.AsyncClient(timeout=20) as client:
            try:
                resp = await client.get(wiki_url)
                if resp.status_code == 200:
                    data = resp.json()
                    titles = data[1] if len(data) > 1 else []
                    urls = data[3] if len(data) > 3 else []
                    for title, url in zip(titles, urls):
                        results.append({
                            "title": title,
                            "href": url,
                            "body": f"Wikipedia result for {title}"
                        })
                if results:
                    return results
            except Exception as e:
                logger.error(f"Wikipedia API fallback failed: {e}")

            # ⚠️ Fallback to multiple academic search strategies
            logger.warning("⚠️ Wikipedia unavailable — trying academic search engines.")
            
            # Try multiple academic-focused search strategies with strict exclusions
            academic_strategies = [
                # Strategy 1: Direct academic site search with exclusions
                f'site:edu {query} OR site:org {query} -site:impact.com -marketing -advertising',
                # Strategy 2: Open access academic sources
                f'site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov OR site:doaj.org {query} -impact.com',
                # Strategy 3: Educational resource databases  
                f'site:eric.ed.gov OR site:scholar.google.com {query} -impact.com -affiliate',
                # Strategy 4: Research paper search with negative keywords
                f'"{query}" filetype:pdf "research" OR "study" OR "analysis" -impact.com -marketing -advertising',
                # Strategy 5: Enhanced academic query with strict filtering
                f'"{query}" (site:edu OR site:org OR "research paper" OR "academic study" OR "journal article") -impact.com -marketing -advertising -affiliate -influencer'
            ]
            
            for strategy_index, academic_query in enumerate(academic_strategies):
                try:
                    bing_url = f"https://www.bing.com/search?q={academic_query}"
                    resp = await client.get(bing_url, follow_redirects=True)
                    
                    if resp.status_code == 200:
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(resp.text, 'html.parser')
                            
                            # Enhanced academic domain detection for open access sources
                            academic_domains = [
                                '.edu', '.org', 'scholar.google', 'researchgate', 'jstor', 'pubmed', 
                                'arxiv', 'ieee', 'acm.org', 'springer', 'sciencedirect', 'nature.com',
                                'wiley.com', 'tandfonline.com', 'academia.edu', 'semanticscholar.org',
                                'eric.ed.gov', 'doaj.org', 'plos.org', 'frontiersin.org', 'mdpi.com',
                                'hindawi.com', 'bmcmedcentral.com', 'nih.gov', 'unesco.org'
                            ]
                            
                            # Domains to exclude as irrelevant or problematic
                            excluded_domains = [
                                'ecourts.gov.in', 'services.ecourts.gov.in', 'hcservices.ecourts.gov.in',
                                'dictionary.com', 'thefreedictionary.com', 'vocabulary.com',
                                'wordhippo.com', 'synonyms.com', 'definitions.net',
                                'shopping.com', 'amazon.com', 'ebay.com', 'alibaba.com',
                                'impact.com', 'partners.impact.com', 'help.impact.com',
                                'marketing.com', 'affiliate', 'influencer', 'advertising',
                                'linkedin.com/pulse', 'medium.com/@', 'quora.com'
                            ]
                            
                            academic_results = []
                            other_results = []
                            
                            for item in soup.find_all('li', class_='b_algo'):
                                a_tag = item.find('a')
                                snippet_tag = item.find('p')
                                if a_tag and snippet_tag:
                                    href = a_tag.get('href', '')
                                    title = a_tag.text
                                    body = snippet_tag.text
                                    
                                    # Resolve redirects to get actual URL for domain checking
                                    final_url = await self._resolve_redirect_url(client, href)
                                    
                                    # Skip URLs from excluded domains (irrelevant/problematic sources)
                                    if any(domain in final_url.lower() for domain in excluded_domains):
                                        logger.info(f"Skipping excluded domain: {final_url}")
                                        continue
                                    
                                    # Additional strict filtering for irrelevant content
                                    if any(blocked in final_url.lower() for blocked in [
                                        'dictionary', 'definition', 'thesaurus', 'vocabulary',
                                        'court', 'legal', 'jupiter', 'planet', 'shopping',
                                        'grammarly.com', 'doubleclick.net', 'ad.', '/ads/',
                                        'impact.com', 'marketing', 'affiliate', 'influencer',
                                        'advertising', 'promotion', 'campaign'
                                    ]):
                                        logger.info(f"Skipping irrelevant URL: {final_url}")
                                        continue
                                    
                                    # Check if resolved URL is from an academic domain
                                    is_academic = any(domain in final_url.lower() for domain in academic_domains)
                                    
                                    # Enhanced relevance check with topic-specific validation
                                    text_content = f"{title} {body}".lower()
                                    query_lower = query.lower()
                                    
                                    # Specific relevance scoring for different topics
                                    relevance_score = 0
                                    topic_match = False
                                    
                                    # AI Education specific validation
                                    if 'ai' in query_lower and 'education' in query_lower:
                                        education_keywords = ['education', 'learning', 'teaching', 'student', 'school', 'classroom', 'curriculum', 'pedagogy', 'instructional']
                                        ai_keywords = ['artificial intelligence', 'ai', 'machine learning', 'deep learning', 'neural network', 'intelligent tutoring', 'adaptive learning']
                                        
                                        education_matches = sum(1 for kw in education_keywords if kw in text_content)
                                        ai_matches = sum(1 for kw in ai_keywords if kw in text_content)
                                        
                                        # Require both AI and education terms for relevance
                                        if education_matches >= 1 and ai_matches >= 1:
                                            relevance_score = education_matches + ai_matches
                                            topic_match = True
                                        
                                        # Block obviously irrelevant content
                                        irrelevant_terms = ['jupiter', 'planet', 'space', 'astronomy', 'machine tool', 'manufacturing', 'court', 'legal']
                                        if any(term in text_content for term in irrelevant_terms):
                                            logger.info(f"Blocking irrelevant content: {title[:50]}...")
                                            continue
                                    
                                    # General AI research validation (when not education-specific)
                                    elif 'artificial intelligence' in query_lower or 'machine learning' in query_lower:
                                        ai_research_keywords = ['research', 'algorithm', 'model', 'training', 'neural', 'data', 'analysis', 'performance', 'accuracy']
                                        ai_keywords = ['artificial intelligence', 'ai', 'machine learning', 'deep learning', 'neural network']
                                        
                                        research_matches = sum(1 for kw in ai_research_keywords if kw in text_content)
                                        ai_matches = sum(1 for kw in ai_keywords if kw in text_content)
                                        
                                        if ai_matches >= 1 and research_matches >= 1:
                                            relevance_score = ai_matches + research_matches
                                            topic_match = True
                                    
                                    # Generic topic validation (fallback)
                                    else:
                                        # Extract key terms from query for basic relevance
                                        query_terms = [term.strip('"') for term in query_lower.split() if len(term.strip('"')) > 3]
                                        term_matches = sum(1 for term in query_terms if term in text_content)
                                        
                                        if term_matches >= 2:
                                            relevance_score = term_matches
                                            topic_match = True
                                    
                                    # Skip if no topic relevance found
                                    if not topic_match:
                                        logger.info(f"Skipping non-relevant content: {title[:50]}...")
                                        continue
                                    
                                    # Also check for academic keywords in title and body
                                    academic_keywords = ['research', 'study', 'analysis', 'journal', 'university', 'academic', 'scholar']
                                    has_academic_keywords = sum(1 for kw in academic_keywords if kw in text_content) >= 2
                                    
                                    result = {
                                        "title": title,
                                        "href": final_url,  # Use resolved URL
                                        "body": body,
                                        "relevance_score": relevance_score
                                    }
                                    
                                    if is_academic or has_academic_keywords:
                                        academic_results.append(result)
                                    else:
                                        other_results.append(result)
                            
                            # If we found academic results, prioritize them
                            if academic_results:
                                results = academic_results[:max_results]
                                if len(results) < max_results:
                                    results.extend(other_results[:max_results - len(results)])
                                logger.info(f"Strategy {strategy_index + 1}: Found {len(academic_results)} academic sources, {len(other_results)} other sources")
                                return results
                                
                        except ImportError:
                            logger.warning("BeautifulSoup not available for HTML parsing")
                except Exception as e:
                    logger.warning(f"Academic search strategy {strategy_index + 1} failed: {e}")
                    continue
            
            # If no academic search strategies worked, fall back to DuckDuckGo

            # ⚠️ Final fallback: DuckDuckGo HTML scraping
            logger.warning("⚠️ Bing unavailable — falling back to DuckDuckGo HTML scraping.")
            ddg_url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                resp = await client.post(ddg_url, data=params, headers=headers)
                if resp.status_code == 200:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        for link in soup.find_all('a', class_='result__a'):
                            snippet_tag = link.find_next('a', class_='result__snippet')
                            results.append({
                                "title": link.text,
                                "href": link.get('href', ''),
                                "body": snippet_tag.text if snippet_tag else ""
                            })
                            if len(results) >= max_results:
                                break
                        if results:
                            return results
                    except ImportError:
                        logger.warning("BeautifulSoup not available for HTML parsing")
            except Exception as e:
                logger.error(f"DuckDuckGo HTML fallback failed: {e}")

        # If all fail, return empty list
        return results


# -------------------------
# Content Fetcher
# -------------------------
class ContentFetcher:
    """Fetches and cleans content from web URLs with academic focus."""
    async def fetch_content(self, url: str) -> Optional[str]:
        try:
            # Filter out dictionary and definition websites
            dictionary_domains = [
                'dictionary.cambridge.org', 'merriam-webster.com', 'thefreedictionary.com',
                'vocabulary.com', 'dict.cc', 'wordreference.com', 'urbandictionary.com',
                'britannica.com/dictionary', 'oxfordlearnersdictionaries.com'
            ]
            
            # Prioritize academic and educational domains
            academic_domains = [
                '.edu', '.org', 'scholar.google', 'researchgate.net', 'jstor.org', 
                'pubmed.ncbi.nlm.nih.gov', 'arxiv.org', 'ieee.org', 'acm.org',
                'springer.com', 'sciencedirect.com', 'wiley.com', 'tandfonline.com',
                'academia.edu', 'semanticscholar.org', 'ncbi.nlm.nih.gov'
            ]
            
            if any(domain in url.lower() for domain in dictionary_domains):
                logger.info(f"Skipping dictionary source: {url}")
                return None
            
            # Check if it's an academic source
            is_academic = any(domain in url.lower() for domain in academic_domains)
            if is_academic:
                logger.info(f"Processing academic source: {url}")
                
            async with httpx.AsyncClient(timeout=30.0, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1'
            }, follow_redirects=True) as client:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 302 and 'unsupported_browser' in str(e.response.headers.get('location', '')):
                        logger.warning(f"Academic site {url} requires different browser headers, trying alternative approach")
                        # Try with different headers for academic sites
                        try:
                            academic_headers = {
                                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.15',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                                'Accept-Language': 'en-US,en;q=0.9',
                                'Accept-Encoding': 'gzip, deflate, br',
                                'Referer': 'https://www.google.com/',
                                'Connection': 'keep-alive'
                            }
                            response = await client.get(url, headers=academic_headers)
                            response.raise_for_status()
                        except:
                            logger.warning(f"Cannot access content from {url}, generating summary from available metadata")
                            return f"Source: {url}\\n\\nThis academic source was identified but content could not be accessed due to access restrictions. The source appears to be from a reputable academic platform based on the URL structure."
                    elif e.response.status_code == 403:
                        logger.warning(f"403 Forbidden for {url}, trying alternative approach...")
                        # For ResearchGate and similar, try to get abstract/summary
                        if 'researchgate' in url.lower():
                            try:
                                # Try ResearchGate API approach or alternative URL structure
                                alt_url = url.replace('/publication/', '/publication/preview/')
                                response = await client.get(alt_url)
                                if response.status_code == 200:
                                    content = response.text
                                else:
                                    raise httpx.HTTPStatusError("Still blocked", request=None, response=e.response)
                            except:
                                logger.info(f"ResearchGate source {url} blocked, generating summary from metadata")
                                return f"Source: {url}\\n\\nThis ResearchGate publication was identified as relevant but could not be accessed due to platform restrictions. ResearchGate is a reputable academic networking platform with peer-reviewed research publications."
                        else:
                            logger.info(f"Academic source {url} blocked, generating summary from metadata")
                            return f"Source: {url}\\n\\nThis academic source was identified but access was restricted. The source appears to be from a legitimate academic institution or publisher based on domain verification."
                    else:
                        raise
                
                # Enhanced content extraction for academic sources
                content = response.text
                
                # Remove script and style tags
                content = re.sub(r'<script.*?</script>|<style.*?</style>', '', content, flags=re.DOTALL)
                
                # For academic sources, try to extract main content areas
                if is_academic:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Look for common academic content containers
                        academic_selectors = [
                            'article', '.article-content', '.paper-content', '.abstract',
                            '.full-text', '.content', '#content', 'main', '.main-content'
                        ]
                        
                        for selector in academic_selectors:
                            academic_content = soup.select_one(selector)
                            if academic_content and len(academic_content.get_text(strip=True)) > 500:
                                content = str(academic_content)
                                logger.info(f"Extracted academic content using selector: {selector}")
                                break
                    except ImportError:
                        logger.warning("BeautifulSoup not available for enhanced academic parsing")
                
                # Clean HTML tags
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Filter out content that looks like dictionary definitions
                if len(content) < 500:  # Too short for substantial content
                    return None
                    
                # Check for dictionary-like patterns
                dictionary_indicators = [
                    'pronunciation:', 'phonetic:', 'etymology:', 'definition:',
                    'part of speech:', 'noun:', 'verb:', 'adjective:', 'adverb:',
                    '/ˈ', '/ˌ', 'IPA:', '[pronunciation]'
                ]
                
                content_lower = content.lower()
                dictionary_score = sum(1 for indicator in dictionary_indicators if indicator in content_lower)
                
                if dictionary_score >= 3:  # Likely a dictionary page
                    logger.info(f"Filtering out dictionary-like content from: {url}")
                    return None
                
                # Additional relevance check for off-topic content
                irrelevant_indicators = [
                    'jupiter', 'planet', 'solar system', 'astronomy', 'comet',
                    'court case', 'legal', 'litigation', 'judicial',
                    'shopping', 'buy now', 'price', 'cart', 'checkout',
                    'pronunciation', 'phonetic', 'etymology', 'definition',
                    'grammarly', 'advertisement', 'doubleclick'
                ]
                
                irrelevant_score = sum(1 for indicator in irrelevant_indicators if indicator in content_lower)
                if irrelevant_score >= 2:
                    logger.warning(f"Filtering out off-topic content from: {url}")
                    return None
                
                # For AI education queries, ensure content relevance
                if 'ai' in url.lower() or 'education' in url.lower():
                    required_keywords = ['artificial intelligence', 'ai', 'education', 'learning', 'student', 'teach']
                    if not any(kw in content_lower for kw in required_keywords):
                        logger.warning(f"Content not relevant to AI education: {url}")
                        return None
                
                return content
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return None
            
    async def close(self):
        pass


# -------------------------
# GLOBALS
# -------------------------
llm_manager = FreeLLMManager()
search_manager = FreeSearchManager()
content_fetcher = ContentFetcher()


# -------------------------
# PUBLIC HELPERS
# -------------------------
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
        return SourceSummary(
            url="",  # Will be set by caller
            title="",  # Will be set by caller
            key_points=[summary_text[:200] + "..." if len(summary_text) > 200 else summary_text],
            relevance_explanation="Generated from content analysis",
            credibility_assessment="Unknown source credibility",
            summary=summary_text,
            confidence_score=0.8
        )
    except Exception as e:
        logger.error(f"Source summarization failed: {e}")
        return SourceSummary(
            url="",
            title="",
            key_points=[],
            relevance_explanation="Error during summarization",
            credibility_assessment="Could not assess",
            summary=f"Error: {e}",
            confidence_score=0.1
        )
