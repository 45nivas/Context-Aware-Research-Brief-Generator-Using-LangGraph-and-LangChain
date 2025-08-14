"""
Configuration management for the Research Brief Generator.
Handles environment variables, model configurations, and system settings.
Now supports FREE alternatives: Google Gemini + Ollama + DuckDuckGo
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "research-brief-generator"
    print("✅ LangSmith tracing enabled")


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None


@dataclass
class SearchConfig:
    """Configuration for search tools."""
    use_duckduckgo: bool
    serpapi_api_key: Optional[str]
    max_results: int
    max_sources_per_brief: int


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str


@dataclass
class APIConfig:
    """API server configuration."""
    host: str
    port: int
    debug: bool


@dataclass
class TracingConfig:
    """LangSmith tracing configuration."""
    api_key: Optional[str]
    project: str
    endpoint: str
    enabled: bool


class Config:
    """Main configuration class supporting FREE alternatives."""
    
    def __init__(self):
        # FREE Model configurations
        self.primary_model = os.getenv("PRIMARY_MODEL", "gemini-1.5-flash")
        self.secondary_model = os.getenv("SECONDARY_MODEL", "gemini-1.5-pro")
        self.local_model = os.getenv("LOCAL_MODEL", "llama3.1:8b")
        
        # Google Gemini configuration
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Ollama configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Search configuration (FREE alternatives)
        self.search = SearchConfig(
            use_duckduckgo=os.getenv("USE_DUCKDUCKGO_SEARCH", "true").lower() == "true",
            serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
            max_results=int(os.getenv("MAX_SEARCH_RESULTS", "10")),
            max_sources_per_brief=int(os.getenv("MAX_SOURCES_PER_BRIEF", "5"))
        )
        
        # Database configuration
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./research_briefs.db")
        )
        
        # API configuration
        self.api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", os.getenv("API_PORT", "8000"))),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
        
        # Tracing configuration (optional)
        self.tracing = TracingConfig(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            project=os.getenv("LANGSMITH_PROJECT", "research-brief-generator"),
            endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            enabled=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        )
        
        # Convenience properties
        self.max_search_results = self.search.max_results
        self.max_sources_per_brief = self.search.max_sources_per_brief
        
        # Node-specific configurations for FREE setup
        self.node_configs = {
            "context_summarization": {
                "prefer_local": False,
                "max_retries": 2,
                "timeout": 60  # Longer timeout for free tier
            },
            "planning": {
                "prefer_local": False,
                "max_retries": 3,
                "timeout": 90
            },
            "search": {
                "max_retries": 2,
                "timeout": 60,
                "concurrent_queries": 2  # Reduced for free tier
            },
            "content_fetching": {
                "max_retries": 2,
                "timeout": 30,
                "max_content_length": 8000,  # Reduced for free tier
                "concurrent_fetches": 3
            },
            "per_source_summarization": {
                "prefer_local": False,
                "max_retries": 2,
                "timeout": 90,
                "concurrent_summaries": 2
            },
            "synthesis": {
                "prefer_local": False,
                "max_retries": 3,
                "timeout": 120
            },
            "post_processing": {
                "prefer_local": False,
                "max_retries": 2,
                "timeout": 60
            }
        }
    
    def get_available_services(self) -> Dict[str, bool]:
        """Check which services are available."""
        return {
            "google_gemini": bool(self.google_api_key),
            "ollama": True,  # Assume available, will be checked at runtime
            "duckduckgo": self.search.use_duckduckgo,
            "serpapi": bool(self.search.serpapi_api_key),
            "tracing": bool(self.tracing.api_key)
        }
    
    def get_model_rationale(self) -> Dict[str, str]:
        """Return the rationale for FREE model selection."""
        return {
            "primary_model": f"Google Gemini {self.primary_model} - FREE tier with 15 req/min, excellent for most tasks",
            "secondary_model": f"Google Gemini {self.secondary_model} - More capable model for complex analysis",
            "local_model": f"Ollama {self.local_model} - Completely FREE local processing, unlimited usage",
            "search": "DuckDuckGo (free) + SerpAPI (free tier) for comprehensive web search"
        }
    
    def validate_free_setup(self) -> Dict[str, Any]:
        """Validate FREE setup and provide recommendations."""
        status = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check Google Gemini
        if not self.google_api_key:
            status["warnings"].append("No Google API key - get free key from https://makersuite.google.com/app/apikey")
            status["recommendations"].append("Set GOOGLE_API_KEY for best performance")
        
        # Check search services
        if not self.search.use_duckduckgo and not self.search.serpapi_api_key:
            status["errors"].append("No search services configured")
            status["valid"] = False
            status["recommendations"].append("Enable DuckDuckGo search or add SerpAPI key")
        
        # Check database
        if not self.database.url:
            status["errors"].append("No database URL configured")
            status["valid"] = False
        
        return status
    
    def validate(self) -> bool:
        """Validate configuration for FREE setup."""
        validation = self.validate_free_setup()
        if not validation["valid"]:
            error_msg = "Configuration validation failed:\n"
            for error in validation["errors"]:
                error_msg += f"  - {error}\n"
            raise ValueError(error_msg)
        return True


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
