"""
Token usage and cost tracking utilities.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TokenUsage:
    """Track token usage for LLM calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model_name: str = ""
    provider: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutionMetrics:
    """Track execution metrics for research brief generation."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    token_usage: Dict[str, TokenUsage] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    nodes_executed: list = field(default_factory=list)
    errors_encountered: list = field(default_factory=list)
    
    def finish(self):
        """Mark execution as finished and calculate duration."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.total_cost_usd = sum(usage.cost_usd for usage in self.token_usage.values())
    
    def add_token_usage(self, node_name: str, usage: TokenUsage):
        """Add token usage for a specific node."""
        self.token_usage[node_name] = usage
    
    def add_node_execution(self, node_name: str):
        """Record that a node was executed."""
        self.nodes_executed.append({
            "node": node_name,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_error(self, node_name: str, error: str):
        """Record an error during execution."""
        self.errors_encountered.append({
            "node": node_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/reporting."""
        return {
            "duration_seconds": self.duration_seconds,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": sum(usage.total_tokens for usage in self.token_usage.values()),
            "nodes_executed": len(self.nodes_executed),
            "errors_count": len(self.errors_encountered),
            "token_usage_by_node": {
                name: {
                    "total_tokens": usage.total_tokens,
                    "cost_usd": usage.cost_usd,
                    "model": usage.model_name,
                    "provider": usage.provider
                }
                for name, usage in self.token_usage.items()
            },
            "execution_details": self.nodes_executed,
            "errors": self.errors_encountered
        }

# Free tier cost estimates (approximate)
FREE_TIER_COSTS = {
    "gemini-pro": {
        "input_cost_per_1k": 0.0,  # Free tier
        "output_cost_per_1k": 0.0  # Free tier
    },
    "ollama": {
        "input_cost_per_1k": 0.0,  # Completely free
        "output_cost_per_1k": 0.0  # Completely free
    }
}

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for token usage."""
    if model_name.startswith("gemini"):
        # Free tier for Gemini
        return 0.0
    elif model_name.startswith("ollama"):
        # Always free for Ollama
        return 0.0
    else:
        # Conservative estimate for other models
        return (input_tokens * 0.0015 + output_tokens * 0.002) / 1000

def create_token_usage(
    model_name: str,
    provider: str,
    input_tokens: int,
    output_tokens: int
) -> TokenUsage:
    """Create a TokenUsage object with cost calculation."""
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(model_name, input_tokens, output_tokens)
    
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        model_name=model_name,
        provider=provider
    )
