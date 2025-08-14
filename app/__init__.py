"""
Research Brief Generator - A Context-Aware AI Research System

This package provides a complete research brief generation system using LangGraph
for workflow orchestration and LangChain for LLM abstraction.

Key Components:
- LangGraph workflow with typed state and modular nodes
- Multi-LLM configuration with different providers
- Persistent user context and brief storage
- FastAPI REST API and CLI interfaces
- Comprehensive testing and observability

Usage:
    # CLI
    python -m app.cli generate "AI trends in healthcare" --depth 3
    
    # API
    from app.workflow import research_workflow
    from app.models import BriefRequest, DepthLevel
    
    request = BriefRequest(
        topic="AI trends in healthcare",
        depth=DepthLevel.COMPREHENSIVE,
        user_id="user123"
    )
    
    result = await research_workflow.run_workflow(request)
"""

__version__ = "1.0.0"
__author__ = "Research Brief Generator Team"
__email__ = "contact@researchbrief.ai"

from app.models import (
    BriefRequest, 
    FinalBrief, 
    DepthLevel, 
    GraphState,
    ResearchPlan,
    SourceSummary
)

from app.workflow import research_workflow
from app.config import config
from app.database import db_manager

__all__ = [
    "BriefRequest",
    "FinalBrief", 
    "DepthLevel",
    "GraphState",
    "ResearchPlan",
    "SourceSummary",
    "research_workflow",
    "config",
    "db_manager"
]
