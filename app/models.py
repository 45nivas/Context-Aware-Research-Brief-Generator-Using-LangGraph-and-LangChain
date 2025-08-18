"""
Pydantic models for the Research Brief Generator.
Provides type safety and schema validation for all data structures.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class DepthLevel(int, Enum):
    """Research depth levels."""
    QUICK = 1
    STANDARD = 2
    COMPREHENSIVE = 3
    EXHAUSTIVE = 4


class ResearchPlanStep(BaseModel):
    """Individual step in the research plan."""
    step_number: int = Field(..., description="Sequential step number")
    query: str = Field(..., description="Search query for this step")
    rationale: str = Field(..., description="Why this step is needed")
    expected_sources: int = Field(default=3, description="Expected number of sources")


class ResearchPlan(BaseModel):
    """Complete research plan with multiple steps."""
    topic: str = Field(..., description="Main research topic")
    steps: List[ResearchPlanStep] = Field(..., description="List of research steps")
    estimated_duration: int = Field(..., description="Estimated duration in minutes")
    depth_level: DepthLevel = Field(..., description="Research depth level")


class SearchResult(BaseModel):
    """Individual search result."""
    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Brief snippet from the source")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class SourceContent(BaseModel):
    """Full content fetched from a source."""
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")
    content: str = Field(..., description="Full text content")
    fetch_timestamp: datetime = Field(default_factory=datetime.utcnow)
    content_length: int = Field(..., description="Length of content in characters")


class SourceSummary(BaseModel):
    """Structured summary of a single source."""
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")
    key_points: List[str] = Field(..., description="Main points from the source")
    relevance_explanation: str = Field(..., description="Why this source is relevant")
    credibility_assessment: str = Field(..., description="Assessment of source credibility")
    summary: str = Field(..., description="Comprehensive summary of the source")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in summary accuracy")


class Reference(BaseModel):
    """Citation reference for the final brief."""
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    access_date: datetime = Field(default_factory=datetime.utcnow)
    relevance_note: str = Field(..., description="Brief note on relevance")


class BriefMetadata(BaseModel):
    """Metadata for the research brief."""
    creation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    research_duration: int = Field(..., description="Time taken in seconds")
    total_sources_found: int = Field(..., description="Total sources discovered")
    sources_used: int = Field(..., description="Sources actually used in brief")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    depth_level: DepthLevel = Field(..., description="Research depth level used")
    token_usage: Dict[str, Any] = Field(default_factory=dict, description="Token usage by model")


class FinalBrief(BaseModel):
    """Final research brief output."""
    title: str = Field(..., description="Brief title")
    executive_summary: str = Field(..., description="High-level summary")
    key_findings: List[str] = Field(..., description="Main findings")
    detailed_analysis: str = Field(..., description="Comprehensive analysis")
    implications: str = Field(..., description="Implications and insights")
    limitations: str = Field(..., description="Research limitations")
    references: List[Reference] = Field(..., description="Source references")
    metadata: BriefMetadata = Field(..., description="Brief metadata")

    @validator('references')
    def validate_references(cls, v):
        """Ensure at least one reference is included."""
        if len(v) < 1:
            raise ValueError("At least one reference is required")
        return v


class BriefRequest(BaseModel):
    """Request model for brief generation."""
    topic: str = Field(..., min_length=10, max_length=500, description="Research topic", example="The impact of AI on education")
    depth: DepthLevel = Field(default=DepthLevel.STANDARD, description="Research depth level (can be integer 1-4 or string: QUICK, STANDARD, COMPREHENSIVE, EXHAUSTIVE)", example="STANDARD")
    follow_up: bool = Field(default=False, description="Whether this is a follow-up request", example=False)
    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier", example="testuser123")
    context: Optional[str] = Field(None, description="Additional context from user", example="Recent trends and challenges")


class UserContext(BaseModel):
    """User's historical context and preferences."""
    user_id: str = Field(..., description="User identifier")
    previous_topics: List[str] = Field(default_factory=list, description="Previously researched topics")
    brief_summaries: List[str] = Field(default_factory=list, description="Summaries of previous briefs")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class GraphState(BaseModel):
    """State object for LangGraph workflow."""
    # Input parameters
    topic: str
    depth: DepthLevel
    follow_up: bool
    user_id: str
    context: Optional[str] = None
    
    # Workflow state
    user_context: Optional[UserContext] = None
    research_plan: Optional[ResearchPlan] = None
    search_results: List[SearchResult] = Field(default_factory=list)
    source_contents: List[SourceContent] = Field(default_factory=list)
    source_summaries: List[SourceSummary] = Field(default_factory=list)
    final_brief: Optional[FinalBrief] = None
    
    # Execution metadata
    start_time: datetime = Field(default_factory=datetime.utcnow)
    current_step: str = Field(default="initialization")
    errors: List[str] = Field(default_factory=list)
    retry_count: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class NodeResult(BaseModel):
    """Result from a graph node execution."""
    node_name: str
    success: bool
    execution_time: float
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    should_retry: bool = False
