"""
Unit tests for Pydantic models and schema validation.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models import (
    BriefRequest, DepthLevel, ResearchPlan, ResearchPlanStep,
    SearchResult, SourceContent, SourceSummary, FinalBrief,
    BriefMetadata, Reference, GraphState, UserContext
)


class TestBriefRequest:
    """Test BriefRequest model validation."""
    
    def test_valid_brief_request(self):
        """Test valid brief request creation."""
        request = BriefRequest(
            topic="AI in healthcare technology trends",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user_123"
        )
        
        assert request.topic == "AI in healthcare technology trends"
        assert request.depth == DepthLevel.STANDARD
        assert request.follow_up is False
        assert request.user_id == "test_user_123"
        assert request.context is None
    
    def test_topic_too_short(self):
        """Test topic length validation."""
        with pytest.raises(ValidationError) as exc_info:
            BriefRequest(
                topic="Short",  # Less than 10 characters
                depth=DepthLevel.STANDARD,
                follow_up=False,
                user_id="test_user"
            )
        
        assert "at least 10 characters" in str(exc_info.value)
    
    def test_topic_too_long(self):
        """Test topic maximum length validation."""
        long_topic = "x" * 501  # More than 500 characters
        
        with pytest.raises(ValidationError) as exc_info:
            BriefRequest(
                topic=long_topic,
                depth=DepthLevel.STANDARD,
                follow_up=False,
                user_id="test_user"
            )
        
        assert "at most 500 characters" in str(exc_info.value)
    
    def test_empty_user_id(self):
        """Test user_id validation."""
        with pytest.raises(ValidationError) as exc_info:
            BriefRequest(
                topic="Valid topic here",
                depth=DepthLevel.STANDARD,
                follow_up=False,
                user_id=""  # Empty user_id
            )
        
        assert "at least 1 character" in str(exc_info.value)
    
    def test_default_values(self):
        """Test default values are set correctly."""
        request = BriefRequest(
            topic="AI in healthcare",
            user_id="test_user"
        )
        
        assert request.depth == DepthLevel.STANDARD
        assert request.follow_up is False
        assert request.context is None


class TestDepthLevel:
    """Test DepthLevel enum."""
    
    def test_depth_level_values(self):
        """Test depth level enum values."""
        assert DepthLevel.QUICK == 1
        assert DepthLevel.STANDARD == 2
        assert DepthLevel.COMPREHENSIVE == 3
        assert DepthLevel.EXHAUSTIVE == 4
    
    def test_depth_level_names(self):
        """Test depth level enum names."""
        assert DepthLevel.QUICK.name == "QUICK"
        assert DepthLevel.STANDARD.name == "STANDARD"
        assert DepthLevel.COMPREHENSIVE.name == "COMPREHENSIVE"
        assert DepthLevel.EXHAUSTIVE.name == "EXHAUSTIVE"


class TestResearchPlan:
    """Test ResearchPlan and ResearchPlanStep models."""
    
    def test_valid_research_plan(self):
        """Test valid research plan creation."""
        steps = [
            ResearchPlanStep(
                step_number=1,
                query="AI healthcare overview",
                rationale="Get general understanding",
                expected_sources=3
            ),
            ResearchPlanStep(
                step_number=2,
                query="AI diagnostic tools",
                rationale="Focus on diagnostic applications",
                expected_sources=2
            )
        ]
        
        plan = ResearchPlan(
            topic="AI in Healthcare",
            steps=steps,
            estimated_duration=60,
            depth_level=DepthLevel.COMPREHENSIVE
        )
        
        assert plan.topic == "AI in Healthcare"
        assert len(plan.steps) == 2
        assert plan.estimated_duration == 60
        assert plan.depth_level == DepthLevel.COMPREHENSIVE
    
    def test_research_plan_step_defaults(self):
        """Test ResearchPlanStep default values."""
        step = ResearchPlanStep(
            step_number=1,
            query="test query",
            rationale="test rationale"
        )
        
        assert step.expected_sources == 3  # Default value


class TestSearchResult:
    """Test SearchResult model."""
    
    def test_valid_search_result(self):
        """Test valid search result creation."""
        result = SearchResult(
            title="AI in Healthcare Research",
            url="https://example.com/research",
            snippet="This article discusses AI applications...",
            relevance_score=0.85
        )
        
        assert result.title == "AI in Healthcare Research"
        assert result.url == "https://example.com/research"
        assert result.snippet == "This article discusses AI applications..."
        assert result.relevance_score == 0.85
    
    def test_relevance_score_validation(self):
        """Test relevance score bounds validation."""
        # Test valid bounds
        SearchResult(
            title="Test",
            url="https://test.com",
            snippet="Test snippet",
            relevance_score=0.0
        )
        
        SearchResult(
            title="Test",
            url="https://test.com", 
            snippet="Test snippet",
            relevance_score=1.0
        )
        
        # Test invalid bounds
        with pytest.raises(ValidationError):
            SearchResult(
                title="Test",
                url="https://test.com",
                snippet="Test snippet", 
                relevance_score=-0.1
            )
        
        with pytest.raises(ValidationError):
            SearchResult(
                title="Test",
                url="https://test.com",
                snippet="Test snippet",
                relevance_score=1.1
            )


class TestSourceContent:
    """Test SourceContent model."""
    
    def test_valid_source_content(self):
        """Test valid source content creation."""
        content = SourceContent(
            url="https://example.com/article",
            title="Test Article",
            content="Full article content here...",
            content_length=100
        )
        
        assert content.url == "https://example.com/article"
        assert content.title == "Test Article"
        assert content.content == "Full article content here..."
        assert content.content_length == 100
        assert isinstance(content.fetch_timestamp, datetime)


class TestSourceSummary:
    """Test SourceSummary model."""
    
    def test_valid_source_summary(self):
        """Test valid source summary creation."""
        summary = SourceSummary(
            url="https://example.com/article",
            title="Test Article",
            key_points=["Point 1", "Point 2", "Point 3"],
            relevance_explanation="Highly relevant to the research topic",
            credibility_assessment="High credibility - peer reviewed",
            summary="Comprehensive summary of the article content",
            confidence_score=0.9
        )
        
        assert summary.url == "https://example.com/article"
        assert len(summary.key_points) == 3
        assert summary.confidence_score == 0.9
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        with pytest.raises(ValidationError):
            SourceSummary(
                url="https://test.com",
                title="Test",
                key_points=["Point 1"],
                relevance_explanation="Test",
                credibility_assessment="Test",
                summary="Test summary",
                confidence_score=1.5  # Invalid - too high
            )


class TestFinalBrief:
    """Test FinalBrief model."""
    
    def test_valid_final_brief(self, sample_final_brief):
        """Test valid final brief structure."""
        assert sample_final_brief.title == "AI in Healthcare: Transforming Medical Practice"
        assert len(sample_final_brief.key_findings) == 3
        assert len(sample_final_brief.references) == 1
        assert isinstance(sample_final_brief.metadata, BriefMetadata)
    
    def test_references_validation(self):
        """Test that at least one reference is required."""
        with pytest.raises(ValidationError) as exc_info:
            FinalBrief(
                title="Test Brief",
                executive_summary="Test summary",
                key_findings=["Finding 1"],
                detailed_analysis="Test analysis",
                implications="Test implications",
                limitations="Test limitations",
                references=[],  # Empty references should fail
                metadata=BriefMetadata(
                    research_duration=60,
                    total_sources_found=1,
                    sources_used=1,
                    confidence_score=0.8,
                    depth_level=DepthLevel.STANDARD
                )
            )
        
        assert "at least one reference" in str(exc_info.value)


class TestReference:
    """Test Reference model."""
    
    def test_valid_reference(self):
        """Test valid reference creation."""
        ref = Reference(
            title="Test Article",
            url="https://example.com/test",
            relevance_note="Highly relevant to research topic"
        )
        
        assert ref.title == "Test Article"
        assert ref.url == "https://example.com/test"
        assert ref.relevance_note == "Highly relevant to research topic"
        assert isinstance(ref.access_date, datetime)


class TestBriefMetadata:
    """Test BriefMetadata model."""
    
    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = BriefMetadata(
            research_duration=120,
            total_sources_found=5,
            sources_used=3,
            confidence_score=0.85,
            depth_level=DepthLevel.COMPREHENSIVE,
            token_usage={"gpt-4": {"total_tokens": 2500}}
        )
        
        assert metadata.research_duration == 120
        assert metadata.total_sources_found == 5
        assert metadata.sources_used == 3
        assert metadata.confidence_score == 0.85
        assert metadata.depth_level == DepthLevel.COMPREHENSIVE
        assert isinstance(metadata.creation_timestamp, datetime)


class TestGraphState:
    """Test GraphState model."""
    
    def test_valid_graph_state(self):
        """Test valid graph state creation."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        
        assert state.topic == "AI in Healthcare"
        assert state.depth == DepthLevel.STANDARD
        assert state.follow_up is False
        assert state.user_id == "test_user"
        assert state.current_step == "initialization"
        assert isinstance(state.start_time, datetime)
    
    def test_graph_state_defaults(self):
        """Test graph state default values."""
        state = GraphState(
            topic="Test Topic",
            depth=DepthLevel.QUICK,
            follow_up=True,
            user_id="test_user"
        )
        
        assert state.search_results == []
        assert state.source_contents == []
        assert state.source_summaries == []
        assert state.errors == []
        assert state.retry_count == {}


class TestUserContext:
    """Test UserContext model."""
    
    def test_valid_user_context(self):
        """Test valid user context creation."""
        context = UserContext(
            user_id="test_user_123",
            previous_topics=["AI", "Healthcare", "Technology"],
            brief_summaries=["Summary 1", "Summary 2"],
            preferences={"theme": "dark", "depth": "standard"}
        )
        
        assert context.user_id == "test_user_123"
        assert len(context.previous_topics) == 3
        assert len(context.brief_summaries) == 2
        assert context.preferences["theme"] == "dark"
        assert isinstance(context.last_updated, datetime)
    
    def test_user_context_defaults(self):
        """Test user context default values."""
        context = UserContext(user_id="test_user")
        
        assert context.previous_topics == []
        assert context.brief_summaries == []
        assert context.preferences == {}
        assert isinstance(context.last_updated, datetime)
