"""
Test configuration and fixtures for the Research Brief Generator.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from app.models import (
    BriefRequest, DepthLevel, GraphState, ResearchPlan, ResearchPlanStep,
    SearchResult, SourceContent, SourceSummary, FinalBrief, BriefMetadata, Reference
)


@pytest.fixture
def sample_brief_request():
    """Sample brief request for testing."""
    return BriefRequest(
        topic="Artificial Intelligence in Healthcare",
        depth=DepthLevel.STANDARD,
        follow_up=False,
        user_id="test_user_123",
        context="Focus on recent developments and applications"
    )


@pytest.fixture
def sample_research_plan():
    """Sample research plan for testing."""
    return ResearchPlan(
        topic="Artificial Intelligence in Healthcare",
        steps=[
            ResearchPlanStep(
                step_number=1,
                query="AI healthcare applications 2024",
                rationale="Get overview of current applications",
                expected_sources=3
            ),
            ResearchPlanStep(
                step_number=2,
                query="machine learning medical diagnosis",
                rationale="Focus on diagnostic applications",
                expected_sources=3
            )
        ],
        estimated_duration=45,
        depth_level=DepthLevel.STANDARD
    )


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        SearchResult(
            title="AI Revolutionizes Medical Diagnosis",
            url="https://example.com/ai-diagnosis",
            snippet="AI systems are showing remarkable accuracy in medical diagnosis...",
            relevance_score=0.9
        ),
        SearchResult(
            title="Machine Learning in Radiology",
            url="https://example.com/ml-radiology", 
            snippet="Machine learning algorithms are transforming radiology practices...",
            relevance_score=0.8
        ),
        SearchResult(
            title="Healthcare AI Ethics and Challenges",
            url="https://example.com/ai-ethics",
            snippet="The implementation of AI in healthcare raises important ethical questions...",
            relevance_score=0.7
        )
    ]


@pytest.fixture
def sample_source_contents():
    """Sample source contents for testing."""
    return [
        SourceContent(
            url="https://example.com/ai-diagnosis",
            title="AI Revolutionizes Medical Diagnosis",
            content="Artificial intelligence systems are revolutionizing medical diagnosis by providing unprecedented accuracy and speed. Recent studies show that AI can match or exceed human radiologists in detecting certain conditions. The technology uses deep learning algorithms trained on millions of medical images to identify patterns invisible to the human eye.",
            fetch_timestamp=datetime.utcnow(),
            content_length=512
        ),
        SourceContent(
            url="https://example.com/ml-radiology",
            title="Machine Learning in Radiology",
            content="Machine learning is transforming radiology by automating image analysis and improving diagnostic accuracy. Radiologists can now leverage AI tools to process scans more efficiently, reducing interpretation time while maintaining high accuracy. The integration of ML in radiology workflows is becoming standard practice in modern healthcare facilities.",
            fetch_timestamp=datetime.utcnow(),
            content_length=448
        )
    ]


@pytest.fixture
def sample_source_summaries():
    """Sample source summaries for testing."""
    return [
        SourceSummary(
            url="https://example.com/ai-diagnosis",
            title="AI Revolutionizes Medical Diagnosis",
            key_points=[
                "AI systems match or exceed human radiologist accuracy",
                "Deep learning algorithms trained on millions of medical images",
                "Can identify patterns invisible to human eye"
            ],
            relevance_explanation="Directly addresses AI applications in medical diagnosis",
            credibility_assessment="High credibility - peer-reviewed research",
            summary="Comprehensive overview of AI diagnostic capabilities and performance",
            confidence_score=0.9
        ),
        SourceSummary(
            url="https://example.com/ml-radiology",
            title="Machine Learning in Radiology",
            key_points=[
                "ML automates image analysis in radiology",
                "Improves diagnostic accuracy and efficiency",
                "Becoming standard practice in modern healthcare"
            ],
            relevance_explanation="Focuses specifically on ML applications in radiology",
            credibility_assessment="High credibility - industry publication",
            summary="Details ML integration in radiology workflows and benefits",
            confidence_score=0.8
        )
    ]


@pytest.fixture
def sample_final_brief():
    """Sample final brief for testing."""
    return FinalBrief(
        title="AI in Healthcare: Transforming Medical Practice",
        executive_summary="Artificial Intelligence is revolutionizing healthcare through advanced diagnostic capabilities, improved treatment planning, and enhanced operational efficiency.",
        key_findings=[
            "AI diagnostic systems achieve human-level or superior accuracy",
            "Machine learning streamlines radiology workflows",
            "Ethical considerations remain paramount in AI healthcare deployment"
        ],
        detailed_analysis="The integration of artificial intelligence in healthcare represents a paradigm shift in medical practice. AI systems demonstrate remarkable capabilities in medical imaging, with some studies showing diagnostic accuracy that matches or exceeds that of experienced radiologists.",
        implications="Healthcare organizations must invest in AI infrastructure while ensuring ethical deployment and maintaining human oversight in critical decisions.",
        limitations="This research is based on limited sources and may not reflect the complete current state of AI in healthcare. Rapid technological advancement may make some findings outdated quickly.",
        references=[
            Reference(
                title="AI Revolutionizes Medical Diagnosis",
                url="https://example.com/ai-diagnosis",
                access_date=datetime.utcnow(),
                relevance_note="Primary source on AI diagnostic capabilities"
            )
        ],
        metadata=BriefMetadata(
            research_duration=120,
            total_sources_found=5,
            sources_used=3,
            confidence_score=0.85,
            depth_level=DepthLevel.STANDARD,
            token_usage={"gpt-4": {"total_tokens": 2500}}
        )
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock_response = Mock()
    mock_response.content = """
    {
        "topic": "Artificial Intelligence in Healthcare",
        "steps": [
            {
                "step_number": 1,
                "query": "AI healthcare applications 2024",
                "rationale": "Get overview of current applications",
                "expected_sources": 3
            }
        ],
        "estimated_duration": 45,
        "depth_level": 2
    }
    """
    return mock_response


@pytest.fixture
def mock_search_manager():
    """Mock search manager for testing."""
    mock_manager = Mock()
    mock_manager.combined_search = AsyncMock(return_value=[
        SearchResult(
            title="Test AI Article",
            url="https://example.com/test",
            snippet="Test content about AI in healthcare",
            relevance_score=0.8
        )
    ])
    return mock_manager


@pytest.fixture
def mock_content_fetcher():
    """Mock content fetcher for testing."""
    mock_fetcher = Mock()
    mock_fetcher.fetch_multiple = AsyncMock(return_value=[
        SourceContent(
            url="https://example.com/test",
            title="Test AI Article",
            content="Detailed test content about AI in healthcare applications",
            fetch_timestamp=datetime.utcnow(),
            content_length=200
        )
    ])
    return mock_fetcher


@pytest.fixture 
def mock_db_manager():
    """Mock database manager for testing."""
    mock_db = Mock()
    mock_db.init_db = AsyncMock()
    mock_db.get_user_context = AsyncMock(return_value=None)
    mock_db.save_user_context = AsyncMock()
    mock_db.save_research_brief = AsyncMock(return_value="test_brief_id")
    mock_db.update_user_context_with_brief = AsyncMock()
    return mock_db


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class MockLLMManager:
    """Mock LLM manager for testing."""
    
    def __init__(self):
        self.token_usage = {"gpt-4": {"total_tokens": 1000}}
    
    def get_llm(self, model_type: str):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content='{"test": "response"}'))
        return mock_llm
    
    def get_token_usage(self):
        return self.token_usage
    
    def reset_token_usage(self):
        self.token_usage = {}


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager fixture."""
    return MockLLMManager()
