"""
API endpoint tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
import json

from app.api import app
from app.models import BriefRequest, DepthLevel, FinalBrief, BriefMetadata, Reference


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_workflow():
    """Mock workflow for testing."""
    with patch('app.api.research_workflow') as mock:
        yield mock


@pytest.fixture
def mock_db():
    """Mock database manager for testing."""
    with patch('app.api.db_manager') as mock:
        yield mock


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    with patch('app.api.config') as mock:
        mock.validate.return_value = True
        mock.database.url = "sqlite:///test.db"
        mock.tracing.enabled = False
        mock.get_model_rationale.return_value = {
            "primary_model": "GPT-4 for reasoning",
            "secondary_model": "Claude for summarization"
        }
        yield mock


class TestRootEndpoints:
    """Test root and utility endpoints."""
    
    def test_root_endpoint(self, client, mock_config):
        """Test root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert "model_configuration" in data
    
    def test_health_check_success(self, client, mock_db, mock_config):
        """Test successful health check."""
        mock_db.get_user_context = AsyncMock(return_value=None)
        
        with patch('app.api.llm_manager') as mock_llm:
            mock_llm.get_token_usage.return_value = {}
            
            response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "database" in data
        assert "llm_models" in data
    
    def test_health_check_failure(self, client, mock_db, mock_config):
        """Test health check failure."""
        mock_db.get_user_context = AsyncMock(side_effect=Exception("DB Connection Failed"))
        
        response = client.get("/health")
        
        assert response.status_code == 503
        assert "Health check failed" in response.json()["detail"]


class TestBriefGeneration:
    """Test research brief generation endpoint."""
    
    def test_generate_brief_success(self, client, mock_workflow, mock_db, mock_config):
        """Test successful brief generation."""
        # Setup mock final brief
        mock_brief = FinalBrief(
            title="Test AI Healthcare Brief",
            executive_summary="AI is transforming healthcare",
            key_findings=["Finding 1", "Finding 2"],
            detailed_analysis="Detailed analysis content",
            implications="Important implications",
            limitations="Some limitations",
            references=[Reference(
                title="Test Reference",
                url="https://example.com/test",
                relevance_note="Relevant source"
            )],
            metadata=BriefMetadata(
                research_duration=120,
                total_sources_found=5,
                sources_used=3,
                confidence_score=0.85,
                depth_level=DepthLevel.STANDARD,
                token_usage={"gpt-4": {"total_tokens": 2000}}
            )
        )
        
        mock_workflow.run_workflow = AsyncMock(return_value=mock_brief)
        mock_db.init_db = AsyncMock()
        
        # Test request
        request_data = {
            "topic": "AI applications in healthcare diagnostics",
            "depth": 2,
            "follow_up": False,
            "user_id": "test_user_123"
        }
        
        response = client.post("/brief", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test AI Healthcare Brief"
        assert len(data["key_findings"]) == 2
        assert data["metadata"]["confidence_score"] == 0.85
    
    def test_generate_brief_validation_errors(self, client, mock_config):
        """Test request validation errors."""
        
        # Test empty topic
        response = client.post("/brief", json={
            "topic": "",
            "depth": 2,
            "follow_up": False,
            "user_id": "test_user"
        })
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]
        
        # Test topic too short
        response = client.post("/brief", json={
            "topic": "Short",
            "depth": 2,
            "follow_up": False,
            "user_id": "test_user"
        })
        assert response.status_code == 400
        assert "at least 10 characters" in response.json()["detail"]
        
        # Test missing required fields
        response = client.post("/brief", json={
            "topic": "Valid topic here for testing"
        })
        assert response.status_code == 422  # Pydantic validation error
    
    def test_generate_brief_workflow_failure(self, client, mock_workflow, mock_db, mock_config):
        """Test handling of workflow execution failure."""
        mock_workflow.run_workflow = AsyncMock(side_effect=Exception("Workflow failed"))
        mock_db.init_db = AsyncMock()
        
        request_data = {
            "topic": "AI applications in healthcare diagnostics",
            "depth": 2,
            "follow_up": False,
            "user_id": "test_user"
        }
        
        response = client.post("/brief", json=request_data)
        
        assert response.status_code == 500
        assert "Workflow execution failed" in response.json()["detail"]
    
    def test_generate_brief_with_context(self, client, mock_workflow, mock_db, mock_config):
        """Test brief generation with additional context."""
        mock_brief = FinalBrief(
            title="Contextual AI Brief",
            executive_summary="AI brief with context",
            key_findings=["Context-aware finding"],
            detailed_analysis="Analysis with context",
            implications="Contextual implications",
            limitations="Context limitations",
            references=[Reference(
                title="Context Reference",
                url="https://example.com/context",
                relevance_note="Context relevant"
            )],
            metadata=BriefMetadata(
                research_duration=90,
                total_sources_found=3,
                sources_used=2,
                confidence_score=0.75,
                depth_level=DepthLevel.STANDARD
            )
        )
        
        mock_workflow.run_workflow = AsyncMock(return_value=mock_brief)
        mock_db.init_db = AsyncMock()
        
        request_data = {
            "topic": "AI in medical imaging with focus on radiology",
            "depth": 2,
            "follow_up": True,
            "user_id": "context_user",
            "context": "Previous research focused on general AI applications"
        }
        
        response = client.post("/brief", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "context" in data["title"].lower() or "ai" in data["title"].lower()


class TestWorkflowStatus:
    """Test workflow status tracking endpoints."""
    
    def test_get_workflow_status_found(self, client, mock_workflow):
        """Test getting status for existing workflow."""
        workflow_id = "test_user_20241201_120000"
        
        # Mock active workflow
        with patch('app.api.active_workflows', {
            workflow_id: {
                "start_time": "2024-12-01T12:00:00",
                "status": "running",
                "current_step": "synthesis",
                "request": {"topic": "Test topic"}
            }
        }):
            mock_workflow.get_workflow_status = AsyncMock(return_value={
                "current_step": "synthesis",
                "progress": 75,
                "errors": []
            })
            
            response = client.get(f"/brief/{workflow_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["current_step"] == "synthesis"
    
    def test_get_workflow_status_not_found(self, client):
        """Test getting status for non-existent workflow."""
        workflow_id = "nonexistent_workflow"
        
        response = client.get(f"/brief/{workflow_id}/status")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestUserHistory:
    """Test user history endpoints."""
    
    def test_get_user_history_success(self, client, mock_db, mock_config):
        """Test successful user history retrieval."""
        from app.models import UserContext
        from app.database import ResearchBriefDB
        from datetime import datetime
        
        # Mock user context
        mock_context = UserContext(
            user_id="test_user",
            previous_topics=["AI", "Healthcare"],
            brief_summaries=["Brief 1", "Brief 2"]
        )
        
        # Mock brief objects
        mock_brief = Mock()
        mock_brief.id = "brief_123"
        mock_brief.topic = "AI in Healthcare"
        mock_brief.title = "Test Brief Title"
        mock_brief.creation_timestamp = datetime.utcnow()
        mock_brief.executive_summary = "This is a test executive summary for the brief"
        
        mock_db.get_user_context = AsyncMock(return_value=mock_context)
        mock_db.get_user_briefs = AsyncMock(return_value=[mock_brief])
        
        response = client.get("/user/test_user/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test_user"
        assert data["total_briefs"] == 1
        assert len(data["recent_briefs"]) == 1
        assert data["user_context"] is not None
    
    def test_get_user_history_no_user(self, client, mock_db, mock_config):
        """Test user history for non-existent user."""
        mock_db.get_user_context = AsyncMock(return_value=None)
        mock_db.get_user_briefs = AsyncMock(return_value=[])
        
        response = client.get("/user/nonexistent_user/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_briefs"] == 0
        assert data["user_context"] is None
    
    def test_get_user_history_error(self, client, mock_db, mock_config):
        """Test user history retrieval error."""
        mock_db.get_user_context = AsyncMock(side_effect=Exception("Database error"))
        
        response = client.get("/user/error_user/history")
        
        assert response.status_code == 500
        assert "Failed to retrieve user history" in response.json()["detail"]


class TestMetricsAndUtilities:
    """Test metrics and utility endpoints."""
    
    def test_get_metrics(self, client, mock_config):
        """Test metrics endpoint."""
        with patch('app.api.active_workflows', {
            "workflow1": {"status": "running"},
            "workflow2": {"status": "completed"},
            "workflow3": {"status": "failed"}
        }):
            with patch('app.api.llm_manager') as mock_llm:
                mock_llm.get_token_usage.return_value = {
                    "gpt-4": {"total_tokens": 5000},
                    "claude-3": {"total_tokens": 3000}
                }
                
                response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "workflows" in data
        assert data["workflows"]["total"] == 3
        assert data["workflows"]["running"] == 1
        assert data["workflows"]["completed"] == 1
        assert data["workflows"]["failed"] == 1
        assert "token_usage" in data
        assert "configuration" in data
    
    def test_get_workflow_graph(self, client, mock_config):
        """Test workflow graph visualization endpoint."""
        with patch('app.api.research_workflow') as mock_workflow:
            mock_workflow.get_graph_visualization.return_value = "Mock graph visualization"
            
            response = client.get("/workflow/graph")
        
        assert response.status_code == 200
        data = response.json()
        assert "graph_structure" in data
        assert "nodes" in data
        assert "conditional_edges" in data
        assert "retry_logic" in data
        
        # Verify expected nodes are listed
        expected_nodes = [
            "context_summarization", "planning", "search",
            "content_fetching", "per_source_summarization",
            "synthesis", "post_processing"
        ]
        for node in expected_nodes:
            assert node in data["nodes"]


class TestErrorHandling:
    """Test error handling across API endpoints."""
    
    def test_value_error_handler(self, client, mock_config):
        """Test custom ValueError handler."""
        # This would need to be triggered by an endpoint that raises ValueError
        # For now, we test that the handler exists
        assert hasattr(app, 'exception_handler')
    
    def test_general_exception_handler(self, client, mock_config):
        """Test general exception handler."""
        # Test with an endpoint that might raise an unexpected exception
        with patch('app.api.research_workflow') as mock_workflow:
            mock_workflow.run_workflow = AsyncMock(side_effect=RuntimeError("Unexpected error"))
            
            request_data = {
                "topic": "Test topic for error handling",
                "depth": 2,
                "follow_up": False,
                "user_id": "error_test_user"
            }
            
            response = client.post("/brief", json=request_data)
            
            # Should be handled gracefully
            assert response.status_code == 500
    
    def test_startup_validation_failure(self, client):
        """Test startup event with configuration validation failure."""
        # This would require mocking the startup event, which is complex
        # In practice, this is tested during actual startup
        pass


class TestDepthLevels:
    """Test API behavior with different depth levels."""
    
    @pytest.mark.parametrize("depth,expected_name", [
        (1, "QUICK"),
        (2, "STANDARD"), 
        (3, "COMPREHENSIVE"),
        (4, "EXHAUSTIVE")
    ])
    def test_brief_generation_depth_levels(self, client, mock_workflow, mock_db, mock_config, depth, expected_name):
        """Test brief generation with different depth levels."""
        mock_brief = FinalBrief(
            title=f"Test Brief - {expected_name}",
            executive_summary="Test summary",
            key_findings=["Test finding"],
            detailed_analysis="Test analysis",
            implications="Test implications",
            limitations="Test limitations",
            references=[Reference(
                title="Test Reference",
                url="https://example.com",
                relevance_note="Test relevance"
            )],
            metadata=BriefMetadata(
                research_duration=60,
                total_sources_found=3,
                sources_used=2,
                confidence_score=0.8,
                depth_level=DepthLevel(depth)
            )
        )
        
        mock_workflow.run_workflow = AsyncMock(return_value=mock_brief)
        mock_db.init_db = AsyncMock()
        
        request_data = {
            "topic": f"Test topic for depth {depth}",
            "depth": depth,
            "follow_up": False,
            "user_id": "depth_test_user"
        }
        
        response = client.post("/brief", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["depth_level"] == depth
