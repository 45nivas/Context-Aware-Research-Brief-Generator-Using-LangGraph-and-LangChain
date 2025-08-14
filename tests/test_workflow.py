"""
End-to-end integration tests for the complete workflow.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from app.models import BriefRequest, DepthLevel, FinalBrief
from app.workflow import ResearchWorkflow


class TestWorkflowIntegration:
    """Test complete workflow integration."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, sample_brief_request):
        """Test complete workflow execution with mocked components."""
        
        # Create workflow instance
        workflow = ResearchWorkflow()
        
        # Mock all external dependencies
        with patch('app.nodes.db_manager') as mock_db:
            with patch('app.nodes.llm_manager') as mock_llm:
                with patch('app.nodes.search_manager') as mock_search:
                    with patch('app.nodes.content_fetcher') as mock_fetcher:
                        
                        # Setup database mocks
                        mock_db.init_db = AsyncMock()
                        mock_db.get_user_context = AsyncMock(return_value=None)
                        mock_db.save_research_brief = AsyncMock(return_value="brief_id")
                        mock_db.update_user_context_with_brief = AsyncMock()
                        
                        # Setup LLM mocks
                        mock_llm.get_llm.return_value = AsyncMock()
                        mock_llm.get_token_usage.return_value = {"gpt-4": {"total_tokens": 1000}}
                        mock_llm.reset_token_usage = Mock()
                        
                        # Mock LLM responses for different nodes
                        planning_response = Mock()
                        planning_response.content = """{
                            "topic": "AI in Healthcare",
                            "steps": [
                                {
                                    "step_number": 1,
                                    "query": "AI healthcare applications",
                                    "rationale": "Overview of applications",
                                    "expected_sources": 3
                                }
                            ],
                            "estimated_duration": 30,
                            "depth_level": 2
                        }"""
                        
                        summary_response = Mock()
                        summary_response.content = """{
                            "url": "https://example.com/test",
                            "title": "AI in Healthcare",
                            "key_points": ["AI improves diagnosis", "ML enhances treatment"],
                            "relevance_explanation": "Directly relevant",
                            "credibility_assessment": "High credibility",
                            "summary": "Comprehensive overview of AI in healthcare",
                            "confidence_score": 0.9
                        }"""
                        
                        brief_response = Mock()
                        brief_response.content = """{
                            "title": "AI in Healthcare: Comprehensive Analysis",
                            "executive_summary": "AI is transforming healthcare",
                            "key_findings": ["AI improves diagnostic accuracy", "ML optimizes treatment"],
                            "detailed_analysis": "Detailed analysis of AI applications",
                            "implications": "Significant implications for healthcare",
                            "limitations": "Some limitations exist",
                            "references": [],
                            "metadata": {
                                "research_duration": 120,
                                "total_sources_found": 3,
                                "sources_used": 2,
                                "confidence_score": 0.85,
                                "depth_level": 2,
                                "token_usage": {}
                            }
                        }"""
                        
                        # Configure LLM to return appropriate responses
                        mock_llm.get_llm.return_value.ainvoke.side_effect = [
                            planning_response,
                            summary_response, 
                            brief_response
                        ]
                        
                        # Setup search mocks
                        from app.models import SearchResult
                        mock_search.combined_search = AsyncMock(return_value=[
                            SearchResult(
                                title="AI in Healthcare Research",
                                url="https://example.com/test",
                                snippet="AI applications in healthcare are growing",
                                relevance_score=0.9
                            )
                        ])
                        
                        # Setup content fetcher mocks
                        from app.models import SourceContent
                        mock_fetcher.fetch_multiple = AsyncMock(return_value=[
                            SourceContent(
                                url="https://example.com/test",
                                title="AI in Healthcare Research",
                                content="Full content about AI in healthcare applications",
                                content_length=200
                            )
                        ])
                        
                        # Mock the Pydantic parsers to return expected objects
                        with patch('app.nodes.PydanticOutputParser') as mock_parser:
                            from app.models import (
                                ResearchPlan, ResearchPlanStep, SourceSummary, 
                                FinalBrief, BriefMetadata, Reference
                            )
                            
                            # Create mock parser instances for different calls
                            def parser_side_effect(pydantic_object):
                                mock_instance = Mock()
                                mock_instance.get_format_instructions.return_value = "Format instructions"
                                
                                if pydantic_object == ResearchPlan:
                                    mock_instance.parse.return_value = ResearchPlan(
                                        topic="AI in Healthcare",
                                        steps=[ResearchPlanStep(
                                            step_number=1,
                                            query="AI healthcare applications",
                                            rationale="Overview of applications",
                                            expected_sources=3
                                        )],
                                        estimated_duration=30,
                                        depth_level=DepthLevel.STANDARD
                                    )
                                elif pydantic_object == SourceSummary:
                                    mock_instance.parse.return_value = SourceSummary(
                                        url="https://example.com/test",
                                        title="AI in Healthcare",
                                        key_points=["AI improves diagnosis", "ML enhances treatment"],
                                        relevance_explanation="Directly relevant",
                                        credibility_assessment="High credibility",
                                        summary="Comprehensive overview of AI in healthcare",
                                        confidence_score=0.9
                                    )
                                elif pydantic_object == FinalBrief:
                                    mock_instance.parse.return_value = FinalBrief(
                                        title="AI in Healthcare: Comprehensive Analysis",
                                        executive_summary="AI is transforming healthcare",
                                        key_findings=["AI improves diagnostic accuracy", "ML optimizes treatment"],
                                        detailed_analysis="Detailed analysis of AI applications",
                                        implications="Significant implications for healthcare",
                                        limitations="Some limitations exist",
                                        references=[Reference(
                                            title="AI in Healthcare Research",
                                            url="https://example.com/test",
                                            relevance_note="Primary source"
                                        )],
                                        metadata=BriefMetadata(
                                            research_duration=120,
                                            total_sources_found=3,
                                            sources_used=2,
                                            confidence_score=0.85,
                                            depth_level=DepthLevel.STANDARD,
                                            token_usage={}
                                        )
                                    )
                                
                                return mock_instance
                            
                            mock_parser.side_effect = parser_side_effect
                            
                            # Execute workflow
                            result = await workflow.run_workflow(sample_brief_request)
        
        # Verify result
        assert isinstance(result, FinalBrief)
        assert result.title == "AI in Healthcare: Comprehensive Analysis"
        assert len(result.key_findings) == 2
        assert len(result.references) > 0
        assert result.metadata.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_follow_up(self):
        """Test workflow execution with follow-up context."""
        from app.models import UserContext
        
        workflow = ResearchWorkflow()
        
        # Create follow-up request
        follow_up_request = BriefRequest(
            topic="AI Healthcare Follow-up Research",
            depth=DepthLevel.STANDARD,
            follow_up=True,
            user_id="test_user",
            context="Building on previous AI research"
        )
        
        # Mock existing user context
        existing_context = UserContext(
            user_id="test_user",
            previous_topics=["AI in Healthcare", "Machine Learning"],
            brief_summaries=["Previous AI research summary"]
        )
        
        with patch('app.nodes.db_manager') as mock_db:
            with patch('app.nodes.llm_manager') as mock_llm:
                with patch('app.nodes.search_manager') as mock_search:
                    with patch('app.nodes.content_fetcher') as mock_fetcher:
                        
                        # Setup mocks
                        mock_db.init_db = AsyncMock()
                        mock_db.get_user_context = AsyncMock(return_value=existing_context)
                        mock_db.save_research_brief = AsyncMock(return_value="brief_id")
                        mock_db.update_user_context_with_brief = AsyncMock()
                        
                        mock_llm.get_llm.return_value = AsyncMock()
                        mock_llm.get_token_usage.return_value = {}
                        mock_llm.reset_token_usage = Mock()
                        
                        # Mock simplified responses to avoid complex parsing
                        mock_llm.get_llm.return_value.ainvoke.return_value = Mock(
                            content="Mock LLM response"
                        )
                        
                        from app.models import SearchResult, SourceContent
                        mock_search.combined_search = AsyncMock(return_value=[
                            SearchResult(
                                title="Follow-up Research",
                                url="https://example.com/followup",
                                snippet="Follow-up research content",
                                relevance_score=0.8
                            )
                        ])
                        
                        mock_fetcher.fetch_multiple = AsyncMock(return_value=[
                            SourceContent(
                                url="https://example.com/followup",
                                title="Follow-up Research", 
                                content="Detailed follow-up content",
                                content_length=150
                            )
                        ])
                        
                        # Execute workflow - even if parsing fails, we should get a fallback brief
                        result = await workflow.run_workflow(follow_up_request)
        
        # Verify we get a result (may be fallback)
        assert isinstance(result, FinalBrief)
        assert result.title is not None
        assert len(result.title) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow error handling and fallback mechanisms."""
        
        workflow = ResearchWorkflow()
        
        error_request = BriefRequest(
            topic="Error Test Topic",
            depth=DepthLevel.QUICK,
            follow_up=False,
            user_id="error_user"
        )
        
        with patch('app.nodes.db_manager') as mock_db:
            with patch('app.nodes.llm_manager') as mock_llm:
                with patch('app.nodes.search_manager') as mock_search:
                    with patch('app.nodes.content_fetcher') as mock_fetcher:
                        
                        # Setup failing mocks
                        mock_db.init_db = AsyncMock()
                        mock_db.get_user_context = AsyncMock(side_effect=Exception("DB Error"))
                        mock_db.save_research_brief = AsyncMock(side_effect=Exception("Save Error"))
                        mock_db.update_user_context_with_brief = AsyncMock()
                        
                        mock_llm.get_llm.return_value = AsyncMock()
                        mock_llm.get_llm.return_value.ainvoke.side_effect = Exception("LLM Error")
                        mock_llm.get_token_usage.return_value = {}
                        mock_llm.reset_token_usage = Mock()
                        
                        mock_search.combined_search = AsyncMock(side_effect=Exception("Search Error"))
                        mock_fetcher.fetch_multiple = AsyncMock(side_effect=Exception("Fetch Error"))
                        
                        # Execute workflow - should handle errors gracefully
                        result = await workflow.run_workflow(error_request)
        
        # Should get fallback brief even with all errors
        assert isinstance(result, FinalBrief)
        assert result.title is not None
        assert len(result.title) > 0
        assert result.metadata.confidence_score <= 0.5  # Low confidence due to errors
    
    @pytest.mark.asyncio
    async def test_workflow_depth_levels(self):
        """Test workflow behavior with different depth levels."""
        
        workflow = ResearchWorkflow()
        
        # Test each depth level
        for depth in [DepthLevel.QUICK, DepthLevel.STANDARD, DepthLevel.COMPREHENSIVE, DepthLevel.EXHAUSTIVE]:
            request = BriefRequest(
                topic=f"AI Test Topic for {depth.name}",
                depth=depth,
                follow_up=False,
                user_id="depth_test_user"
            )
            
            with patch('app.nodes.db_manager') as mock_db:
                with patch('app.nodes.llm_manager') as mock_llm:
                    with patch('app.nodes.search_manager') as mock_search:
                        with patch('app.nodes.content_fetcher') as mock_fetcher:
                            
                            # Setup basic mocks
                            mock_db.init_db = AsyncMock()
                            mock_db.get_user_context = AsyncMock(return_value=None)
                            mock_db.save_research_brief = AsyncMock(return_value="brief_id")
                            mock_db.update_user_context_with_brief = AsyncMock()
                            
                            mock_llm.get_llm.return_value = AsyncMock()
                            mock_llm.get_llm.return_value.ainvoke.return_value = Mock(content="Mock response")
                            mock_llm.get_token_usage.return_value = {}
                            mock_llm.reset_token_usage = Mock()
                            
                            from app.models import SearchResult, SourceContent
                            mock_search.combined_search = AsyncMock(return_value=[
                                SearchResult(
                                    title=f"Test Article {depth.name}",
                                    url=f"https://example.com/{depth.name.lower()}",
                                    snippet=f"Content for {depth.name} depth",
                                    relevance_score=0.8
                                )
                            ])
                            
                            mock_fetcher.fetch_multiple = AsyncMock(return_value=[
                                SourceContent(
                                    url=f"https://example.com/{depth.name.lower()}",
                                    title=f"Test Article {depth.name}",
                                    content=f"Detailed content for {depth.name} research",
                                    content_length=100
                                )
                            ])
                            
                            # Execute workflow
                            result = await workflow.run_workflow(request)
            
            # Verify result
            assert isinstance(result, FinalBrief)
            assert result.metadata.depth_level == depth
            assert depth.name.lower() in result.title.lower() or "test" in result.title.lower()
    
    @pytest.mark.asyncio
    async def test_workflow_status_tracking(self):
        """Test workflow status tracking functionality."""
        
        workflow = ResearchWorkflow()
        thread_id = "test_thread_123"
        
        # Test getting status for non-existent workflow
        status = await workflow.get_workflow_status(thread_id)
        assert "error" in status or status.get("current_step") == "error"
    
    def test_workflow_graph_visualization(self):
        """Test workflow graph visualization."""
        
        workflow = ResearchWorkflow()
        graph_viz = workflow.get_graph_visualization()
        
        assert isinstance(graph_viz, str)
        assert "context_summarization" in graph_viz
        assert "planning" in graph_viz
        assert "search" in graph_viz
        assert "synthesis" in graph_viz
        assert "post_processing" in graph_viz
    
    def test_workflow_conditional_transitions(self):
        """Test workflow conditional transition logic."""
        
        workflow = ResearchWorkflow()
        
        # Test _should_fetch_content logic
        from app.models import GraphState, SearchResult
        
        # State with search results - should fetch content
        state_with_results = GraphState(
            topic="Test",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test"
        )
        state_with_results.search_results = [SearchResult(
            title="Test",
            url="https://test.com",
            snippet="Test snippet",
            relevance_score=0.8
        )]
        
        decision = workflow._should_fetch_content(state_with_results)
        assert decision == "fetch_content"
        
        # State without search results - should skip to summarization
        state_no_results = GraphState(
            topic="Test",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test"
        )
        
        decision = workflow._should_fetch_content(state_no_results)
        assert decision == "skip_to_summarization"
    
    def test_progress_calculation(self):
        """Test workflow progress calculation."""
        
        workflow = ResearchWorkflow()
        
        # Test various steps
        assert workflow._calculate_progress("initialization") > 0
        assert workflow._calculate_progress("context_summarization") > workflow._calculate_progress("initialization")
        assert workflow._calculate_progress("planning") > workflow._calculate_progress("context_summarization")
        assert workflow._calculate_progress("synthesis") > workflow._calculate_progress("search")
        assert workflow._calculate_progress("post_processing") > workflow._calculate_progress("synthesis")
        
        # Test unknown step
        assert workflow._calculate_progress("unknown_step") == 0
