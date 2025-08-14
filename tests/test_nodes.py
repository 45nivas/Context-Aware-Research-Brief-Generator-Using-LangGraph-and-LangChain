"""
Unit tests for individual workflow nodes.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from app.models import GraphState, DepthLevel, UserContext, ResearchPlan, ResearchPlanStep
from app.nodes import (
    context_summarization_node,
    planning_node,
    search_node,
    content_fetching_node,
    per_source_summarization_node,
    synthesis_node,
    post_processing_node
)


class TestContextSummarizationNode:
    """Test context summarization node."""
    
    @pytest.mark.asyncio
    async def test_skip_context_for_new_query(self):
        """Test skipping context summarization for non-follow-up queries."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,  # Not a follow-up
            user_id="test_user"
        )
        
        result = await context_summarization_node(state)
        
        assert result.current_step == "context_summarization"
        assert result.user_context is not None
        assert result.user_context.user_id == "test_user"
    
    @pytest.mark.asyncio
    async def test_process_context_for_follow_up(self, mock_db_manager, mock_llm_manager):
        """Test processing context for follow-up queries."""
        # Setup existing user context
        existing_context = UserContext(
            user_id="test_user",
            previous_topics=["Previous Topic"],
            brief_summaries=["Previous brief summary"]
        )
        
        state = GraphState(
            topic="AI in Healthcare Follow-up",
            depth=DepthLevel.STANDARD,
            follow_up=True,
            user_id="test_user"
        )
        
        with patch('app.nodes.db_manager', mock_db_manager):
            with patch('app.nodes.llm_manager', mock_llm_manager):
                mock_db_manager.get_user_context.return_value = existing_context
                
                # Mock LLM response
                mock_llm = mock_llm_manager.get_llm("secondary")
                mock_llm.ainvoke.return_value = Mock(content="Contextual summary of previous research")
                
                result = await context_summarization_node(state)
        
        assert result.current_step == "context_summarization"
        assert result.user_context is not None
        assert "last_context_summary" in result.user_context.preferences
    
    @pytest.mark.asyncio
    async def test_handle_context_error(self, mock_db_manager):
        """Test error handling in context summarization."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=True,
            user_id="test_user"
        )
        
        with patch('app.nodes.db_manager', mock_db_manager):
            # Simulate database error
            mock_db_manager.get_user_context.side_effect = Exception("Database error")
            
            result = await context_summarization_node(state)
        
        assert result.current_step == "context_summarization"
        assert len(result.errors) > 0
        assert "Context summarization failed" in result.errors[0]
        assert result.user_context is not None  # Should create empty context


class TestPlanningNode:
    """Test planning node."""
    
    @pytest.mark.asyncio
    async def test_successful_planning(self, mock_llm_manager):
        """Test successful research plan generation."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.user_context = UserContext(user_id="test_user")
        
        # Mock successful LLM response
        plan_json = """{
            "topic": "AI in Healthcare",
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
        }"""
        
        with patch('app.nodes.llm_manager', mock_llm_manager):
            mock_llm = mock_llm_manager.get_llm("primary")
            mock_llm.ainvoke.return_value = Mock(content=plan_json)
            
            with patch('app.nodes.PydanticOutputParser') as mock_parser:
                mock_parser_instance = Mock()
                mock_parser.return_value = mock_parser_instance
                mock_parser_instance.parse.return_value = ResearchPlan(
                    topic="AI in Healthcare",
                    steps=[ResearchPlanStep(
                        step_number=1,
                        query="AI healthcare applications 2024",
                        rationale="Get overview of current applications",
                        expected_sources=3
                    )],
                    estimated_duration=45,
                    depth_level=DepthLevel.STANDARD
                )
                
                result = await planning_node(state)
        
        assert result.current_step == "planning"
        assert result.research_plan is not None
        assert len(result.research_plan.steps) == 1
        assert result.research_plan.topic == "AI in Healthcare"
    
    @pytest.mark.asyncio
    async def test_planning_fallback(self, mock_llm_manager):
        """Test fallback plan creation when LLM fails."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.user_context = UserContext(user_id="test_user")
        
        with patch('app.nodes.llm_manager', mock_llm_manager):
            mock_llm = mock_llm_manager.get_llm("primary")
            mock_llm.ainvoke.side_effect = Exception("LLM error")
            
            result = await planning_node(state)
        
        assert result.current_step == "planning"
        assert result.research_plan is not None
        assert len(result.research_plan.steps) == 2  # Fallback has 2 steps
        assert len(result.errors) > 0


class TestSearchNode:
    """Test search node."""
    
    @pytest.mark.asyncio
    async def test_successful_search(self, sample_research_plan, sample_search_results, mock_search_manager):
        """Test successful search execution."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.research_plan = sample_research_plan
        
        with patch('app.nodes.search_manager', mock_search_manager):
            mock_search_manager.combined_search.return_value = sample_search_results
            
            result = await search_node(state)
        
        assert result.current_step == "search"
        assert len(result.search_results) > 0
        assert all(hasattr(r, 'relevance_score') for r in result.search_results)
    
    @pytest.mark.asyncio
    async def test_search_no_plan(self):
        """Test search with no research plan."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        # No research plan set
        
        result = await search_node(state)
        
        assert result.current_step == "search"
        assert len(result.errors) > 0
        assert "No research plan available" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, sample_research_plan, mock_search_manager):
        """Test search error handling."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.research_plan = sample_research_plan
        
        with patch('app.nodes.search_manager', mock_search_manager):
            mock_search_manager.combined_search.side_effect = Exception("Search API error")
            
            result = await search_node(state)
        
        assert result.current_step == "search"
        assert len(result.errors) > 0
        assert result.search_results == []


class TestContentFetchingNode:
    """Test content fetching node."""
    
    @pytest.mark.asyncio
    async def test_successful_content_fetching(self, sample_search_results, sample_source_contents, mock_content_fetcher):
        """Test successful content fetching."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.search_results = sample_search_results
        
        with patch('app.nodes.content_fetcher', mock_content_fetcher):
            mock_content_fetcher.fetch_multiple.return_value = sample_source_contents
            
            result = await content_fetching_node(state)
        
        assert result.current_step == "content_fetching"
        assert len(result.source_contents) == len(sample_source_contents)
    
    @pytest.mark.asyncio
    async def test_content_fetching_no_results(self):
        """Test content fetching with no search results."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        # No search results
        
        result = await content_fetching_node(state)
        
        assert result.current_step == "content_fetching"
        assert len(result.errors) > 0
        assert result.source_contents == []


class TestPerSourceSummarizationNode:
    """Test per-source summarization node."""
    
    @pytest.mark.asyncio
    async def test_successful_summarization(self, sample_source_contents, mock_llm_manager):
        """Test successful source summarization."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.source_contents = sample_source_contents
        
        summary_json = """{
            "url": "https://example.com/test",
            "title": "Test Article",
            "key_points": ["Point 1", "Point 2"],
            "relevance_explanation": "Relevant to topic",
            "credibility_assessment": "High credibility",
            "summary": "Test summary",
            "confidence_score": 0.8
        }"""
        
        with patch('app.nodes.llm_manager', mock_llm_manager):
            mock_llm = mock_llm_manager.get_llm("secondary")
            mock_llm.ainvoke.return_value = Mock(content=summary_json)
            
            with patch('app.nodes.PydanticOutputParser') as mock_parser:
                from app.models import SourceSummary
                mock_parser_instance = Mock()
                mock_parser.return_value = mock_parser_instance
                mock_parser_instance.parse.return_value = SourceSummary(
                    url="https://example.com/test",
                    title="Test Article",
                    key_points=["Point 1", "Point 2"],
                    relevance_explanation="Relevant to topic",
                    credibility_assessment="High credibility",
                    summary="Test summary",
                    confidence_score=0.8
                )
                
                result = await per_source_summarization_node(state)
        
        assert result.current_step == "per_source_summarization"
        assert len(result.source_summaries) > 0
    
    @pytest.mark.asyncio
    async def test_summarization_fallback(self, sample_search_results):
        """Test fallback to search results when no content available."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.search_results = sample_search_results
        # No source contents - should use search results
        
        result = await per_source_summarization_node(state)
        
        assert result.current_step == "per_source_summarization"
        assert len(result.source_summaries) > 0  # Should create fallback summaries


class TestSynthesisNode:
    """Test synthesis node."""
    
    @pytest.mark.asyncio
    async def test_successful_synthesis(self, sample_source_summaries, mock_llm_manager):
        """Test successful brief synthesis."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.source_summaries = sample_source_summaries
        
        brief_json = """{
            "title": "AI in Healthcare Brief",
            "executive_summary": "Test executive summary",
            "key_findings": ["Finding 1", "Finding 2"],
            "detailed_analysis": "Test detailed analysis",
            "implications": "Test implications",
            "limitations": "Test limitations",
            "references": [],
            "metadata": {
                "research_duration": 120,
                "total_sources_found": 3,
                "sources_used": 2,
                "confidence_score": 0.8,
                "depth_level": 2,
                "token_usage": {}
            }
        }"""
        
        with patch('app.nodes.llm_manager', mock_llm_manager):
            mock_llm = mock_llm_manager.get_llm("primary")
            mock_llm.ainvoke.return_value = Mock(content=brief_json)
            
            with patch('app.nodes.PydanticOutputParser') as mock_parser:
                from app.models import FinalBrief, BriefMetadata, Reference
                mock_parser_instance = Mock()
                mock_parser.return_value = mock_parser_instance
                mock_parser_instance.parse.return_value = FinalBrief(
                    title="AI in Healthcare Brief",
                    executive_summary="Test executive summary",
                    key_findings=["Finding 1", "Finding 2"],
                    detailed_analysis="Test detailed analysis",
                    implications="Test implications",
                    limitations="Test limitations",
                    references=[Reference(
                        title="Test Reference",
                        url="https://example.com",
                        relevance_note="Test relevance"
                    )],
                    metadata=BriefMetadata(
                        research_duration=120,
                        total_sources_found=3,
                        sources_used=2,
                        confidence_score=0.8,
                        depth_level=DepthLevel.STANDARD,
                        token_usage={}
                    )
                )
                
                result = await synthesis_node(state)
        
        assert result.current_step == "synthesis"
        assert result.final_brief is not None
        assert result.final_brief.title == "AI in Healthcare Brief"
    
    @pytest.mark.asyncio
    async def test_synthesis_no_summaries(self):
        """Test synthesis with no source summaries."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        # No source summaries
        
        result = await synthesis_node(state)
        
        assert result.current_step == "synthesis"
        assert len(result.errors) > 0
        assert result.final_brief is not None  # Should create fallback brief


class TestPostProcessingNode:
    """Test post-processing node."""
    
    @pytest.mark.asyncio
    async def test_successful_post_processing(self, sample_final_brief, mock_db_manager):
        """Test successful post-processing."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.final_brief = sample_final_brief
        
        with patch('app.nodes.db_manager', mock_db_manager):
            mock_db_manager.save_research_brief.return_value = "test_brief_id"
            
            result = await post_processing_node(state)
        
        assert result.current_step == "post_processing"
        assert result.final_brief is not None
        mock_db_manager.save_research_brief.assert_called_once()
        mock_db_manager.update_user_context_with_brief.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_post_processing_no_brief(self):
        """Test post-processing with no final brief."""
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        # No final brief
        
        result = await post_processing_node(state)
        
        assert result.current_step == "post_processing"
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_post_processing_validation(self, mock_db_manager):
        """Test post-processing validation and correction."""
        from app.models import FinalBrief, BriefMetadata, Reference
        
        # Create incomplete brief
        incomplete_brief = FinalBrief(
            title="",  # Empty title - should be corrected
            executive_summary="",  # Empty summary - should be corrected
            key_findings=[],  # Empty findings - should be corrected
            detailed_analysis="",
            implications="",
            limitations="",
            references=[],  # Empty references - should be corrected
            metadata=BriefMetadata(
                research_duration=60,
                total_sources_found=1,
                sources_used=1,
                confidence_score=0.5,
                depth_level=DepthLevel.STANDARD
            )
        )
        
        state = GraphState(
            topic="AI in Healthcare",
            depth=DepthLevel.STANDARD,
            follow_up=False,
            user_id="test_user"
        )
        state.final_brief = incomplete_brief
        
        with patch('app.nodes.db_manager', mock_db_manager):
            result = await post_processing_node(state)
        
        assert result.current_step == "post_processing"
        assert result.final_brief.title != ""  # Should be corrected
        assert len(result.final_brief.references) > 0  # Should have default reference
