"""
LangGraph workflow orchestration for the Research Brief Generator.
Implements the complete workflow with conditional transitions, retry logic, and checkpointing.
"""

from typing import Dict, Any, List
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models import GraphState, BriefRequest, FinalBrief, DepthLevel
from app.nodes import (
    context_summarization_node,
    planning_node,
    search_node,
    content_fetching_node,
    per_source_summarization_node,
    synthesis_node,
    post_processing_node
)
from app.config import config
from app.llm_tools_free import llm_manager


class ResearchWorkflow:
    """Main workflow orchestrator using LangGraph."""
    
    def __init__(self):
        self.graph = None
        self.memory = MemorySaver()
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("context_summarization", context_summarization_node)
        workflow.add_node("planning", planning_node)
        workflow.add_node("search", search_node)
        workflow.add_node("content_fetching", content_fetching_node)
        workflow.add_node("per_source_summarization", per_source_summarization_node)
        workflow.add_node("synthesis", synthesis_node)
        workflow.add_node("post_processing", post_processing_node)
        
        # Add conditional transitions
        
        # Start with context summarization
        workflow.set_entry_point("context_summarization")
        
        # Context summarization always goes to planning
        workflow.add_edge("context_summarization", "planning")
        
        # Planning always goes to search
        workflow.add_edge("planning", "search")
        
        # Search goes to content fetching if we have results
        workflow.add_conditional_edges(
            "search",
            self._should_fetch_content,
            {
                "fetch_content": "content_fetching",
                "skip_to_summarization": "per_source_summarization"
            }
        )
        
        # Content fetching always goes to summarization
        workflow.add_edge("content_fetching", "per_source_summarization")
        
        # Summarization goes to synthesis if we have summaries
        workflow.add_conditional_edges(
            "per_source_summarization",
            self._should_synthesize,
            {
                "synthesize": "synthesis",
                "retry_search": "search",
                "fail": END
            }
        )
        
        # Synthesis goes to post-processing
        workflow.add_edge("synthesis", "post_processing")
        
        # Post-processing is the end
        workflow.add_edge("post_processing", END)
        
        # Compile the graph with checkpointing
        self.graph = workflow.compile(checkpointer=self.memory)
    
    def _should_fetch_content(self, state: GraphState) -> str:
        """Decide whether to fetch full content or skip to summarization."""
        if state.search_results and len(state.search_results) > 0:
            return "fetch_content"
        else:
            return "skip_to_summarization"
    
    def _should_synthesize(self, state: GraphState) -> str:
        """Decide whether to proceed with synthesis, retry search, or fail."""
        
        # Check if we have enough summaries to proceed
        if state.source_summaries and len(state.source_summaries) >= 1:
            return "synthesize"
        
        # Check if we should retry search (max 1 retry)
        retry_count = state.retry_count.get("search", 0)
        if retry_count < 1 and state.search_results:
            state.retry_count["search"] = retry_count + 1
            return "retry_search"
        
        # If we have search results but no summaries, try to create minimal summaries
        if state.search_results:
            # Create basic summaries from search snippets
            from app.models import SourceSummary
            minimal_summaries = []
            for result in state.search_results[:3]:
                minimal_summaries.append(SourceSummary(
                    url=result.url,
                    title=result.title,
                    key_points=[result.snippet],
                    relevance_explanation="Basic summary from search result",
                    credibility_assessment="Not assessed",
                    summary=result.snippet,
                    confidence_score=0.4
                ))
            state.source_summaries = minimal_summaries
            return "synthesize"
        
        # If all else fails, we have to end
        return "fail"
    
    async def run_workflow(self, request: BriefRequest) -> FinalBrief:
        """
        Execute the complete research workflow.
        
        Args:
            request: Research brief request
            
        Returns:
            Final research brief
        """
        # Reset token usage tracking
        llm_manager.reset_token_usage()
        
        # Create initial state
        initial_state = GraphState(
            topic=request.topic,
            depth=request.depth,
            follow_up=request.follow_up,
            user_id=request.user_id,
            context=request.context
        )
        
        # Configuration for this run
        config_dict = {
            "configurable": {
                "thread_id": f"{request.user_id}_{datetime.utcnow().isoformat()}"
            }
        }
        
        try:
            # Run the workflow
            result = await self.graph.ainvoke(initial_state, config=config_dict)
            
            if result.final_brief:
                return result.final_brief
            else:
                # Create emergency fallback brief
                from app.models import BriefMetadata, Reference
                
                fallback_brief = FinalBrief(
                    title=f"Research Brief: {request.topic}",
                    executive_summary="Research workflow encountered errors and could not complete successfully. This is a minimal emergency response.",
                    key_findings=["Workflow execution failed", "No comprehensive analysis available"],
                    detailed_analysis="The research workflow encountered multiple errors and could not generate a comprehensive analysis.",
                    implications="Due to workflow failures, implications cannot be reliably determined.",
                    limitations="This brief has severe limitations due to workflow execution errors.",
                    references=[Reference(
                        title="Workflow Error",
                        url="",
                        access_date=datetime.utcnow(),
                        relevance_note="No sources were successfully processed"
                    )],
                    metadata=BriefMetadata(
                        research_duration=int((datetime.utcnow() - initial_state.start_time).total_seconds()),
                        total_sources_found=0,
                        sources_used=0,
                        confidence_score=0.1,
                        depth_level=request.depth,
                        token_usage=llm_manager.get_token_usage()
                    )
                )
                return fallback_brief
        
        except Exception as e:
            # Final fallback if entire workflow fails
            from app.models import BriefMetadata, Reference
            
            error_brief = FinalBrief(
                title=f"Research Brief: {request.topic}",
                executive_summary=f"Research workflow failed with error: {str(e)}",
                key_findings=["Workflow execution failed completely"],
                detailed_analysis=f"The research workflow encountered a critical error: {str(e)}",
                implications="Cannot provide implications due to workflow failure.",
                limitations="This brief is severely limited due to complete workflow failure.",
                references=[Reference(
                    title="Workflow Error",
                    url="",
                    access_date=datetime.utcnow(),
                    relevance_note="No sources were processed due to error"
                )],
                metadata=BriefMetadata(
                    research_duration=0,
                    total_sources_found=0,
                    sources_used=0,
                    confidence_score=0.0,
                    depth_level=request.depth,
                    token_usage={}
                )
            )
            return error_brief
    
    async def get_workflow_status(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow execution.
        
        Args:
            thread_id: Thread identifier for the workflow
            
        Returns:
            Status information
        """
        try:
            config_dict = {"configurable": {"thread_id": thread_id}}
            state = await self.graph.aget_state(config_dict)
            
            return {
                "current_step": state.values.get("current_step", "unknown"),
                "errors": state.values.get("errors", []),
                "progress": self._calculate_progress(state.values.get("current_step", "")),
                "start_time": state.values.get("start_time"),
                "retry_counts": state.values.get("retry_count", {})
            }
        except Exception as e:
            return {
                "error": f"Failed to get workflow status: {str(e)}",
                "current_step": "error",
                "progress": 0
            }
    
    def _calculate_progress(self, current_step: str) -> int:
        """Calculate workflow progress percentage based on current step."""
        steps = [
            "initialization",
            "context_summarization", 
            "planning",
            "search",
            "content_fetching",
            "per_source_summarization",
            "synthesis",
            "post_processing"
        ]
        
        if current_step in steps:
            return int((steps.index(current_step) + 1) / len(steps) * 100)
        return 0
    
    def get_graph_visualization(self) -> str:
        """
        Get a text-based visualization of the workflow graph.
        
        Returns:
            ASCII representation of the workflow
        """
        return """
Research Brief Generator Workflow:

┌─────────────────────┐
│   START REQUEST     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Context             │◄─── Skip if follow_up = false
│ Summarization       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Planning            │
│ (Generate Steps)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Search              │◄─── Retry once if needed
│ (Multi-source)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Content Fetching    │◄─── Skip if no results
│ (Full Text)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Per-Source          │
│ Summarization       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Synthesis           │
│ (Final Brief)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Post-Processing     │
│ (Validation & Save) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    END RESULT       │
└─────────────────────┘

Key Features:
- Conditional transitions based on data availability
- Retry logic for failed operations
- Checkpointing for resumable execution
- Comprehensive error handling
- Token usage tracking
- User context management
        """


# Global workflow instance
research_workflow = ResearchWorkflow()
