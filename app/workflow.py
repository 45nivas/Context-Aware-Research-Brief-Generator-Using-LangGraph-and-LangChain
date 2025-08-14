"""
LangGraph workflow orchestration for the Research Brief Generator.
Implements the complete workflow with conditional transitions, retry logic, and checkpointing.
"""

from typing import Dict, Any
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models import GraphState, BriefRequest, FinalBrief, BriefMetadata, Reference
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
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("context_summarization", context_summarization_node)
        workflow.add_node("planning", planning_node)
        workflow.add_node("search", search_node)
        workflow.add_node("content_fetching", content_fetching_node)
        workflow.add_node("per_source_summarization", per_source_summarization_node)
        workflow.add_node("synthesis", synthesis_node)
        workflow.add_node("post_processing", post_processing_node)

        # Set entry and edges
        workflow.set_entry_point("context_summarization")
        workflow.add_edge("context_summarization", "planning")
        workflow.add_edge("planning", "search")
        workflow.add_edge("content_fetching", "per_source_summarization")
        workflow.add_edge("synthesis", "post_processing")
        workflow.add_edge("post_processing", END)

        # Add conditional transitions
        workflow.add_conditional_edges(
            "search",
            self._should_fetch_content,
            {
                "fetch_content": "content_fetching",
                "skip_to_summarization": "per_source_summarization"
            }
        )
        workflow.add_conditional_edges(
            "per_source_summarization",
            self._should_synthesize,
            {
                "synthesize": "synthesis",
                "retry_search": "search",
                "fail": END
            }
        )

        # Compile the graph with checkpointing
        self.graph = workflow.compile(checkpointer=self.memory)

    def _should_fetch_content(self, state: GraphState) -> str:
        """Decide whether to fetch full content based on search results."""
        if state.search_results and len(state.search_results) > 0:
            return "fetch_content"
        else:
            return "skip_to_summarization"

    def _should_synthesize(self, state: GraphState) -> str:
        """Decide whether to proceed with synthesis, retry search, or fail."""
        if state.source_summaries and len(state.source_summaries) >= 1:
            return "synthesize"

        retry_count = state.retry_count.get("search", 0) if state.retry_count else 0
        if retry_count < 1 and state.search_results:
            # Note: Modifying state in a conditional edge is an anti-pattern.
            # This logic should ideally be in its own node if it needs to update the retry_count.
            return "retry_search"

        return "fail"

    async def run_workflow(self, request: BriefRequest) -> FinalBrief:
        """Execute the complete research workflow."""
        llm_manager.reset_token_usage()

        start_time = datetime.utcnow()
        initial_state = {
            "topic": request.topic,
            "depth": request.depth,
            "follow_up": request.follow_up,
            "user_id": request.user_id,
            "context": request.context,
            "start_time": start_time,
            "errors": [],
            "research_plan": None,
            "search_results": [],
            "source_contents": [],
            "source_summaries": [],
            "final_brief": None,
            "user_context": None,
            "retry_count": {"search": 0},
            "current_step": "initializing"
        }

        config_dict = {"configurable": {"thread_id": f"{request.user_id}_{start_time.isoformat()}"}}

        try:
            # Using ainvoke() is simpler and directly returns the final state dictionary.
            final_state = await self.graph.ainvoke(initial_state, config=config_dict)

            if final_state and final_state.get("final_brief"):
                return final_state["final_brief"]
            else:
                # This fallback will now have more information.
                duration = int((datetime.utcnow() - start_time).total_seconds())
                error_message = f"Workflow ended but the 'final_brief' was not populated. Final state: {final_state}"

                fallback_brief = FinalBrief(
                    title=f"Research Brief: {request.topic}",
                    executive_summary="Workflow completed but no valid brief was generated.",
                    key_findings=["The research process finished without producing a final document."],
                    detailed_analysis=error_message,
                    implications="Implications could not be determined.",
                    limitations="This brief is severely limited due to an incomplete workflow.",
                    references=[Reference(title="Workflow Error", url="", access_date=datetime.utcnow(), relevance_note="No sources were processed")],
                    metadata=BriefMetadata(
                        research_duration=duration,
                        total_sources_found=len(final_state.get("search_results", []) if final_state else []),
                        sources_used=len(final_state.get("source_summaries", []) if final_state else []),
                        confidence_score=0.1,
                        depth_level=request.depth,
                        token_usage=llm_manager.get_token_usage()
                    )
                )
                return fallback_brief

        except Exception as e:
            # This handles crashes during the workflow execution itself.
            error_brief = FinalBrief(
                title=f"Research Brief: {request.topic}",
                executive_summary=f"Research workflow failed with a critical error: {str(e)}",
                key_findings=["Workflow execution failed completely"],
                detailed_analysis=f"The research workflow encountered a critical error: {str(e)}",
                implications="Cannot provide implications due to workflow failure.",
                limitations="This brief is severely limited due to complete workflow failure.",
                references=[Reference(title="Workflow Error", url="", access_date=datetime.utcnow(), relevance_note="No sources were processed due to error")],
                metadata=BriefMetadata(
                    research_duration=int((datetime.utcnow() - start_time).total_seconds()),
                    total_sources_found=0,
                    sources_used=0,
                    confidence_score=0.0,
                    depth_level=request.depth,
                    token_usage={}
                )
            )
            return error_brief

    async def get_workflow_status(self, thread_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow execution."""
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
            return {"error": f"Failed to get workflow status: {str(e)}"}

    def _calculate_progress(self, current_step: str) -> int:
        """Calculate workflow progress percentage based on current step."""
        steps = ["initialization", "context_summarization", "planning", "search", "content_fetching", "per_source_summarization", "synthesis", "post_processing"]
        if current_step in steps:
            return int((steps.index(current_step) + 1) / len(steps) * 100)
        return 0

    def get_graph_visualization(self) -> str:
        """Get a text-based visualization of the workflow graph."""
        return "Graph visualization is available at the /workflow/graph endpoint."


# Global workflow instance
research_workflow = ResearchWorkflow()