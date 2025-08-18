"""
LangGraph workflow nodes for the Research Brief Generator.
Each node represents a distinct step in the research process using FREE alternatives.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from app.models import (
    GraphState, ResearchPlan, ResearchPlanStep, SourceSummary, SourceContent,
    FinalBrief, BriefMetadata, Reference, UserContext
)
from .llm_tools_free import (
    generate_text_with_fallback,
    search_web_free,
    fetch_content_free,
    summarize_source_free,
    llm_manager
)
from app.database import db_manager

logger = logging.getLogger(__name__)


async def context_summarization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 1: Summarize user's previous research context if this is a follow-up query.
    """
    state = GraphState.model_validate(state)
    state.current_step = "context_summarization"
    logger.info("Executing context_summarization_node...")
    
    if not state.follow_up:
        state.user_context = UserContext(user_id=state.user_id)
        return state.model_dump(mode="json")
    
    try:
        user_context = await db_manager.get_user_context(state.user_id)
        if not user_context or not user_context.brief_summaries:
            state.user_context = UserContext(user_id=state.user_id)
            return state.model_dump(mode="json")

        llm = llm_manager.secondary_llm # Use the stronger model for context
        if not llm:
            raise ValueError("Secondary LLM (Gemini) not available for context summarization.")
            
        system_prompt = """You are an expert research assistant. Summarize the user's previous research context to inform the current research query.
Create a concise summary highlighting key themes, relevant background, and potential connections to the current topic: {current_topic}."""

        context_text = "\n".join(user_context.brief_summaries[-3:])
        human_prompt = f"Previous research summaries:\n{context_text}\n\nCurrent research topic: {state.topic}\n\nProvide a contextual summary to help inform the new research."

        messages = [SystemMessage(content=system_prompt.format(current_topic=state.topic)), HumanMessage(content=human_prompt)]
        
        response = await llm.ainvoke(messages)
        summary = response.content if hasattr(response, 'content') else str(response)
        user_context.preferences["last_context_summary"] = summary
        state.user_context = user_context

    except Exception as e:
        logger.error(f"Context summarization failed: {e}")
        state.errors.append(f"Context summarization failed: {str(e)}")
        state.user_context = UserContext(user_id=state.user_id) # Ensure a default context
    
    return state.model_dump(mode="json")


async def planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Generate a structured research plan.
    """
    state = GraphState.model_validate(state)
    state.current_step = "planning"
    logger.info("Executing planning_node...")

    try:
        llm = llm_manager.primary_llm # Use primary for planning
        if not llm:
            raise ValueError("Primary LLM (OpenRouter) not available for planning.")

        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        format_instructions = parser.get_format_instructions()
        
        system_prompt = f"""You are an expert research strategist. Create a research plan for the given topic.
- Depth Level {state.depth.value} ({state.depth.name}): Generate {state.depth.value + 1} focused search queries.
- For each step, provide a specific search query and a clear rationale.
{format_instructions}"""
        
        context_info = f"\nPrevious research context:\n{state.user_context.preferences.get('last_context_summary', 'None')}" if state.follow_up else ""
        human_prompt = f"Research Topic: {state.topic}{context_info}\n\nCreate a systematic research plan."
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        
        response_str = await generate_text_with_fallback(messages)
        research_plan = parser.parse(response_str)
        state.research_plan = research_plan

    except Exception as e:
        logger.error(f"Planning failed: {e}. Creating a fallback plan.")
        state.errors.append(f"Planning failed: {str(e)}")
        fallback_steps = [
            ResearchPlanStep(step_number=1, query=f"{state.topic} overview", rationale="Get a general overview."),
            ResearchPlanStep(step_number=2, query=f"{state.topic} recent developments", rationale="Find the latest information.")
        ]
        state.research_plan = ResearchPlan(topic=state.topic, steps=fallback_steps)
        
    return state.model_dump(mode="json")


async def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Execute search queries and collect sources.
    """
    state = GraphState.model_validate(state)
    state.current_step = "search"
    logger.info("Executing search_node...")

    try:
        if not state.research_plan:
            raise ValueError("No research plan available for search.")

        search_tasks = [search_web_free(step.query) for step in state.research_plan.steps]
        results_of_tasks = await asyncio.gather(*search_tasks)
        
        all_search_results = [item for sublist in results_of_tasks for item in sublist]

        seen_urls = set()
        unique_results = []
        for res in all_search_results:
            if res.get('href') and res['href'] not in seen_urls:
                unique_results.append(res)
                seen_urls.add(res['href'])
        
        state.search_results = unique_results[:10] # Limit to top 10 unique results
        
        if not state.search_results:
            raise ValueError("No search results found after executing the plan.")

    except Exception as e:
        logger.error(f"Search failed: {e}")
        state.errors.append(f"Search failed: {str(e)}")
        state.search_results = []

    return state.model_dump(mode="json")


async def content_fetching_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 4: Fetch full content from selected sources.
    """
    state = GraphState.model_validate(state)
    state.current_step = "content_fetching"
    logger.info("Executing content_fetching_node...")

    try:
        if not state.search_results:
            raise ValueError("No search results to fetch content from.")

        tasks = {fetch_content_free(result['href']): result for result in state.search_results[:5]} # Fetch top 5
        
        source_contents = []
        for future in asyncio.as_completed(tasks):
            result_meta = tasks[future]
            content = await future
            if content and len(content.strip()) > 100: # Basic content quality check
                source_contents.append(SourceContent(
                    url=result_meta['href'],
                    title=result_meta['title'],
                    content=content
                ))
        
        state.source_contents = source_contents
        if not state.source_contents:
            logger.warning("No significant content could be fetched from any sources.")

    except Exception as e:
        logger.error(f"Content fetching failed: {e}")
        state.errors.append(f"Content fetching failed: {str(e)}")
        state.source_contents = []

    return state.model_dump(mode="json")


async def per_source_summarization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 5: Create structured summaries for each source.
    """
    state = GraphState.model_validate(state)
    state.current_step = "per_source_summarization"
    logger.info("Executing per_source_summarization_node...")

    try:
        # Use fetched content if available, otherwise fall back to search snippets
        sources_to_process = state.source_contents if state.source_contents else state.search_results
        if not sources_to_process:
            raise ValueError("No sources available for summarization.")

        summarization_tasks = [
            summarize_source_free(
                content=source.content if isinstance(source, SourceContent) else source.get('body', ''),
                query=state.topic
            ) for source in sources_to_process
        ]
        
        summaries = await asyncio.gather(*summarization_tasks)
        
        # Add original source info back to the summary object
        for i, summary in enumerate(summaries):
            source = sources_to_process[i]
            summary.url = source.url if isinstance(source, SourceContent) else source.get('href')
            summary.title = source.title if isinstance(source, SourceContent) else source.get('title')

        state.source_summaries = [s for s in summaries if s.summary and "Error" not in s.summary]
        
        if not state.source_summaries:
            raise ValueError("Failed to create any valid source summaries.")

    except Exception as e:
        logger.error(f"Source summarization failed: {e}")
        state.errors.append(f"Source summarization failed: {str(e)}")
        state.source_summaries = []

    return state.model_dump(mode="json")


async def synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 6: Synthesize all information into a final research brief.
    """
    state = GraphState.model_validate(state)
    state.current_step = "synthesis"
    logger.info("Executing synthesis_node...")

    try:
        if not state.source_summaries:
            raise ValueError("No source summaries available for synthesis.")

        llm = llm_manager.secondary_llm # Use stronger model for synthesis
        if not llm:
            raise ValueError("Secondary LLM (Gemini) not available for synthesis.")

        parser = PydanticOutputParser(pydantic_object=FinalBrief)
        format_instructions = parser.get_format_instructions()
        
        summaries_text = "\n\n---\n\n".join(
            [f"Source Title: {s.title}\nURL: {s.url}\nSummary: {s.summary}" for s in state.source_summaries]
        )

        system_prompt = f"""You are an expert research analyst. Synthesize the provided source summaries into a professional, evidence-based research brief for the topic: "{state.topic}".
Your brief must include: an executive summary, key findings (3-5 bullet points), a detailed analysis, implications, and limitations.
Base your entire analysis STRICTLY on the provided source summaries. Do not invent information.
{format_instructions}"""
        
        human_prompt = f"Here are the source summaries to synthesize:\n\n{summaries_text}"
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        
        response_str = await generate_text_with_fallback(messages)
        brief = parser.parse(response_str)
        state.final_brief = brief

    except Exception as e:
        logger.error(f"Synthesis failed: {e}. Creating a fallback brief.")
        state.errors.append(f"Synthesis failed: {str(e)}")
        state.final_brief = FinalBrief(
            title=f"Failed Research Brief: {state.topic}",
            executive_summary="The synthesis process failed to generate a brief. This may be due to errors in the source material or model failures.",
            key_findings=["Synthesis process failed."],
            detailed_analysis="Could not be generated.",
            implications="Could not be determined.",
            limitations="The primary limitation is the failure of the synthesis AI model.",
            references=[],
            metadata=BriefMetadata(research_duration=0, total_sources_found=0, sources_used=0, confidence_score=0.1, depth_level=state.depth.value)
        )
        
    return state.model_dump(mode="json")


async def post_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 7: Final validation, formatting, and saving.
    """
    state = GraphState.model_validate(state)
    state.current_step = "post_processing"
    logger.info("Executing post_processing_node...")

    try:
        if not state.final_brief:
            raise ValueError("No final brief available for post-processing.")
        
        brief = state.final_brief
        
        # Populate references from source summaries
        brief.references = [
            Reference(
                title=summary.title,
                url=summary.url,
                relevance_note=summary.summary[:150] + "..."
            ) for summary in state.source_summaries
        ]
        
        # Populate metadata
        end_time = datetime.utcnow()
        duration = int((end_time - state.start_time).total_seconds())
        
        brief.metadata = BriefMetadata(
            research_duration=duration,
            total_sources_found=len(state.search_results),
            sources_used=len(state.source_summaries),
            confidence_score=0.85, # Example score
            depth_level=state.depth.value,
            token_usage={} # Placeholder
        )
        
        # Save to database
        await db_manager.save_brief(brief, state.user_id)
        
        # Update user context
        brief_summary_for_context = f"Topic: {brief.title}. Summary: {brief.executive_summary[:200]}..."
        await db_manager.update_user_context_with_brief(state.user_id, brief.title, brief_summary_for_context)
        
        state.final_brief = brief

    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        state.errors.append(f"Post-processing failed: {str(e)}")

    return state.model_dump(mode="json")