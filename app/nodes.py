"""
LangGraph workflow nodes for the Research Brief Generator.
Each node represents a distinct step in the research process using FREE alternatives.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser

from app.models import (
    GraphState, ResearchPlan, ResearchPlanStep, SourceSummary, SourceContent,
    FinalBrief, BriefMetadata, Reference, NodeResult, UserContext, DepthLevel
)
from .llm_tools_free import (
    generate_text_with_fallback,
    search_web_free,
    fetch_content_free,
    summarize_source_free,
    llm_manager,
    search_manager
)
from app.database import db_manager
from app.config import config

logger = logging.getLogger(__name__)


async def context_summarization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 1: Summarize user's previous research context if this is a follow-up query.
    """
    try:
        state = GraphState.model_validate(state)
        state.current_step = "context_summarization"
        
        if not state.follow_up:
            state.user_context = UserContext(user_id=state.user_id)
            return state.model_dump(mode="json")
        
        user_context = await db_manager.get_user_context(state.user_id)
        if not user_context:
            user_context = UserContext(user_id=state.user_id)
            state.user_context = user_context
            return state.model_dump(mode="json")
        
        if user_context.brief_summaries:
            llm = llm_manager.secondary_llm
            system_prompt = """You are an expert research assistant. Summarize the user's previous research context to inform the current research query.

Previous research topics: {topics}
Previous brief summaries: {summaries}

Create a concise context summary that highlights:
1. Key themes and patterns in previous research
2. Relevant background knowledge
3. Potential connections to the current topic: {current_topic}

Keep the summary under 500 words and focus on actionable insights."""

            human_prompt = f"""Current research topic: {state.topic}
Additional context: {state.context or 'None provided'}

Please provide a contextual summary that will help inform the current research."""

            messages = [
                SystemMessage(content=system_prompt.format(
                    topics=", ".join(user_context.previous_topics[-5:]),
                    summaries="\n".join(user_context.brief_summaries[-3:]),
                    current_topic=state.topic
                )),
                HumanMessage(content=human_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            user_context.preferences["last_context_summary"] = response.content
            user_context.last_updated = datetime.utcnow()
        
        state.user_context = user_context
        
    except Exception as e:
        logger.error(f"Context summarization failed: {e}")
        if not isinstance(state, GraphState):
            state = GraphState.model_validate(state)
        state.errors.append(f"Context summarization failed: {str(e)}")
        state.user_context = UserContext(user_id=state.user_id)
    
    return state.model_dump(mode="json")


async def planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Generate a structured research plan.
    """
    try:
        state = GraphState.model_validate(state)
        state.current_step = "planning"
        llm = llm_manager.primary_llm
        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        format_instructions = parser.get_format_instructions()
        
        system_prompt = """You are an expert research strategist. Create a research plan for the given topic.

Consider the research depth level:
- Level 1 (QUICK): 2 focused search queries.
- Level 2 (STANDARD): 3 comprehensive search queries.
- Level 3 (COMPREHENSIVE): 4 detailed search queries.
- Level 4 (EXHAUSTIVE): 5 extensive search queries.

For each research step, provide:
1. A specific, targeted search query
2. Clear rationale for why this step is needed
3. Expected number of quality sources (always 3)

{format_instructions}"""
        context_info = ""
        if state.user_context and state.user_context.preferences.get("last_context_summary"):
            context_info = f"\nPrevious research context:\n{state.user_context.preferences['last_context_summary']}"
        human_prompt = f"""Research Topic: {state.topic}
Depth Level: {state.depth.name} ({state.depth.value})
Additional Context: {state.context or 'None'}{context_info}

Create a systematic research plan that will thoroughly explore this topic."""
        messages = [
            SystemMessage(content=system_prompt.format(format_instructions=format_instructions)),
            HumanMessage(content=human_prompt)
        ]
        max_retries = config.node_configs["planning"]["max_retries"]
        for attempt in range(max_retries):
            try:
                response = await llm.ainvoke(messages)
                research_plan = parser.parse(response.content)
                state.research_plan = research_plan
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1)
        if not state.research_plan:
            raise ValueError("Failed to generate research plan after retries")
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        if not isinstance(state, GraphState):
            state = GraphState.model_validate(state)
        state.errors.append(f"Planning failed: {str(e)}")
        fallback_steps = [
            ResearchPlanStep(step_number=1, query=f"{state.topic} overview", rationale="Get general overview of the topic", expected_sources=3),
            ResearchPlanStep(step_number=2, query=f"{state.topic} recent developments", rationale="Find latest information and trends", expected_sources=3)
        ]
        state.research_plan = ResearchPlan(topic=state.topic, steps=fallback_steps, estimated_duration=30, depth_level=state.depth)
    return state.model_dump(mode="json")


async def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Execute search queries and collect sources.
    """
    try:
        state = GraphState.model_validate(state)
        state.current_step = "search"
        if not state.research_plan:
            raise ValueError("No research plan available for search")
        search_tasks = [
            search_manager.search_with_fallback(step.query)
            for step in state.research_plan.steps
        ]
        all_search_results = []
        max_concurrent = config.node_configs["search"]["concurrent_queries"]
        for i in range(0, len(search_tasks), max_concurrent):
            batch = search_tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            for result in batch_results:
                if isinstance(result, list):
                    all_search_results.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"A search task failed: {result}")
        seen_urls = set()
        unique_results = [
            r for r in all_search_results if r.url and r.url not in seen_urls and not seen_urls.add(r.url)
        ]
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        state.search_results = unique_results[:config.search.max_sources_per_brief]
        if not state.search_results:
            raise ValueError("No search results found")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        if not isinstance(state, GraphState):
            state = GraphState.model_validate(state)
        state.errors.append(f"Search failed: {str(e)}")
        state.search_results = []
    return state.model_dump(mode="json")


async def content_fetching_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 4: Fetch full content from selected sources.
    """
    try:
        state = GraphState.model_validate(state)
        state.current_step = "content_fetching"
        if not state.search_results:
            raise ValueError("No search results to fetch content from")
        urls_to_fetch = [result.url for result in state.search_results if result.url]
        source_map = {r.url: r.title for r in state.search_results}
        source_contents = []
        for url in urls_to_fetch[:config.search.max_sources_per_brief]:
            try:
                content = await fetch_content_free(url)
                if content and len(content.strip()) > 100:
                    source_contents.append(SourceContent(
                        url=url, title=source_map.get(url, "Unknown Title"), content=content,
                        fetch_timestamp=datetime.utcnow(), content_type="text/plain",
                        word_count=len(content.split()),
                        content_length=len(content)
                    ))
            except Exception as e:
                logger.warning(f"Failed to fetch content from {url}: {e}")
        state.source_contents = source_contents
        if not state.source_contents:
            raise ValueError("Failed to fetch significant content from any sources")
    except Exception as e:
        logger.error(f"Content fetching failed: {e}")
        if not isinstance(state, GraphState):
            state = GraphState.model_validate(state)
        state.errors.append(f"Content fetching failed: {str(e)}")
        state.source_contents = []
    return state.model_dump(mode="json")


async def per_source_summarization_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 5: Create structured summaries for each source.
    """
    try:
        state = GraphState.model_validate(state)
        state.current_step = "per_source_summarization"
        if not state.source_contents and not state.search_results:
            raise ValueError("No sources available for summarization")
        llm = llm_manager.secondary_llm
        parser = PydanticOutputParser(pydantic_object=SourceSummary)
        format_instructions = parser.get_format_instructions()
        system_prompt = """You are an expert research analyst. Create a structured summary of the given source content.

Focus on:
1. Key points relevant to the research topic: {topic}
2. Credibility and reliability of the source
3. How this source contributes to understanding the topic
4. Specific insights and data points

{format_instructions}"""
        sources_to_process = []
        if state.source_contents:
            for content in state.source_contents:
                sources_to_process.append({'url': content.url, 'title': content.title, 'text': content.content})
        else:
            for result in state.search_results:
                sources_to_process.append({'url': result.url, 'title': result.title, 'text': result.snippet})
        async def summarize_source(source_data):
            human_prompt = f"""Source Information:
Title: {source_data['title']}
URL: {source_data['url']}
Content: {source_data['text'][:5000]}...

Please provide a comprehensive structured summary for the research topic: "{state.topic}"."""
            messages = [
                SystemMessage(content=system_prompt.format(topic=state.topic, format_instructions=format_instructions)),
                HumanMessage(content=human_prompt)
            ]
            max_retries = config.node_configs["per_source_summarization"]["max_retries"]
            for attempt in range(max_retries):
                try:
                    response = await llm.ainvoke(messages)
                    return parser.parse(response.content)
                except Exception as e:
                    if attempt == max_retries - 1:
                        return SourceSummary(
                            url=source_data['url'], title=source_data['title'],
                            key_points=[source_data['text'][:200] + "..."],
                            relevance_explanation="Content analysis failed.",
                            credibility_assessment="Unable to assess.",
                            summary=source_data['text'][:500] + "...",
                            confidence_score=0.3
                        )
                    await asyncio.sleep(1)
        summaries = []
        max_concurrent = config.node_configs["per_source_summarization"]["concurrent_summaries"]
        for i in range(0, len(sources_to_process), max_concurrent):
            batch = sources_to_process[i:i + max_concurrent]
            batch_tasks = [summarize_source(source) for source in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            summaries.extend(r for r in batch_results if r is not None)
        state.source_summaries = summaries
        if not state.source_summaries:
            raise ValueError("Failed to create any source summaries")
    except Exception as e:
        logger.error(f"Source summarization failed: {e}")
        if not isinstance(state, GraphState):
            state = GraphState.model_validate(state)
        state.errors.append(f"Source summarization failed: {str(e)}")
        state.source_summaries = []
    return state.model_dump(mode="json")


async def synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 6: Synthesize all information into a final research brief.
    """
    try:
        state = GraphState.model_validate(state)
        state.current_step = "synthesis"
        if not state.source_summaries:
            raise ValueError("No source summaries available for synthesis")
        llm = llm_manager.primary_llm
        parser = PydanticOutputParser(pydantic_object=FinalBrief)
        format_instructions = parser.get_format_instructions()
        source_info = "\n\n".join([f"Source {i+1}: {s.title}..." for i, s in enumerate(state.source_summaries)])
        system_prompt = """You are an expert research analyst. Synthesize the provided source summaries into a professional, evidence-based research brief...{format_instructions}"""
        human_prompt = f"""Research Topic: {state.topic}...\n---{source_info}---"""
        messages = [
            SystemMessage(content=system_prompt.format(format_instructions=format_instructions)),
            HumanMessage(content=human_prompt)
        ]
        max_retries = config.node_configs["synthesis"]["max_retries"]
        for attempt in range(max_retries):
            try:
                response = await llm.ainvoke(messages)
                brief = parser.parse(response.content)
                state.final_brief = brief
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2)
        if not state.final_brief:
            raise ValueError("Failed to generate final brief after retries")
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        if not isinstance(state, GraphState):
            state = GraphState.model_validate(state)
        state.errors.append(f"Synthesis failed: {str(e)}")
        # FINAL FIX: Added all required fields to the fallback BriefMetadata
        state.final_brief = FinalBrief(
            title=f"Research Brief: {state.topic}",
            executive_summary="Synthesis failed.",
            key_findings=[],
            detailed_analysis="",
            implications="",
            limitations="",
            references=[],
            metadata=BriefMetadata(
                research_duration=0,
                total_sources_found=len(state.search_results),
                sources_used=0,
                confidence_score=0.1,
                depth_level=state.depth
            )
        )
    return state.model_dump(mode="json")


async def post_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 7: Final validation, formatting, and saving.
    """
    try:
        state = GraphState.model_validate(state)
        state.current_step = "post_processing"
        if not state.final_brief:
            # FINAL FIX: Added all required fields to the fallback BriefMetadata
            state.final_brief = FinalBrief(
                title=f"Research Brief: {state.topic}",
                executive_summary="Workflow failed before synthesis.",
                key_findings=[],
                detailed_analysis="",
                implications="",
                limitations="",
                references=[],
                metadata=BriefMetadata(
                    research_duration=0,
                    total_sources_found=0,
                    sources_used=0,
                    confidence_score=0.1,
                    depth_level=state.depth
                )
            )
        
        brief = state.final_brief
        if not brief.title: brief.title = f"Research Brief: {state.topic}"
        if not brief.references:
            brief.references = [
                Reference(title=summary.title, url=summary.url, access_date=datetime.utcnow(), relevance_note=summary.relevance_explanation[:150])
                for summary in state.source_summaries
            ] if state.source_summaries else [Reference(title="No sources processed", url="", access_date=datetime.utcnow(), relevance_note="Processing failed.")]
        
        end_time = datetime.utcnow()
        brief.metadata.research_duration = int((end_time - state.start_time).total_seconds())
        brief.metadata.token_usage = llm_manager.get_token_usage()
        brief.metadata.total_sources_found = len(state.search_results)
        brief.metadata.sources_used = len(state.source_summaries)
        brief.metadata.creation_timestamp = end_time
        
        brief_dict_for_db = brief.model_dump(mode='json')
        await db_manager.save_research_brief(brief_dict_for_db, state.user_id, state.topic)
        
        brief_summary = f"Topic: {state.topic}. Summary: {brief.executive_summary[:200]}..."
        await db_manager.update_user_context_with_brief(state.user_id, state.topic, brief_summary)
        
        state.final_brief = brief
    
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        if not isinstance(state, GraphState):
            state = GraphState.model_validate(state)
        state.errors.append(f"Post-processing failed: {str(e)}")
    
    return state.model_dump(mode="json")
