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


async def context_summarization_node(state: GraphState) -> GraphState:
    """
    Node 1: Summarize user's previous research context if this is a follow-up query.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with user context
    """
    start_time = datetime.utcnow()
    state.current_step = "context_summarization"
    
    try:
        if not state.follow_up:
            # Skip context summarization for new queries
            state.user_context = UserContext(user_id=state.user_id)
            return state
        
        # Retrieve user's historical context
        user_context = await db_manager.get_user_context(state.user_id)
        if not user_context:
            user_context = UserContext(user_id=state.user_id)
            state.user_context = user_context
            return state
        
        # If user has previous briefs, summarize the context
        if user_context.brief_summaries:
            llm = llm_manager.get_llm("secondary")  # Use Claude for summarization
            
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
                    topics=", ".join(user_context.previous_topics[-5:]),  # Last 5 topics
                    summaries="\n".join(user_context.brief_summaries[-3:]),  # Last 3 summaries
                    current_topic=state.topic
                )),
                HumanMessage(content=human_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            
            # Update user context with contextual summary
            user_context.preferences["last_context_summary"] = response.content
            user_context.last_updated = datetime.utcnow()
        
        state.user_context = user_context
        
    except Exception as e:
        state.errors.append(f"Context summarization failed: {str(e)}")
        # Create empty context to continue
        state.user_context = UserContext(user_id=state.user_id)
    
    return state


async def planning_node(state: GraphState) -> GraphState:
    """
    Node 2: Generate a structured research plan based on the topic and context.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with research plan
    """
    start_time = datetime.utcnow()
    state.current_step = "planning"
    
    try:
        llm = llm_manager.get_llm("primary")  # Use GPT-4 for planning
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        format_instructions = parser.get_format_instructions()
        
        system_prompt = """You are an expert research strategist. Create a comprehensive research plan for the given topic.

Consider the research depth level:
- QUICK (1): 2-3 focused queries, 15-20 minutes
- STANDARD (2): 4-5 comprehensive queries, 30-45 minutes  
- COMPREHENSIVE (3): 6-8 detailed queries, 60-90 minutes
- EXHAUSTIVE (4): 10+ extensive queries, 120+ minutes

For each research step, provide:
1. A specific, targeted search query
2. Clear rationale for why this step is needed
3. Expected number of quality sources

{format_instructions}"""

        # Build context information
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
                await asyncio.sleep(1)  # Brief delay before retry
        
        if not state.research_plan:
            raise ValueError("Failed to generate research plan after retries")
    
    except Exception as e:
        state.errors.append(f"Planning failed: {str(e)}")
        # Create fallback plan
        fallback_steps = [
            ResearchPlanStep(
                step_number=1,
                query=f"{state.topic} overview",
                rationale="Get general overview of the topic",
                expected_sources=3
            ),
            ResearchPlanStep(
                step_number=2,
                query=f"{state.topic} recent developments",
                rationale="Find latest information and trends",
                expected_sources=3
            )
        ]
        state.research_plan = ResearchPlan(
            topic=state.topic,
            steps=fallback_steps,
            estimated_duration=30,
            depth_level=state.depth
        )
    
    return state


async def search_node(state: GraphState) -> GraphState:
    """
    Node 3: Execute search queries and collect sources.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with search results
    """
    start_time = datetime.utcnow()
    state.current_step = "search"
    
    try:
        if not state.research_plan:
            raise ValueError("No research plan available for search")
        
        all_search_results = []
        
        # Execute searches for each step in the plan
        search_tasks = []
        for step in state.research_plan.steps:
            task = search_manager.combined_search(
                step.query, 
                max_results=step.expected_sources
            )
            search_tasks.append(task)
        
        # Run searches with controlled concurrency
        max_concurrent = config.node_configs["search"]["concurrent_queries"]
        for i in range(0, len(search_tasks), max_concurrent):
            batch = search_tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, list):
                    all_search_results.extend(result)
        
        # Deduplicate and rank results
        seen_urls = set()
        unique_results = []
        
        for result in all_search_results:
            if result.url not in seen_urls and result.url:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Sort by relevance and limit to max sources per brief
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        state.search_results = unique_results[:config.search.max_sources_per_brief]
        
        if not state.search_results:
            raise ValueError("No search results found")
    
    except Exception as e:
        state.errors.append(f"Search failed: {str(e)}")
        # Continue with empty results - will be handled in later nodes
        state.search_results = []
    
    return state


async def content_fetching_node(state: GraphState) -> GraphState:
    """
    Node 4: Fetch full content from selected sources.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with source contents
    """
    start_time = datetime.utcnow()
    state.current_step = "content_fetching"
    
    try:
        if not state.search_results:
            raise ValueError("No search results available for content fetching")
        
        # Extract URLs from search results
        urls = [result.url for result in state.search_results if result.url]
        
        # Fetch content from URLs using free tools
        source_contents = []
        for url in urls[:5]:  # Limit to 5 URLs for free tier
            try:
                content = await fetch_content_free(url)
                if content and len(content.strip()) > 100:  # Only keep substantial content
                    source_content = SourceContent(
                        url=url,
                        title=next((r.title for r in state.search_results if r.url == url), ""),
                        content=content,
                        fetch_timestamp=datetime.now(),
                        content_type="text/html",
                        word_count=len(content.split())
                    )
                    source_contents.append(source_content)
            except Exception as e:
                logger.warning(f"Failed to fetch content from {url}: {e}")
        
        state.source_contents = source_contents
        
        if not state.source_contents:
            raise ValueError("Failed to fetch content from any sources")
    
    except Exception as e:
        state.errors.append(f"Content fetching failed: {str(e)}")
        # Continue with empty content - will use search snippets as fallback
        state.source_contents = []
    
    return state


async def per_source_summarization_node(state: GraphState) -> GraphState:
    """
    Node 5: Create structured summaries for each source.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with source summaries
    """
    start_time = datetime.utcnow()
    state.current_step = "per_source_summarization"
    
    try:
        if not state.source_contents and not state.search_results:
            raise ValueError("No sources available for summarization")
        
        llm = llm_manager.get_llm("secondary")  # Use Claude for summarization
        parser = PydanticOutputParser(pydantic_object=SourceSummary)
        format_instructions = parser.get_format_instructions()
        
        system_prompt = """You are an expert research analyst. Create a structured summary of the given source content.

Focus on:
1. Key points relevant to the research topic
2. Credibility and reliability of the source
3. How this source contributes to understanding the topic
4. Specific insights and data points

{format_instructions}"""

        # Create summarization tasks
        summarization_tasks = []
        sources_to_process = []
        
        # Use full content if available, otherwise use search snippets
        if state.source_contents:
            for content in state.source_contents:
                sources_to_process.append({
                    'url': content.url,
                    'title': content.title,
                    'text': content.content
                })
        else:
            # Fallback to search result snippets
            for result in state.search_results:
                sources_to_process.append({
                    'url': result.url,
                    'title': result.title,
                    'text': result.snippet
                })
        
        async def summarize_source(source_data):
            human_prompt = f"""Research Topic: {state.topic}

Source Information:
Title: {source_data['title']}
URL: {source_data['url']}
Content: {source_data['text'][:5000]}...  

Please provide a comprehensive structured summary."""

            messages = [
                SystemMessage(content=system_prompt.format(format_instructions=format_instructions)),
                HumanMessage(content=human_prompt)
            ]
            
            max_retries = config.node_configs["per_source_summarization"]["max_retries"]
            for attempt in range(max_retries):
                try:
                    response = await llm.ainvoke(messages)
                    return parser.parse(response.content)
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Create fallback summary
                        return SourceSummary(
                            url=source_data['url'],
                            title=source_data['title'],
                            key_points=[source_data['text'][:200] + "..."],
                            relevance_explanation="Content analysis failed, using raw text",
                            credibility_assessment="Unable to assess",
                            summary=source_data['text'][:500] + "...",
                            confidence_score=0.3
                        )
                    await asyncio.sleep(1)
        
        # Process sources with controlled concurrency
        max_concurrent = config.node_configs["per_source_summarization"]["concurrent_summaries"]
        summaries = []
        
        for i in range(0, len(sources_to_process), max_concurrent):
            batch = sources_to_process[i:i + max_concurrent]
            batch_tasks = [summarize_source(source) for source in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, SourceSummary):
                    summaries.append(result)
        
        state.source_summaries = summaries
        
        if not state.source_summaries:
            raise ValueError("Failed to create any source summaries")
    
    except Exception as e:
        state.errors.append(f"Source summarization failed: {str(e)}")
        # Create minimal fallback summaries
        fallback_summaries = []
        for result in state.search_results[:3]:  # Use first 3 search results
            fallback_summaries.append(SourceSummary(
                url=result.url,
                title=result.title,
                key_points=[result.snippet],
                relevance_explanation="Fallback summary from search snippet",
                credibility_assessment="Not assessed",
                summary=result.snippet,
                confidence_score=0.5
            ))
        state.source_summaries = fallback_summaries
    
    return state


async def synthesis_node(state: GraphState) -> GraphState:
    """
    Node 6: Synthesize all information into a final research brief.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with final brief
    """
    start_time = datetime.utcnow()
    state.current_step = "synthesis"
    
    try:
        if not state.source_summaries:
            raise ValueError("No source summaries available for synthesis")
        
        llm = llm_manager.get_llm("primary")  # Use GPT-4 for synthesis
        parser = PydanticOutputParser(pydantic_object=FinalBrief)
        format_instructions = parser.get_format_instructions()
        
        # Calculate execution time and metadata
        execution_time = (datetime.utcnow() - state.start_time).total_seconds()
        token_usage = llm_manager.get_token_usage()
        
        # Create metadata
        metadata = BriefMetadata(
            creation_timestamp=datetime.utcnow(),
            research_duration=int(execution_time),
            total_sources_found=len(state.search_results),
            sources_used=len(state.source_summaries),
            confidence_score=sum(s.confidence_score for s in state.source_summaries) / len(state.source_summaries),
            depth_level=state.depth,
            token_usage=token_usage
        )
        
        system_prompt = """You are an expert research analyst creating a comprehensive research brief. 

Synthesize the provided source summaries into a well-structured, professional research brief that provides:
1. Clear, actionable insights
2. Evidence-based analysis
3. Balanced perspective acknowledging limitations
4. Professional tone suitable for decision-makers

{format_instructions}"""

        # Compile source information
        source_info = ""
        for i, summary in enumerate(state.source_summaries, 1):
            source_info += f"\nSource {i}:\nTitle: {summary.title}\nURL: {summary.url}\n"
            source_info += f"Key Points: {', '.join(summary.key_points)}\n"
            source_info += f"Summary: {summary.summary}\n"
            source_info += f"Relevance: {summary.relevance_explanation}\n"
            source_info += f"Credibility: {summary.credibility_assessment}\n"
            source_info += f"Confidence: {summary.confidence_score}\n"
            source_info += "-" * 50
        
        human_prompt = f"""Research Topic: {state.topic}
Research Depth: {state.depth.name}

Source Summaries:
{source_info}

Additional Metadata to Include:
{metadata.dict()}

Please create a comprehensive research brief that synthesizes all this information."""

        messages = [
            SystemMessage(content=system_prompt.format(format_instructions=format_instructions)),
            HumanMessage(content=human_prompt)
        ]
        
        max_retries = config.node_configs["synthesis"]["max_retries"]
        for attempt in range(max_retries):
            try:
                response = await llm.ainvoke(messages)
                brief = parser.parse(response.content)
                
                # Override metadata with calculated values
                brief.metadata = metadata
                
                # Ensure references are created
                if not brief.references:
                    brief.references = [
                        Reference(
                            title=summary.title,
                            url=summary.url,
                            access_date=datetime.utcnow(),
                            relevance_note=summary.relevance_explanation[:100] + "..."
                        )
                        for summary in state.source_summaries
                    ]
                
                state.final_brief = brief
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2)
        
        if not state.final_brief:
            raise ValueError("Failed to generate final brief after retries")
    
    except Exception as e:
        state.errors.append(f"Synthesis failed: {str(e)}")
        # Create fallback brief
        state.final_brief = FinalBrief(
            title=f"Research Brief: {state.topic}",
            executive_summary="Research synthesis failed. This is a minimal fallback brief.",
            key_findings=["Research process encountered errors", "Limited analysis available"],
            detailed_analysis="Due to processing errors, detailed analysis is not available.",
            implications="Unable to provide comprehensive implications due to synthesis failure.",
            limitations="This brief has significant limitations due to processing errors.",
            references=[Reference(
                title=summary.title,
                url=summary.url,
                access_date=datetime.utcnow(),
                relevance_note="Source from failed synthesis"
            ) for summary in state.source_summaries[:3]],
            metadata=BriefMetadata(
                research_duration=int((datetime.utcnow() - state.start_time).total_seconds()),
                total_sources_found=len(state.search_results),
                sources_used=len(state.source_summaries),
                confidence_score=0.3,
                depth_level=state.depth,
                token_usage=llm_manager.get_token_usage()
            )
        )
    
    return state


async def post_processing_node(state: GraphState) -> GraphState:
    """
    Node 7: Final validation, formatting, and cleanup.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated graph state with validated final brief
    """
    start_time = datetime.utcnow()
    state.current_step = "post_processing"
    
    try:
        if not state.final_brief:
            raise ValueError("No final brief available for post-processing")
        
        # Validate the brief structure
        brief = state.final_brief
        
        # Ensure all required fields are present and valid
        if not brief.title:
            brief.title = f"Research Brief: {state.topic}"
        
        if not brief.executive_summary:
            brief.executive_summary = "Executive summary not available."
        
        if not brief.key_findings:
            brief.key_findings = ["Key findings not available."]
        
        if not brief.detailed_analysis:
            brief.detailed_analysis = "Detailed analysis not available."
        
        if not brief.implications:
            brief.implications = "Implications analysis not available."
        
        if not brief.limitations:
            brief.limitations = "Research limitations not documented."
        
        if not brief.references:
            brief.references = [Reference(
                title="No sources available",
                url="",
                access_date=datetime.utcnow(),
                relevance_note="No sources were successfully processed"
            )]
        
        # Update final metadata
        brief.metadata.research_duration = int((datetime.utcnow() - state.start_time).total_seconds())
        brief.metadata.token_usage = llm_manager.get_token_usage()
        
        # Save the brief to database
        brief_id = await db_manager.save_research_brief(brief, state.user_id, state.topic)
        
        # Update user context
        brief_summary = f"Topic: {state.topic}. Summary: {brief.executive_summary[:200]}..."
        await db_manager.update_user_context_with_brief(state.user_id, state.topic, brief_summary)
        
        state.final_brief = brief
    
    except Exception as e:
        state.errors.append(f"Post-processing failed: {str(e)}")
        # Brief should still be usable even if post-processing fails
    
    return state
