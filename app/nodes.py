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
        
        # Enhanced topic-specific search strategy
        topic_lower = state.topic.lower()
        domain_context = ""
        search_modifiers = []
        
        # Detect topic domain and add specific context
        if "artificial intelligence" in topic_lower or "ai" in topic_lower:
            if "education" in topic_lower or "learning" in topic_lower or "teaching" in topic_lower:
                domain_context = "AI/ML in Education Research"
                search_modifiers = [
                    '"artificial intelligence in education" OR "AI in education"',
                    '"machine learning classroom" OR "AI tutoring systems"',
                    '"educational technology" AND "artificial intelligence"',
                    '"AI-powered learning" OR "intelligent tutoring"',
                    '"personalized learning AI" OR "adaptive learning systems"'
                ]
            else:
                domain_context = "Artificial Intelligence Research"
                search_modifiers = [
                    '"artificial intelligence" research applications',
                    '"machine learning" algorithms implementation',
                    '"deep learning" systems development',
                    '"neural networks" practical applications',
                    '"AI ethics" AND "responsible AI"'
                ]
        elif "machine learning" in topic_lower:
            domain_context = "Machine Learning Research"
            search_modifiers = [
                '"machine learning" algorithms research',
                '"supervised learning" OR "unsupervised learning"',
                '"predictive modeling" case studies',
                '"data science" machine learning applications',
                '"ML model deployment" best practices'
            ]
        else:
            # Generic academic research approach
            domain_context = "Academic Research"
            search_modifiers = [
                f'"{state.topic}" research study',
                f'"{state.topic}" academic analysis',
                f'"{state.topic}" case study implementation',
                f'"{state.topic}" empirical research findings'
            ]

        system_prompt = f"""You are an expert research strategist specializing in {domain_context}. Create a research plan for the given topic that will find substantive academic and research content.

CRITICAL REQUIREMENTS:
- Topic: {state.topic}
- Domain Context: {domain_context}
- Generate {state.depth.value + 1} HIGHLY SPECIFIC search queries focused on research and academic content
- Each query must target scholarly articles, research papers, case studies, and implementation reports
- AVOID dictionary definitions, general overviews, and basic explanations

ENHANCED SEARCH STRATEGY:
- Use domain-specific terminology and academic language
- Include research indicators: "study", "research", "analysis", "implementation", "case study"
- Target academic sources: "academic", "journal", "conference", "proceedings", "publication"
- Use quoted phrases for exact matches on key concepts
- Combine multiple relevant terms with AND/OR operators

SUGGESTED SEARCH PATTERNS:
{chr(10).join(f"- {modifier}" for modifier in search_modifiers)}

QUALITY REQUIREMENTS:
- Each query should target different aspects of the research topic
- Prioritize peer-reviewed and academic sources
- Include both theoretical and practical implementation perspectives
- Focus on recent developments and current research trends

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
            url = res.get('href', res.get('url', ''))
            if url and url not in seen_urls:
                # Create SearchResult objects from the raw results
                from app.models import SearchResult
                search_result = SearchResult(
                    title=res.get('title', 'Untitled'),
                    url=url,
                    snippet=res.get('body', res.get('snippet', '')),
                    relevance_score=0.8  # Default relevance
                )
                unique_results.append(search_result)
                seen_urls.add(url)
        
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

        tasks = []
        for result in state.search_results[:5]:  # Fetch top 5
            url = result.url if hasattr(result, 'url') else result.get('href', '')
            if url:
                tasks.append(fetch_content_free(url))
        
        if not tasks:
            raise ValueError("No valid URLs found to fetch content from.")
        
        contents = await asyncio.gather(*tasks)
        
        source_contents = []
        for i, content in enumerate(contents):
            if content and len(content.strip()) > 100:  # Basic content quality check
                result = state.search_results[i]
                url = result.url if hasattr(result, 'url') else result.get('href', '')
                title = result.title if hasattr(result, 'title') else result.get('title', 'Untitled')
                
                source_contents.append(SourceContent(
                    url=url,
                    title=title,
                    content=content,
                    content_length=len(content)
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

        summarization_tasks = []
        for source in sources_to_process:
            if isinstance(source, SourceContent):
                content_text = source.content
            else:
                # Handle SearchResult objects
                content_text = source.snippet if hasattr(source, 'snippet') else source.get('body', '')
            
            summarization_tasks.append(summarize_source_free(content_text, state.topic))
        
        summaries = await asyncio.gather(*summarization_tasks)
        
        # Add original source info back to the summary object
        for i, summary in enumerate(summaries):
            source = sources_to_process[i]
            if isinstance(source, SourceContent):
                summary.url = source.url
                summary.title = source.title
            else:
                # Handle SearchResult objects
                summary.url = source.url if hasattr(source, 'url') else source.get('href', '')
                summary.title = source.title if hasattr(source, 'title') else source.get('title', 'Untitled')

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
        
        summaries_text = "\n\n---\n\n".join(
            [f"Source Title: {s.title}\nURL: {s.url}\nSummary: {s.summary}" for s in state.source_summaries]
        )

        system_prompt = f"""You are an expert research analyst. Synthesize the provided source summaries into a comprehensive research brief about: "{state.topic}".

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON data (no markdown, no schema definitions)
2. Base your analysis strictly on the provided sources
3. Create actual content, not placeholder text
4. Ensure all required fields are present and properly formatted

Required JSON structure with ACTUAL content:"""

        human_prompt = f"""Sources to analyze:
{summaries_text}

Generate a JSON research brief with:
- title: Clear, specific title about {state.topic}
- executive_summary: 2-3 sentence summary of key insights
- key_findings: Array of 3-5 specific findings from the sources
- detailed_analysis: Thorough analysis based on source content
- implications: What these findings mean for stakeholders
- limitations: Research limitations and data gaps
- references: Array of source objects with title, url, relevance_note (required!)
- metadata: Object with exact numeric values:
  * research_duration: integer seconds (e.g., 1800)
  * total_sources_found: integer count (e.g., 5)
  * sources_used: integer count (e.g., 3)
  * confidence_score: float 0.0-1.0 (e.g., 0.85)
  * depth_level: integer 1-4 only (e.g., 2)

CRITICAL: 
1. metadata must use exact numeric formats - no strings like "1 hour" or "moderate"
2. Each reference MUST include relevance_note field explaining why the source is relevant

Example reference format:
"references": [
  {{
    "title": "Source Title Here",
    "url": "https://example.com/source",
    "relevance_note": "This source provides key insights about [specific aspect] relevant to {state.topic}"
  }}
]

Example metadata format:
"metadata": {{
  "research_duration": 2400,
  "total_sources_found": 6,
  "sources_used": 4,
  "confidence_score": 0.8,
  "depth_level": 2
}}

Return ONLY the JSON object, no other text."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        
        response_str = await generate_text_with_fallback(messages)
        
        # Try to parse the response
        try:
            brief = parser.parse(response_str)
        except Exception as parse_error:
            logger.warning(f"Failed to parse LLM response: {parse_error}. Attempting manual JSON extraction.")
            # Try to extract JSON from the response manually
            import json
            import re
            
            # Look for JSON content between ```json and ```
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_str, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(1))
                    # Validate and create FinalBrief from JSON
                    brief = FinalBrief.model_validate(json_data)
                except Exception as json_error:
                    logger.error(f"Failed to parse extracted JSON: {json_error}")
                    raise parse_error
            else:
                # Try direct JSON parsing
                try:
                    json_data = json.loads(response_str)
                    brief = FinalBrief.model_validate(json_data)
                except Exception:
                    raise parse_error
        
        state.final_brief = brief

    except Exception as e:
        logger.error(f"Synthesis failed: {e}. Creating a fallback brief.")
        state.errors.append(f"Synthesis failed: {str(e)}")
        
        # Create fallback references from source summaries
        fallback_references = []
        if state.source_summaries:
            for summary in state.source_summaries[:3]:  # Use first 3 sources
                fallback_references.append(Reference(
                    title=summary.title,
                    url=summary.url,
                    relevance_note="Source used in failed synthesis attempt"
                ))
        else:
            # Fallback reference if no summaries available
            fallback_references.append(Reference(
                title="Research Brief Generator",
                url="http://localhost:8000",
                relevance_note="System-generated fallback reference"
            ))
        
        state.final_brief = FinalBrief(
            title=f"Research Brief: {state.topic}",
            executive_summary="This research brief was generated with limited synthesis capabilities due to processing errors. The content is based on available source materials but may lack comprehensive analysis.",
            key_findings=["Research process encountered synthesis limitations", "Available sources were processed but integration was incomplete", "Results should be considered preliminary"],
            detailed_analysis="The detailed analysis could not be completed due to synthesis model limitations. Source materials were collected and processed, but comprehensive integration was not achieved.",
            implications="Further research and manual review may be required to develop comprehensive insights on this topic.",
            limitations="This brief was generated using fallback processing due to synthesis errors. The analysis may be incomplete and should be supplemented with additional research.",
            references=fallback_references,
            metadata=BriefMetadata(
                research_duration=60, 
                total_sources_found=len(state.search_results) if state.search_results else 0, 
                sources_used=len(state.source_summaries) if state.source_summaries else 0, 
                confidence_score=0.3, 
                depth_level=state.depth
            )
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