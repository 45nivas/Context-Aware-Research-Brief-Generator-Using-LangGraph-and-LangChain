"""
Command Line Interface for the Research Brief Generator.
Provides local execution capabilities with rich output formatting.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.json import JSON

from app.models import BriefRequest, DepthLevel
from app.workflow import research_workflow
from app.database import db_manager
from app.config import config

# Initialize CLI app and console
cli = typer.Typer(name="research-brief-generator", help="AI-powered research brief generation")
console = Console()


@cli.command()
def generate(
    topic: str = typer.Argument(..., help="Research topic"),
    depth: int = typer.Option(2, "--depth", "-d", min=1, max=4, help="Research depth (1-4)"),
    follow_up: bool = typer.Option(False, "--follow-up", "-f", help="This is a follow-up research"),
    user_id: str = typer.Option("cli-user", "--user", "-u", help="User identifier"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("rich", "--format", help="Output format: rich, json, markdown")
):
    """Generate a research brief from the command line."""
    
    console.print(Panel.fit("🔬 Research Brief Generator", style="bold blue"))
    
    try:
        # Validate configuration
        config.validate()
        
        # Create request
        request = BriefRequest(
            topic=topic,
            depth=DepthLevel(depth),
            follow_up=follow_up,
            user_id=user_id,
            context=context
        )
        
        # Display request info
        console.print(f"📝 Topic: {topic}")
        console.print(f"📊 Depth Level: {DepthLevel(depth).name} ({depth})")
        console.print(f"👤 User: {user_id}")
        console.print(f"🔄 Follow-up: {'Yes' if follow_up else 'No'}")
        if context:
            console.print(f"💭 Context: {context}")
        
        console.print()
        
        # Run workflow with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Initializing workflow...", total=None)
            
            async def run_with_progress():
                # Initialize database
                progress.update(task, description="Initializing database...")
                await db_manager.init_db()
                
                progress.update(task, description="Starting research workflow...")
                
                # Execute workflow
                result = await research_workflow.run_workflow(request)
                
                progress.update(task, description="Workflow completed!", total=1, completed=1)
                return result
            
            # Run the async workflow
            result = asyncio.run(run_with_progress())
        
        console.print("✅ Research brief generated successfully!")
        console.print()
        
        # Display results based on format
        if format == "rich":
            display_rich_output(result)
        elif format == "json":
            display_json_output(result, output)
        elif format == "markdown":
            display_markdown_output(result, output)
        else:
            console.print(f"❌ Unknown format: {format}")
            raise typer.Exit(1)
        
        # Display metrics
        display_metrics(result)
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        raise typer.Exit(1)


def display_rich_output(result):
    """Display results in rich format."""
    
    # Title and executive summary
    console.print(Panel(result.title, style="bold green", title="Research Brief"))
    console.print(Panel(result.executive_summary, title="Executive Summary", style="blue"))
    
    # Key findings
    if result.key_findings:
        console.print("\n📋 Key Findings:")
        for i, finding in enumerate(result.key_findings, 1):
            console.print(f"  {i}. {finding}")
    
    # Detailed analysis
    console.print(Panel(result.detailed_analysis, title="Detailed Analysis", style="cyan"))
    
    # Implications
    console.print(Panel(result.implications, title="Implications", style="yellow"))
    
    # References
    if result.references:
        console.print("\n📚 References:")
        ref_table = Table(show_header=True, header_style="bold magenta")
        ref_table.add_column("Title", style="cyan")
        ref_table.add_column("URL", style="blue")
        ref_table.add_column("Relevance", style="green")
        
        for ref in result.references:
            ref_table.add_row(
                ref.title[:50] + "..." if len(ref.title) > 50 else ref.title,
                ref.url[:60] + "..." if len(ref.url) > 60 else ref.url,
                ref.relevance_note[:40] + "..." if len(ref.relevance_note) > 40 else ref.relevance_note
            )
        
        console.print(ref_table)
    
    # Limitations
    console.print(Panel(result.limitations, title="Limitations", style="red"))


def display_json_output(result, output_path: Optional[str]):
    """Display results in JSON format."""
    json_data = result.dict()
    
    if output_path:
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        console.print(f"✅ Results saved to: {output_path}")
    else:
        # Display in console
        console.print(JSON(json.dumps(json_data, indent=2, default=str)))


def display_markdown_output(result, output_path: Optional[str]):
    """Display results in Markdown format."""
    
    markdown_content = f"""# {result.title}

## Executive Summary
{result.executive_summary}

## Key Findings
"""
    
    for i, finding in enumerate(result.key_findings, 1):
        markdown_content += f"{i}. {finding}\n"
    
    markdown_content += f"""
## Detailed Analysis
{result.detailed_analysis}

## Implications
{result.implications}

## References
"""
    
    for i, ref in enumerate(result.references, 1):
        markdown_content += f"{i}. [{ref.title}]({ref.url}) - {ref.relevance_note}\n"
    
    markdown_content += f"""
## Limitations
{result.limitations}

---
*Generated on {result.metadata.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Research Duration: {result.metadata.research_duration}s | Sources: {result.metadata.sources_used}*
"""
    
    if output_path:
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        console.print(f"✅ Results saved to: {output_path}")
    else:
        # Display in console
        console.print(Markdown(markdown_content))


def display_metrics(result):
    """Display execution metrics."""
    console.print("\n📊 Execution Metrics:")
    
    metrics_table = Table(show_header=True, header_style="bold cyan")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    metrics_table.add_row("Research Duration", f"{result.metadata.research_duration}s")
    metrics_table.add_row("Total Sources Found", str(result.metadata.total_sources_found))
    metrics_table.add_row("Sources Used", str(result.metadata.sources_used))
    metrics_table.add_row("Confidence Score", f"{result.metadata.confidence_score:.2f}")
    metrics_table.add_row("Depth Level", result.metadata.depth_level.name)
    
    # Token usage
    if result.metadata.token_usage:
        for model, usage in result.metadata.token_usage.items():
            metrics_table.add_row(f"Tokens ({model})", str(usage.get('total_tokens', 0)))
    
    console.print(metrics_table)


@cli.command()
def history(
    user_id: str = typer.Option("cli-user", "--user", "-u", help="User identifier"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of briefs to show")
):
    """Show user's research brief history."""
    
    async def get_history():
        await db_manager.init_db()
        user_context = await db_manager.get_user_context(user_id)
        briefs = await db_manager.get_user_briefs(user_id, limit)
        return user_context, briefs
    
    try:
        user_context, briefs = asyncio.run(get_history())
        
        console.print(Panel.fit(f"📚 Research History for {user_id}", style="bold blue"))
        
        if not briefs:
            console.print("No research briefs found for this user.")
            return
        
        # Display user context
        if user_context:
            console.print(f"\n👤 User Profile:")
            console.print(f"  Previous Topics: {len(user_context.previous_topics)}")
            console.print(f"  Brief Summaries: {len(user_context.brief_summaries)}")
            console.print(f"  Last Updated: {user_context.last_updated}")
        
        # Display briefs
        console.print(f"\n📋 Recent Briefs ({len(briefs)}):")
        
        history_table = Table(show_header=True, header_style="bold magenta")
        history_table.add_column("Date", style="cyan")
        history_table.add_column("Topic", style="green")
        history_table.add_column("Title", style="blue")
        
        for brief in briefs:
            history_table.add_row(
                brief.creation_timestamp.strftime('%Y-%m-%d %H:%M'),
                brief.topic[:30] + "..." if len(brief.topic) > 30 else brief.topic,
                brief.title[:40] + "..." if len(brief.title) > 40 else brief.title
            )
        
        console.print(history_table)
        
    except Exception as e:
        console.print(f"❌ Error retrieving history: {str(e)}", style="red")
        raise typer.Exit(1)


@cli.command()
def config_info():
    """Display current configuration information."""
    
    console.print(Panel.fit("⚙️  Configuration Information", style="bold blue"))
    
    # Model configuration
    console.print("\n🤖 Model Configuration:")
    config_table = Table(show_header=True, header_style="bold cyan")
    config_table.add_column("Component", style="cyan")
    config_table.add_column("Model", style="green")
    config_table.add_column("Provider", style="blue")
    
    config_table.add_row("Primary (Planning/Synthesis)", config.primary_model.model_name, config.primary_model.provider)
    config_table.add_row("Secondary (Summarization)", config.secondary_model.model_name, config.secondary_model.provider)
    config_table.add_row("Embeddings", config.embedding_model.model_name, config.embedding_model.provider)
    
    console.print(config_table)
    
    # Model rationale
    console.print("\n💭 Model Selection Rationale:")
    rationale = config.get_model_rationale()
    for key, explanation in rationale.items():
        console.print(f"  • {key}: {explanation}")
    
    # Search configuration
    console.print(f"\n🔍 Search Configuration:")
    console.print(f"  Max Results: {config.search.max_results}")
    console.print(f"  Max Sources per Brief: {config.search.max_sources_per_brief}")
    
    # Database
    console.print(f"\n🗄️  Database: {config.database.url}")
    
    # Tracing
    console.print(f"\n📊 LangSmith Tracing: {'Enabled' if config.tracing.enabled else 'Disabled'}")
    if config.tracing.enabled:
        console.print(f"  Project: {config.tracing.project}")


@cli.command()
def workflow_graph():
    """Display the workflow graph visualization."""
    
    console.print(Panel.fit("🔄 Workflow Graph", style="bold blue"))
    console.print(research_workflow.get_graph_visualization())


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
