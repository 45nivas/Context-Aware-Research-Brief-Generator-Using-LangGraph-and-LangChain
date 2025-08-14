"""
FastAPI REST API for the Research Brief Generator.
Provides HTTP endpoints for brief generation and status checking.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio

from app.models import BriefRequest, FinalBrief, DepthLevel
from app.workflow import research_workflow
from app.database import db_manager
from app.config import config
from app.llm_tools_free import llm_manager

# Initialize FastAPI app
app = FastAPI(
    title="Research Brief Generator API",
    description="AI-powered research brief generation using LangGraph and LangChain",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active workflow executions
active_workflows: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize database and validate configuration on startup."""
    try:
        # Validate configuration
        config.validate()
        
        # Initialize database
        await db_manager.init_db()
        
        print("✓ API server initialized successfully")
        print(f"✓ Database initialized at: {config.database.url}")
        print(f"✓ LangSmith tracing: {'enabled' if config.tracing.enabled else 'disabled'}")
        
    except Exception as e:
        print(f"✗ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        from app.llm_tools_free import content_fetcher
        await content_fetcher.close()
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"Warning: Cleanup error: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Research Brief Generator API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "generate_brief": "POST /brief",
            "workflow_status": "GET /brief/{workflow_id}/status",
            "health_check": "GET /health",
            "docs": "GET /docs"
        },
        "model_configuration": config.get_model_rationale()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        await db_manager.get_user_context("health_check_user")
        
        # Test LLM availability (basic check)
        token_usage = llm_manager.get_token_usage()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "database": "connected",
            "llm_models": "available",
            "active_workflows": len(active_workflows)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/brief", response_model=FinalBrief)
async def generate_brief(
    request: BriefRequest,
    background_tasks: BackgroundTasks = None
) -> FinalBrief:
    """
    Generate a research brief.
    
    Args:
        request: Research brief request parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Complete research brief
        
    Raises:
        HTTPException: If generation fails
    """
    try:
        # Validate request
        if not request.topic.strip():
            raise HTTPException(status_code=400, detail="Topic cannot be empty")
        
        if len(request.topic) < 10:
            raise HTTPException(status_code=400, detail="Topic must be at least 10 characters")
        
        # Generate workflow ID for tracking
        workflow_id = f"{request.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Store workflow info
        active_workflows[workflow_id] = {
            "start_time": datetime.utcnow(),
            "status": "running",
            "request": request.dict(),
            "current_step": "initializing"
        }
        
        try:
            # Execute workflow
            result = await research_workflow.run_workflow(request)
            
            # Update workflow status
            active_workflows[workflow_id].update({
                "status": "completed",
                "end_time": datetime.utcnow(),
                "result": "success"
            })
            
            return result
            
        except Exception as e:
            # Update workflow status with error
            active_workflows[workflow_id].update({
                "status": "failed",
                "end_time": datetime.utcnow(),
                "error": str(e)
            })
            raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
        
        finally:
            # Clean up old workflows (keep last 100)
            if len(active_workflows) > 100:
                oldest_keys = sorted(active_workflows.keys())[:50]
                for key in oldest_keys:
                    del active_workflows[key]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")


@app.get("/brief/{workflow_id}/status")
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Get the status of a workflow execution.
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        Workflow status information
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_info = active_workflows[workflow_id]
    
    # Get additional status from the workflow engine if still running
    if workflow_info["status"] == "running":
        try:
            detailed_status = await research_workflow.get_workflow_status(workflow_id)
            workflow_info.update(detailed_status)
        except Exception as e:
            workflow_info["status_error"] = str(e)
    
    return workflow_info


@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get user's research brief history.
    
    Args:
        user_id: User identifier
        limit: Maximum number of briefs to return
        
    Returns:
        User's brief history
    """
    try:
        # Get user context
        user_context = await db_manager.get_user_context(user_id)
        
        # Get user's briefs
        briefs = await db_manager.get_user_briefs(user_id, limit)
        
        return {
            "user_id": user_id,
            "total_briefs": len(briefs),
            "user_context": user_context.dict() if user_context else None,
            "recent_briefs": [
                {
                    "id": brief.id,
                    "topic": brief.topic,
                    "title": brief.title,
                    "creation_timestamp": brief.creation_timestamp,
                    "executive_summary": brief.executive_summary[:200] + "..." if len(brief.executive_summary) > 200 else brief.executive_summary
                }
                for brief in briefs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user history: {str(e)}")


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get API metrics and statistics.
    
    Returns:
        System metrics
    """
    try:
        total_workflows = len(active_workflows)
        running_workflows = sum(1 for w in active_workflows.values() if w["status"] == "running")
        completed_workflows = sum(1 for w in active_workflows.values() if w["status"] == "completed")
        failed_workflows = sum(1 for w in active_workflows.values() if w["status"] == "failed")
        
        # Get token usage
        token_usage = llm_manager.get_token_usage()
        
        return {
            "timestamp": datetime.utcnow(),
            "workflows": {
                "total": total_workflows,
                "running": running_workflows,
                "completed": completed_workflows,
                "failed": failed_workflows
            },
            "token_usage": token_usage,
            "configuration": {
                "primary_model": config.primary_model.model_name,
                "secondary_model": config.secondary_model.model_name,
                "max_sources": config.search.max_sources_per_brief,
                "tracing_enabled": config.tracing.enabled
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@app.get("/workflow/graph")
async def get_workflow_graph() -> Dict[str, Any]:
    """
    Get workflow graph visualization.
    
    Returns:
        Graph structure and visualization
    """
    return {
        "graph_structure": research_workflow.get_graph_visualization(),
        "nodes": [
            "context_summarization",
            "planning", 
            "search",
            "content_fetching",
            "per_source_summarization",
            "synthesis",
            "post_processing"
        ],
        "conditional_edges": [
            "context_summarization -> planning",
            "planning -> search", 
            "search -> content_fetching (if results available)",
            "search -> per_source_summarization (if no results)",
            "content_fetching -> per_source_summarization",
            "per_source_summarization -> synthesis (if summaries available)",
            "per_source_summarization -> search (retry once)",
            "synthesis -> post_processing",
            "post_processing -> END"
        ],
        "retry_logic": {
            "search": "1 retry maximum",
            "per_source_summarization": "2 retries per source",
            "synthesis": "3 retries maximum"
        }
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


def main():
    """Run the API server."""
    uvicorn.run(
        "app.api:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        access_log=True
    )


if __name__ == "__main__":
    main()
