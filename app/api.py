"""
FastAPI REST API for the Research Brief Generator.
Provides HTTP endpoints for brief generation and status checking.
"""

# IMPORTANT: Load environment variables from .env file at the very top
from dotenv import load_dotenv
load_dotenv()

# --- Standard Library Imports ---
import logging
from datetime import datetime
from typing import Dict, Any, List
import os

# --- Third-Party Imports ---
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Local Application Imports ---
from app.models import BriefRequest, FinalBrief
from app.workflow import research_workflow
from app.database import db_manager

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Research Brief Generator API",
    description="AI-powered research brief generation using LangGraph and LangChain",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be more specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for active workflow statuses
active_workflows: Dict[str, Dict[str, Any]] = {}

# --- Background Task for Workflow Execution ---
async def run_workflow_in_background(workflow_id: str, request: BriefRequest):
    """Helper function to run the research workflow and update its status."""
    active_workflows[workflow_id]["status"] = "running"
    try:
        result = await research_workflow.run_workflow(request)
        active_workflows[workflow_id].update({
            "status": "completed",
            "end_time": datetime.utcnow().isoformat(),
            "result": result.dict() 
        })
        await db_manager.save_brief(result, request.user_id)
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}", exc_info=True)
        active_workflows[workflow_id].update({
            "status": "failed",
            "end_time": datetime.utcnow().isoformat(),
            "error": str(e)
        })

# --- API Events ---
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    try:
        await db_manager.init_db()
        logger.info("✓ Database initialized successfully.")
        logger.info("✓ API server startup complete.")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        from app.llm_tools_free import content_fetcher
        await content_fetcher.close()
        logger.info("✓ Web content fetcher session closed.")
    except Exception as e:
        logger.warning(f"Warning: Cleanup error during shutdown: {e}")

# --- API Endpoints ---
@app.get("/", summary="API Root")
async def root():
    """Provides basic information about the API."""
    return {
        "message": "Research Brief Generator API is operational.",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", summary="Health Check")
async def health_check():
    """Performs a health check of the API and its dependencies."""
    try:
        await db_manager.get_user_context("health_check_user")
        # A simple check to ensure llm_manager is initialized
        from app.llm_tools_free import llm_manager
        if not llm_manager.available_services:
             raise RuntimeError("No LLM services are available.")
        return {"status": "healthy", "database": "connected", "llm_services": llm_manager.available_services}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.post("/brief", status_code=status.HTTP_202_ACCEPTED, summary="Generate New Research Brief")
async def generate_brief(request: BriefRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Accepts a research brief request and starts the generation in the background.
    """
    if not request.topic or len(request.topic) < 10:
        raise HTTPException(status_code=400, detail="Topic must be a non-empty string of at least 10 characters.")
    
    workflow_id = f"{request.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    active_workflows[workflow_id] = {
        "workflow_id": workflow_id,
        "start_time": datetime.utcnow().isoformat(),
        "status": "queued",
        "request": request.dict(),
    }
    
    background_tasks.add_task(run_workflow_in_background, workflow_id, request)
    
    return {
        "message": "Research brief generation has been accepted.",
        "workflow_id": workflow_id,
        "status_endpoint": f"/brief/{workflow_id}/status"
    }

@app.get("/brief/{workflow_id}/status", summary="Get Brief Status or Result")
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Retrieves the status of a workflow. If complete, returns the final brief.
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    
    workflow_info = active_workflows[workflow_id]
    
    if workflow_info.get("status") == "completed":
        return JSONResponse(content=workflow_info.get("result"))

    return workflow_info

@app.get("/user/{user_id}/history", summary="Get User's Brief History")
async def get_user_history(user_id: str, limit: int = 10) -> List[FinalBrief]:
    """
    Retrieves a user's research brief history from the database.
    """
    try:
        briefs = await db_manager.get_user_briefs(user_id, limit)
        return briefs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user history: {e}")

@app.get("/metrics", summary="Get API Metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get API metrics and statistics.
    """
    running_workflows = sum(1 for w in active_workflows.values() if w["status"] == "running")
    completed_workflows = sum(1 for w in active_workflows.values() if w["status"] == "completed")
    failed_workflows = sum(1 for w in active_workflows.values() if w["status"] == "failed")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "workflows": {
            "total_tracked": len(active_workflows),
            "running": running_workflows,
            "completed": completed_workflows,
            "failed": failed_workflows
        }
    }

# --- Main Entry Point for Running the Server ---
if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True 
    )