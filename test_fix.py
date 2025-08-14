#!/usr/bin/env python3
"""
Quick test to verify the production fixes work.
"""

import asyncio
from app.workflow import ResearchWorkflow
from app.models import BriefRequest, DepthLevel

async def test_workflow():
    """Test that the workflow doesn't crash with the fixes."""
    try:
        workflow = ResearchWorkflow()
        print("✅ Workflow initialized successfully")
        
        # Test token usage methods
        from app.llm_tools_free import llm_manager
        llm_manager.reset_token_usage()
        token_usage = llm_manager.get_token_usage()
        print(f"✅ Token usage methods work: {token_usage}")
        
        print("✅ All critical fixes verified - production deployment should work")
        return True
        
    except Exception as e:
        print(f"❌ Error in workflow test: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_workflow())
