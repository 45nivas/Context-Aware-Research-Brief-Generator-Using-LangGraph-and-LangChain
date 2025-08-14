"""
Database models and operations for storing user context and research briefs.
Uses SQLAlchemy for ORM and async operations.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.future import select

from app.config import config
from app.models import UserContext

# This Base is used by the init_db.py script to find the tables.
Base = declarative_base()

# -----------------------------
# ORM MODELS
# -----------------------------
class UserContextDB(Base):
    """Database model for user context storage."""
    __tablename__ = "user_contexts"

    user_id = Column(String(100), primary_key=True)
    previous_topics = Column(JSON, default=list)
    brief_summaries = Column(JSON, default=list)
    preferences = Column(JSON, default=dict)
    last_updated = Column(DateTime, default=datetime.utcnow)


class ResearchBriefDB(Base):
    """Database model for research brief storage."""
    __tablename__ = "research_briefs"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(100), nullable=False, index=True)
    topic = Column(String(500), nullable=False)
    title = Column(String(200), nullable=False)
    executive_summary = Column(Text, nullable=False)
    key_findings = Column(JSON, nullable=False)
    detailed_analysis = Column(Text, nullable=False)
    implications = Column(Text, nullable=False)
    limitations = Column(Text, nullable=False)
    references = Column(JSON, nullable=False)
    brief_metadata = Column(JSON, nullable=False)
    creation_timestamp = Column(DateTime, default=datetime.utcnow)

# -----------------------------
# DB MANAGER
# -----------------------------
class DatabaseManager:
    """Manages all database operations."""

    def __init__(self):
        self.engine = create_async_engine(
            config.database.url,
            echo=config.api.debug
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init_db_tables(self):
        """
        Creates all database tables. This should be run by a separate script,
        not by the application at startup.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def init_db(self):
        """
        This method is called by the application's startup event.
        It should NOT create tables to avoid race conditions in multi-worker environments.
        Table creation is handled by the `init_db.py` script.
        """
        pass  # Intentionally left blank.

    @staticmethod
    def _row_to_dict(row_obj) -> Dict[str, Any]:
        """Convert a SQLAlchemy ORM row to a plain dict."""
        if row_obj is None:
            return {}
        return {c.name: getattr(row_obj, c.name) for c in row_obj.__table__.columns}

    async def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """Retrieve user context from the database."""
        async with self.async_session() as session:
            db_obj = await session.get(UserContextDB, user_id)
            if not db_obj:
                return None
            data = self._row_to_dict(db_obj)
            return UserContext.model_validate(data)

    async def save_research_brief(self, brief: Dict[str, Any], user_id: str, topic: str) -> str:
        """Save a research brief to the database and return its ID."""
        import uuid
        brief_id = str(uuid.uuid4())

        if "brief_metadata" not in brief or not brief["brief_metadata"]:
            brief["brief_metadata"] = {
                "creation_timestamp": datetime.utcnow().isoformat(),
                "research_duration": 0,
                "total_sources_found": 0,
                "sources_used": 0,
                "confidence_score": 0,
                "depth_level": brief.get("depth_level", 1),
                "token_usage": {}
            }

        async with self.async_session() as session:
            async with session.begin():
                db_brief = ResearchBriefDB(
                    id=brief_id,
                    user_id=user_id,
                    topic=topic,
                    **brief
                )
                session.add(db_brief)
        return brief_id

    async def get_user_briefs(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get a user's previous research briefs."""
        async with self.async_session() as session:
            result = await session.execute(
                select(ResearchBriefDB)
                .where(ResearchBriefDB.user_id == user_id)
                .order_by(ResearchBriefDB.creation_timestamp.desc())
                .limit(limit)
            )
            return [self._row_to_dict(brief) for brief in result.scalars().all()]

    async def update_user_context_with_brief(self, user_id: str, topic: str, brief_summary: str) -> None:
        """Update user context with new brief information."""
        async with self.async_session() as session:
            async with session.begin():
                db_context = await session.get(UserContextDB, user_id)
                if not db_context:
                    db_context = UserContextDB(
                        user_id=user_id,
                        previous_topics=[],
                        brief_summaries=[],
                        preferences={}
                    )

                if topic not in db_context.previous_topics:
                    # Create a mutable copy before appending
                    new_topics = list(db_context.previous_topics)
                    new_topics.append(topic)
                    db_context.previous_topics = new_topics[-20:]

                # Create a mutable copy before appending
                new_summaries = list(db_context.brief_summaries)
                new_summaries.append(brief_summary)
                db_context.brief_summaries = new_summaries[-10:]
                
                db_context.last_updated = datetime.utcnow()
                await session.merge(db_context)

# Global database manager instance
db_manager = DatabaseManager()