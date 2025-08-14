"""
Database models and operations for storing user context and research briefs.
Uses SQLAlchemy for ORM and async operations.
"""

import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, Column, String, DateTime, Text, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.future import select

from app.config import config
from app.models import UserContext, FinalBrief, BriefMetadata

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

    async def init_db(self):
        """
        Initialize database tables if they don't exist.
        Avoids race conditions by checking table existence explicitly.
        """
        async with self.engine.begin() as conn:
            def _init(sync_conn):
                inspector = inspect(sync_conn)
                existing_tables = inspector.get_table_names()
                needed_tables = Base.metadata.tables.keys()
                missing_tables = [
                    Base.metadata.tables[name] for name in needed_tables
                    if name not in existing_tables
                ]
                if missing_tables:
                    Base.metadata.create_all(sync_conn, tables=missing_tables)

            await conn.run_sync(_init)

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
        brief_id = str(uuid.uuid4())

        async with self.async_session() as session:
            async with session.begin():
                db_brief = ResearchBriefDB(
                    id=brief_id,
                    user_id=user_id,
                    topic=topic,
                    **brief
                )
                session.add(db_brief)

        # Update or create user context
        brief_summary = {
            "brief_id": brief_id,
            "topic": topic,
            "title": brief.get("title"),
            "created_at": datetime.utcnow().isoformat()
        }

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

                if topic not in (db_context.previous_topics or []):
                    db_context.previous_topics = (db_context.previous_topics or []) + [topic]
                    db_context.previous_topics = db_context.previous_topics[-20:]

                db_context.brief_summaries = (db_context.brief_summaries or []) + [brief_summary]
                db_context.brief_summaries = db_context.brief_summaries[-10:]

                db_context.last_updated = datetime.utcnow()

                await session.merge(db_context)

        return brief_id


# Global database manager instance
db_manager = DatabaseManager()
