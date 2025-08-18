"""
Database management for the Research Brief Generator.
Handles SQLite database connections, schema creation, and data operations.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.future import select

from app.models import FinalBrief, UserContext

logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./research_briefs.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# --- SQLAlchemy Models ---

class ResearchBriefDB(Base):
    __tablename__ = "research_briefs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    topic = Column(String)
    creation_timestamp = Column(DateTime, default=datetime.utcnow)
    brief_data = Column(JSON)

class UserContextDB(Base):
    __tablename__ = "user_contexts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    context_data = Column(JSON)

# --- Database Manager Class ---

class DatabaseManager:
    """Handles all database operations."""

    async def init_db(self):
        """Initializes the database and creates tables if they don't exist."""
        async with engine.begin() as conn:
            # checkfirst=True prevents errors on restart by checking for table existence first
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)

    async def get_session(self) -> AsyncSession:
        """Provides an async database session."""
        return AsyncSessionLocal()

    async def save_brief(self, brief: FinalBrief, user_id: str):
        """Saves a completed research brief to the database."""
        async with await self.get_session() as session:
            # FIX: Use model_dump(mode='json') to correctly serialize complex types like datetime
            db_brief = ResearchBriefDB(
                user_id=user_id,
                topic=brief.title,
                brief_data=brief.model_dump(mode='json')
            )
            session.add(db_brief)
            await session.commit()
        logger.info(f"Saved brief for user {user_id} on topic '{brief.title}'")

    async def get_user_briefs(self, user_id: str, limit: int = 10) -> List[FinalBrief]:
        """Retrieves a list of recent briefs for a user."""
        async with await self.get_session() as session:
            result = await session.execute(
                select(ResearchBriefDB)
                .where(ResearchBriefDB.user_id == user_id)
                .order_by(ResearchBriefDB.creation_timestamp.desc())
                .limit(limit)
            )
            briefs = result.scalars().all()
            return [FinalBrief.model_validate(b.brief_data) for b in briefs]

    async def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """Retrieves the context for a given user."""
        async with await self.get_session() as session:
            result = await session.execute(
                select(UserContextDB).where(UserContextDB.user_id == user_id)
            )
            db_context = result.scalar_one_or_none()
            if db_context:
                return UserContext.model_validate(db_context.context_data)
        return None

    async def update_user_context_with_brief(self, user_id: str, topic: str, summary: str):
        """Updates a user's context with a new brief summary."""
        async with await self.get_session() as session:
            result = await session.execute(
                select(UserContextDB).where(UserContextDB.user_id == user_id)
            )
            db_context = result.scalar_one_or_none()

            if db_context:
                user_context = UserContext.model_validate(db_context.context_data)
                user_context.previous_topics.append(topic)
                user_context.brief_summaries.append(summary)
                # Keep history manageable
                user_context.previous_topics = user_context.previous_topics[-10:]
                user_context.brief_summaries = user_context.brief_summaries[-10:]
                # FIX: Use model_dump(mode='json') for correct serialization
                db_context.context_data = user_context.model_dump(mode='json')
                db_context.last_updated = datetime.utcnow()
            else:
                user_context = UserContext(
                    user_id=user_id,
                    previous_topics=[topic],
                    brief_summaries=[summary]
                )
                db_context = UserContextDB(
                    user_id=user_id,
                    # FIX: Use model_dump(mode='json') for correct serialization
                    context_data=user_context.model_dump(mode='json')
                )
                session.add(db_context)
            
            await session.commit()
        logger.info(f"Updated context for user {user_id}")

# Create a single, reusable instance of the manager for the application
db_manager = DatabaseManager()