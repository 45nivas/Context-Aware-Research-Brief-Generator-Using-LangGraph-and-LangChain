"""
Database models and operations for storing user context and research briefs.
Uses SQLAlchemy for ORM and async operations.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import JSON

from app.config import config
from app.models import UserContext, FinalBrief, BriefMetadata

Base = declarative_base()


class UserContextDB(Base):
    """Database model for user context storage."""
    __tablename__ = "user_contexts"
    
    user_id = Column(String(100), primary_key=True)
    previous_topics = Column(JSON, default=list)
    brief_summaries = Column(JSON, default=list)
    preferences = Column(JSON, default=dict)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    def to_pydantic(self) -> UserContext:
        """Convert to Pydantic model."""
        return UserContext(
            user_id=self.user_id,
            previous_topics=self.previous_topics or [],
            brief_summaries=self.brief_summaries or [],
            preferences=self.preferences or {},
            last_updated=self.last_updated
        )


class ResearchBriefDB(Base):
    """Database model for research brief storage."""
    __tablename__ = "research_briefs"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(100), nullable=False)
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
    
    def to_pydantic(self) -> FinalBrief:
        """Convert to Pydantic model."""
        # Reconstruct metadata
        metadata_dict = self.brief_metadata or {}
        metadata = BriefMetadata(**metadata_dict)
        
        return FinalBrief(
            title=self.title,
            executive_summary=self.executive_summary,
            key_findings=self.key_findings or [],
            detailed_analysis=self.detailed_analysis,
            implications=self.implications,
            limitations=self.limitations,
            references=[ref for ref in (self.references or [])],
            metadata=metadata
        )


class DatabaseManager:
    """Manages database operations."""
    
    def __init__(self):
        self.engine = create_async_engine(config.database.url, echo=config.api.debug)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def init_db(self):
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """
        Retrieve user context from database.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserContext if found, None otherwise
        """
        async with self.async_session() as session:
            result = await session.get(UserContextDB, user_id)
            return result.to_pydantic() if result else None
    
    async def save_user_context(self, user_context: UserContext) -> None:
        """
        Save user context to database.
        
        Args:
            user_context: UserContext to save
        """
        async with self.async_session() as session:
            db_context = UserContextDB(
                user_id=user_context.user_id,
                previous_topics=user_context.previous_topics,
                brief_summaries=user_context.brief_summaries,
                preferences=user_context.preferences,
                last_updated=user_context.last_updated
            )
            await session.merge(db_context)
            await session.commit()
    
    async def save_research_brief(self, brief: FinalBrief, user_id: str, topic: str) -> str:
        """
        Save research brief to database.
        
        Args:
            brief: FinalBrief to save
            user_id: User identifier
            topic: Research topic
            
        Returns:
            Brief ID
        """
        import uuid
        brief_id = str(uuid.uuid4())
        
        async with self.async_session() as session:
            db_brief = ResearchBriefDB(
                id=brief_id,
                user_id=user_id,
                topic=topic,
                title=brief.title,
                executive_summary=brief.executive_summary,
                key_findings=brief.key_findings,
                detailed_analysis=brief.detailed_analysis,
                implications=brief.implications,
                limitations=brief.limitations,
                references=[ref.dict() for ref in brief.references],
                brief_metadata=brief.metadata.dict()
            )
            session.add(db_brief)
            await session.commit()
            return brief_id
    
    async def get_user_briefs(self, user_id: str, limit: int = 10) -> List[ResearchBriefDB]:
        """
        Get user's previous research briefs.
        
        Args:
            user_id: User identifier
            limit: Maximum number of briefs to return
            
        Returns:
            List of research briefs
        """
        async with self.async_session() as session:
            from sqlalchemy import select
            
            result = await session.execute(
                select(ResearchBriefDB)
                .where(ResearchBriefDB.user_id == user_id)
                .order_by(ResearchBriefDB.creation_timestamp.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def update_user_context_with_brief(self, user_id: str, topic: str, brief_summary: str) -> None:
        """
        Update user context with new brief information.
        
        Args:
            user_id: User identifier
            topic: Research topic
            brief_summary: Summary of the brief
        """
        context = await self.get_user_context(user_id)
        if not context:
            context = UserContext(user_id=user_id)
        
        # Add topic if not already present
        if topic not in context.previous_topics:
            context.previous_topics.append(topic)
            # Keep only last 20 topics
            context.previous_topics = context.previous_topics[-20:]
        
        # Add brief summary
        context.brief_summaries.append(brief_summary)
        # Keep only last 10 summaries
        context.brief_summaries = context.brief_summaries[-10:]
        
        context.last_updated = datetime.utcnow()
        
        await self.save_user_context(context)


# Global database manager instance
db_manager = DatabaseManager()
# FINAL VERSION