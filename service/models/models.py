from typing import Optional

from uuid import UUID

from sqlmodel import Field, SQLModel


class TopicBase(SQLModel):
    name: str
    count: int
    topic: int


class Topic(TopicBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)  # NOQA
    model_id: UUID
