from typing import List, Optional

from functools import wraps
from uuid import UUID

from sqlalchemy.sql.schema import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


# monkeypath from https://github.com/tiangolo/sqlmodel/issues/9
# without this all database fields are indexed be default
def set_default_index(func):
    """Decorator to set default index for SQLModel
    Can be removed when https://github.com/tiangolo/sqlmodel/pull/11 is merged
    """

    @wraps(func)
    def inner(*args, index=False, **kwargs):
        return func(*args, index=index, **kwargs)

    return inner


# monkey patch field with default index=False
# this works as long as we always call Field()
Field = set_default_index(Field)


class TopicModelBase(SQLModel):
    model_id: UUID = Field()
    version: int = Field(default=1)


class TopicModel(TopicModelBase, table=True):
    __tablename__ = "topic_model"
    __table_args__ = (UniqueConstraint("model_id", "version", name="_model_id_version_uc"),)

    id: Optional[int] = Field(primary_key=True, nullable=False)  # NOQA: A003
    topics: List["Topic"] = Relationship(
        back_populates="topic_model", sa_relationship_kwargs={"cascade": "all,delete"}
    )


class WordBase(SQLModel):
    name: str = Field()
    score: float = Field()


class WordCreate(WordBase):
    topic_id: int


class Word(WordBase, table=True):
    id: Optional[int] = Field(primary_key=True, nullable=False)  # NOQA: A003
    topic_id: int = Field(foreign_key="topic.id")
    topic: "Topic" = Relationship(
        back_populates="top_words", sa_relationship_kwargs={"cascade": "all,delete"}
    )


class TopicBase(SQLModel):
    name: str = Field()
    count: int = Field()
    topic_index: int = Field()


class TopicWithWords(TopicBase):
    top_words: List["WordBase"] = Field(default=[])


class TopicCreate(TopicBase):
    topic_model_id: int


class Topic(TopicBase, table=True):
    id: Optional[int] = Field(primary_key=True, nullable=False)  # NOQA: A003
    topic_model_id: int = Field(foreign_key="topic_model.id")
    top_words: List[Word] = Relationship(
        back_populates="topic", sa_relationship_kwargs={"cascade": "all,delete"}
    )
    topic_model: TopicModel = Relationship(
        back_populates="topics", sa_relationship_kwargs={"cascade": "all,delete"}
    )
