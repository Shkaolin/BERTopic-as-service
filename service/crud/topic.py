from typing import Any, Dict, List

from uuid import UUID

from sqlalchemy.orm import selectinload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from service.crud.base import CRUDBase, ModelType
from service.models.models import Topic, TopicBase, TopicCreate, TopicModel, Word


class CRUDTopic(CRUDBase[Topic, TopicCreate, TopicBase]):
    async def get_model_topics(
        self, db: AsyncSession, *, model_id: UUID, version: int, with_words: bool = False
    ) -> List[ModelType]:
        statement = (
            select(self.model)
            .join(TopicModel)
            .filter(TopicModel.model_id == model_id, TopicModel.version == version)
            .order_by(-self.model.count)
        )
        if with_words:
            statement = statement.options(selectinload(self.model.top_words))
        return (await db.execute(statement)).scalars().all()

    async def save_topics(
        self, db: AsyncSession, *, topics: List[Dict[str, Any]], model: TopicModel
    ) -> None:
        for topic in topics:
            db_obj = self.model.from_orm(
                TopicCreate.parse_obj({**topic, "topic_model_id": model.id})
            )
            db.add(db_obj)
            for word in topic["top_words"]:
                db.add(Word.parse_obj({**word, "topic": db_obj}))

        await db.commit()


topic = CRUDTopic(Topic)
