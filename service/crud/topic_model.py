from typing import Optional

from uuid import UUID

from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from service.crud.base import CRUDBase, ModelType
from service.models.models import TopicModel, TopicModelBase


class CRUDTopicModel(CRUDBase[TopicModel, TopicModelBase, TopicModelBase]):
    async def get_by_id_version(
        self, db: AsyncSession, *, model_id: UUID, version: int
    ) -> Optional[ModelType]:
        statement = select(TopicModel).filter(
            self.model.model_id == model_id, self.model.version == version
        )
        return (await db.execute(statement)).scalars().first()

    async def remove_by_id_version(
        self, db: AsyncSession, *, model_id: UUID, version: int
    ) -> Optional[ModelType]:
        model: Optional[ModelType] = await self.get_by_id_version(
            db, model_id=model_id, version=version
        )
        if model is not None:
            await db.delete(model)
            await db.commit()
        return model

    async def get_max_version(self, db: AsyncSession, *, model_id: UUID) -> int:
        return (
            await db.execute(
                select(self.model)
                .filter(self.model.model_id == model_id)
                .with_only_columns(func.max(self.model.version))
            )
        ).scalar() or 0


topic_model = CRUDTopicModel(TopicModel)
