from typing import Optional, Union

from uuid import UUID

from fastapi_pagination.bases import AbstractPage, AbstractParams
from sqlalchemy import func
from sqlalchemy.exc import NoResultFound
from sqlmodel import desc, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.sql.expression import Select, SelectOfScalar

from service.crud.base import CRUDBase, ModelType
from service.models.models import TopicModel, TopicModelBase


class CRUDTopicModel(CRUDBase[TopicModel, TopicModelBase, TopicModelBase]):
    async def get_by_id_version(
        self, db: AsyncSession, *, model_id: UUID, version: int
    ) -> ModelType:
        statement = select(TopicModel).filter(
            self.model.model_id == model_id, self.model.version == version
        )
        model: Optional[ModelType] = (await db.execute(statement)).scalars().first()
        if model is None:
            raise NoResultFound()
        return model

    async def remove_by_id_version(
        self, db: AsyncSession, *, model_id: UUID, version: int
    ) -> ModelType:
        model: ModelType = await self.get_by_id_version(db, model_id=model_id, version=version)
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

    async def paginate(
        self,
        db: AsyncSession,
        query: Optional[Union[Select[TopicModel], SelectOfScalar[TopicModel]]] = None,
        params: Optional[AbstractParams] = None,
    ) -> AbstractPage[TopicModel]:
        query = select(self.model).order_by(desc(self.model.created_at))
        return await super().paginate(db, query, params)


topic_model = CRUDTopicModel(TopicModel)
