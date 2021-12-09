from typing import Any, Dict, Generic, Optional, Sequence, Type, TypeVar, Union

from fastapi.encoders import jsonable_encoder
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

ModelType = TypeVar("ModelType", bound=SQLModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=SQLModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=SQLModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).

        **Parameters**

        * `model`: A SQLAlchemy model class
        * `schema`: A Pydantic model (schema) class
        """
        self.model = model

    async def get(self, db: AsyncSession, id: Any) -> Optional[ModelType]:  # NOQA
        return await db.get(self.model, id)

    async def get_multi(
        self, db: AsyncSession, *, skip: int = 0, limit: int = 100
    ) -> Sequence[ModelType]:
        statement = select(self.model).offset(skip).limit(limit)
        return (await db.exec(statement)).all()  # type: ignore

    async def create(self, db: AsyncSession, *, obj_in: CreateSchemaType) -> ModelType:
        # TODO: review this
        # .from_orm(hero)
        # obj_in_data = jsonable_encoder(obj_in)
        # db_obj = self.model.from_orm(obj_in)(**obj_in_data)  # type: ignore
        db_obj: ModelType = self.model.from_orm(obj_in)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]],
    ) -> ModelType:
        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def remove(self, db: AsyncSession, *, id: int) -> Optional[ModelType]:  # NOQA
        obj = await db.get(self.model, id)
        if obj is not None:
            db.delete(obj)
            db.commit()
            return obj
        return None
