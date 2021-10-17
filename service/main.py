from typing import Any, List

from bertopic import BERTopic
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import fetch_20newsgroups

app = FastAPI()


class Input(BaseModel):
    texts: List[str] = []


@app.post("/modeling", summary="Running topic modeling")
async def predict(data: Input) -> Any:
    topic_model = BERTopic()
    if data.texts:
        topics, probs = topic_model.fit_transform(data.texts)
    else:
        docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"][
            :100
        ]
        topics, probs = topic_model.fit_transform(docs)
    return topic_model.get_topic_info().to_dict("records")
