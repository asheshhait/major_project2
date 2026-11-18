from __future__ import annotations

from fastapi import APIRouter

from ..core.bangla import split_graphemes
from ..ml.predictor import predict_word
from ..schemas import PredictionRequest

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict/{word_id}")
async def predict_word_endpoint(word_id: int, sample: PredictionRequest):
    result = predict_word([(point.x, point.y) for point in sample.points])
    return {
        **result,
        "word_id": word_id,
        "chars": split_graphemes(result["text"]) if result["text"] != "Unknown" else [],
    }


@router.post("/predict/session/{session_id}")
async def predict_session(session_id: str):
    return {"session_id": session_id, "predictions": []}
