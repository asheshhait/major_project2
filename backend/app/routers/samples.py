from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ..db import BadSample, Sample, Session as DbSession, SessionLocal, Word
from ..schemas import BadSampleCreate, SampleCreate

router = APIRouter(prefix="/api", tags=["samples"])


def _session_and_word(db, session_id: str, word_id: int) -> tuple[DbSession, Word]:
    db_session = db.scalar(select(DbSession).where(DbSession.session_id == session_id))
    if db_session is None:
        raise HTTPException(status_code=404, detail="Upload session not found. Upload the file again.")
    word = db.scalar(
        select(Word).where(Word.id == word_id, Word.session_id == db_session.id)
    )
    if word is None:
        raise HTTPException(status_code=404, detail="Word not found in this upload session.")
    return db_session, word


@router.get("/samples")
async def list_samples():
    with SessionLocal() as db:
        samples = db.scalars(select(Sample).where(Sample.is_bad.is_(False)).order_by(Sample.id)).all()
        return {
            "items": [
                {
                    "id": sample.id,
                    "session_id": sample.session.session_id,
                    "word_id": sample.word_id,
                    "text": sample.text,
                    "chars": json.loads(sample.chars),
                    "created_at": sample.created_at.isoformat(),
                }
                for sample in samples
            ],
            "count": len(samples),
        }


@router.post("/samples")
async def save_sample(sample: SampleCreate):
    text = sample.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter the correct Bangla word before saving.")
    characters = [character.strip() for character in sample.chars if character.strip()]
    if not characters:
        raise HTTPException(status_code=422, detail="Add at least one character in Segmentation.")

    with SessionLocal() as db:
        db_session, word = _session_and_word(db, sample.session_id, sample.word_id)
        saved = Sample(
            session_id=db_session.id,
            word_id=word.id,
            text=text,
            chars=json.dumps(characters, ensure_ascii=False),
            raw_points=json.dumps(
                [point.model_dump() for point in sample.points],
                ensure_ascii=False,
            ),
        )
        db.add(saved)
        word.text = text
        db.commit()
        db.refresh(saved)
        return {"saved": True, "sample_id": saved.id, "text": saved.text}


@router.post("/samples/bad")
async def mark_bad_sample(sample: BadSampleCreate):
    with SessionLocal() as db:
        db_session, word = _session_and_word(db, sample.session_id, sample.word_id)
        db.add(
            BadSample(
                session_id=db_session.id,
                word_id=word.id,
                raw_points=json.dumps(
                    [point.model_dump() for point in sample.points],
                    ensure_ascii=False,
                ),
                reason=sample.reason,
            )
        )
        db.commit()
        return {"saved": True, "message": "Bad sample recorded; it will not be used for training."}
