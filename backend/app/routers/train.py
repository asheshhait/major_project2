from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ..core.segmentation import (
    gap_examples_from_corrections,
    set_learned_gap_threshold,
    train_gap_threshold,
)
from ..db import SegmentationModel, Session as DbSession, SessionLocal, UploadDocument, Word
from ..ml import predictor
from ..ml.yolo_character import start_yolo_training, yolo_training_status

router = APIRouter(prefix="/api", tags=["train"])
training_status = {"status": "idle", "epoch": 0, "loss": None, "eta_seconds": None}


@router.post("/train")
async def train_model():
    global training_status
    count = predictor.reload_saved_templates()
    segmentation = _train_segmentation_model()
    yolo = start_yolo_training()
    if not count and not segmentation["trained"] and yolo["status"] == "unavailable":
        raise HTTPException(
            status_code=400,
            detail=(
                "Save at least one labeled word sample or correct word boxes in an uploaded "
                "document before training."
            ),
        )

    training_status = {
        "status": "complete",
        "epoch": 1,
        "loss": None,
        "eta_seconds": 0,
    }
    return {
        **training_status,
        "trained_samples": count,
        "unique_words": predictor.trained_word_count,
        "model": "DTW template matcher" if count else None,
        "segmentation_model": segmentation,
        "yolo_character_detector": yolo,
        "message": _training_message(count, segmentation, yolo),
    }


@router.get("/train/status")
async def train_status():
    segmentation = _saved_segmentation_status()
    yolo = yolo_training_status()
    overall_status = (
        "training"
        if yolo["status"] == "running"
        else "complete"
        if predictor.trained_sample_count or segmentation["trained"] or yolo["status"] == "complete"
        else "idle"
    )
    return {
        **training_status,
        "status": overall_status,
        "trained_samples": predictor.trained_sample_count,
        "unique_words": predictor.trained_word_count,
        "model": "DTW template matcher" if predictor.trained_sample_count else None,
        "segmentation_model": segmentation,
        "yolo_character_detector": yolo,
    }


def _train_segmentation_model() -> dict:
    with SessionLocal() as db:
        documents = db.scalars(
            select(UploadDocument).where(UploadDocument.corrected.is_(True))
        ).all()
        examples: list[tuple[float, bool]] = []
        used_documents = 0
        for document in documents:
            db_session = db.get(DbSession, document.session_id)
            if db_session is None:
                continue
            try:
                metadata = json.loads(db_session.notes or "{}")
                stroke_rows = json.loads(document.strokes_json)
            except (json.JSONDecodeError, TypeError):
                continue
            active_ids = metadata.get("active_word_ids", [])
            words = db.scalars(
                select(Word)
                .where(Word.session_id == db_session.id, Word.id.in_(active_ids))
            ).all() if active_ids else []
            assignments = {
                int(stroke_id): (word.line_index, word.id)
                for word in words
                for stroke_id in (word.stroke_ids or "").split(",")
                if stroke_id
            }
            if set(assignments) != {int(stroke["id"]) for stroke in stroke_rows}:
                continue
            stroke_bounds = [
                {
                    "id": stroke["id"],
                    "x_min": stroke["bbox"]["x"],
                    "x_max": stroke["bbox"]["x"] + stroke["bbox"]["width"],
                    "y_min": stroke["bbox"]["y"],
                    "y_max": stroke["bbox"]["y"] + stroke["bbox"]["height"],
                    "width": stroke["width"],
                    "height": stroke["height"],
                }
                for stroke in stroke_rows
            ]
            document_examples = gap_examples_from_corrections(stroke_bounds, assignments)
            if document_examples:
                examples.extend(document_examples)
                used_documents += 1

        threshold = train_gap_threshold(examples)
        if threshold is None:
            existing = db.get(SegmentationModel, 1)
            return {
                "trained": False,
                "training_gaps": len(examples),
                "corrected_documents": used_documents,
                "threshold_factor": existing.gap_threshold_factor if existing else None,
                "message": (
                    "Correct more boxes with both within-word and between-word gaps, then train again."
                ),
            }

        model = db.get(SegmentationModel, 1)
        if model is None:
            model = SegmentationModel(id=1)
            db.add(model)
        model.gap_threshold_factor = threshold
        model.training_gaps = len(examples)
        model.corrected_documents = used_documents
        db.commit()
        set_learned_gap_threshold(threshold)
        return {
            "trained": True,
            "training_gaps": len(examples),
            "corrected_documents": used_documents,
            "threshold_factor": threshold,
            "message": "Learned the word-gap threshold from your corrected boxes.",
        }


def _saved_segmentation_status() -> dict:
    with SessionLocal() as db:
        model = db.get(SegmentationModel, 1)
        if model is None:
            return {"trained": False, "training_gaps": 0, "corrected_documents": 0}
        return {
            "trained": True,
            "training_gaps": model.training_gaps,
            "corrected_documents": model.corrected_documents,
            "threshold_factor": model.gap_threshold_factor,
        }


def _training_message(count: int, segmentation: dict, yolo: dict) -> str:
    messages = []
    if count:
        messages.append(
            f"Loaded {count} saved recognition sample(s) across "
            f"{predictor.trained_word_count} word(s) from SQLite."
        )
    if segmentation["trained"]:
        messages.append(segmentation["message"])
    elif segmentation["message"]:
        messages.append(segmentation["message"])
    if yolo["status"] == "running":
        messages.append("YOLOv11 character detector training started in the background.")
    elif yolo["status"] == "failed":
        messages.append(f"YOLOv11 training failed: {yolo['error']}")
    elif yolo["status"] == "unavailable":
        messages.append(yolo["error"])
    elif yolo["status"] == "complete":
        messages.append("YOLOv11 character detector is ready.")
    return " ".join(messages)
