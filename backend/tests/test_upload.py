from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete, select

from backend.app.main import app
from backend.app.core.segmentation import set_learned_gap_threshold
from backend.app.routers import train as train_router
from backend.app.db import (
    BadSample,
    ModelVersion,
    Sample,
    SegmentationModel,
    Session as DbSession,
    SessionLocal,
    UploadDocument,
    Word,
)
from backend.app.ml import predictor


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        train_router,
        "start_yolo_training",
        lambda: {
            "status": "unavailable",
            "error": "YOLO training disabled in backend tests.",
            "metrics": None,
        },
    )
    with TestClient(app) as test_client:
        with SessionLocal() as db:
            baseline_sessions = set(db.scalars(select(DbSession.id)).all())
            existing_model = db.get(SegmentationModel, 1)
            saved_model = (
                {
                    "gap_threshold_factor": existing_model.gap_threshold_factor,
                    "training_gaps": existing_model.training_gaps,
                    "corrected_documents": existing_model.corrected_documents,
                    "created_at": existing_model.created_at,
                }
                if existing_model
                else None
            )
        try:
            yield test_client
        finally:
            with SessionLocal() as db:
                new_sessions = set(db.scalars(select(DbSession.id)).all()) - baseline_sessions
                if new_sessions:
                    db.execute(delete(UploadDocument).where(UploadDocument.session_id.in_(new_sessions)))
                    db.execute(delete(Sample).where(Sample.session_id.in_(new_sessions)))
                    db.execute(delete(BadSample).where(BadSample.session_id.in_(new_sessions)))
                    db.execute(delete(Word).where(Word.session_id.in_(new_sessions)))
                    db.execute(delete(ModelVersion).where(ModelVersion.session_id.in_(new_sessions)))
                    db.execute(delete(DbSession).where(DbSession.id.in_(new_sessions)))
                model = db.get(SegmentationModel, 1)
                if model is not None:
                    db.delete(model)
                if saved_model is not None:
                    db.add(SegmentationModel(id=1, **saved_model))
                db.commit()
                set_learned_gap_threshold(
                    saved_model["gap_threshold_factor"] if saved_model else None
                )
            predictor.reload_saved_templates()


def test_upload_returns_strokes_and_detected_words(client):
    content = (
        "x,y,pen\n"
        "0,0,1\n2,5,1\n4,10,1\n6,5,1\n8,0,1\n0,0,0\n"
        "100,0,1\n102,5,1\n104,10,1\n106,5,1\n108,0,1\n0,0,0\n"
    )

    response = client.post(
        "/api/upload",
        files={"file": ("sample.csv", content, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["valid_rows"] == 12
    assert payload["stats"]["stroke_count"] == 2
    assert payload["stats"]["word_count"] == 2
    assert len(payload["strokes"][0]["points"]) == 5
    assert len(payload["words"]) == 2

    strokes = [
        {
            "id": stroke["id"],
            "x_min": stroke["bbox"]["x"],
            "x_max": stroke["bbox"]["x"] + stroke["bbox"]["width"],
            "y_min": stroke["bbox"]["y"],
            "y_max": stroke["bbox"]["y"] + stroke["bbox"]["height"],
            "width": stroke["width"],
            "height": stroke["height"],
        }
        for stroke in payload["strokes"]
    ]
    resegmented = client.post(
        f"/api/sessions/{payload['session_id']}/segment",
        json={"strokes": strokes},
    )
    assert resegmented.status_code == 200
    detected_words = resegmented.json()["words"]
    assert len(detected_words) == 2
    assert sorted(stroke_id for word in detected_words for stroke_id in word["stroke_ids"]) == [
        stroke["id"] for stroke in payload["strokes"]
    ]
    persisted = client.get(f"/api/sessions/{payload['session_id']}")
    assert [word["id"] for word in persisted.json()["words"]] == [
        word["id"] for word in detected_words
    ]


def test_resegment_requires_every_stroke_from_upload(client):
    content = (
        "x,y,pen\n"
        "0,0,1\n2,5,1\n4,10,1\n6,5,1\n8,0,1\n0,0,0\n"
        "100,0,1\n102,5,1\n104,10,1\n106,5,1\n108,0,1\n0,0,0\n"
    )
    payload = client.post(
        "/api/upload",
        files={"file": ("sample.csv", content, "text/csv")},
    ).json()
    stroke = payload["strokes"][0]

    response = client.post(
        f"/api/sessions/{payload['session_id']}/segment",
        json={
            "strokes": [
                {
                    "id": stroke["id"],
                    "x_min": stroke["bbox"]["x"],
                    "x_max": stroke["bbox"]["x"] + stroke["bbox"]["width"],
                    "y_min": stroke["bbox"]["y"],
                    "y_max": stroke["bbox"]["y"] + stroke["bbox"]["height"],
                    "width": stroke["width"],
                    "height": stroke["height"],
                }
            ]
        },
    )

    assert response.status_code == 422
    assert "every stroke" in response.json()["detail"]


def _four_stroke_document(scale: float = 1.0) -> str:
    rows = ["x,y,pen"]
    for start_x in (0, 12, 60, 72):
        x = start_x * scale
        for px, py in ((x, 0), (x + 5 * scale, 10 * scale), (x + 10 * scale, 20 * scale)):
            rows.append(f"{px},{py},1")
        rows.append("0,0,0")
    return "\n".join(rows)


def _box_for_strokes(strokes, stroke_ids):
    chosen = [stroke for stroke in strokes if stroke["id"] in stroke_ids]
    min_x = min(stroke["bbox"]["x"] for stroke in chosen)
    min_y = min(stroke["bbox"]["y"] for stroke in chosen)
    max_x = max(stroke["bbox"]["x"] + stroke["bbox"]["width"] for stroke in chosen)
    max_y = max(stroke["bbox"]["y"] + stroke["bbox"]["height"] for stroke in chosen)
    return {
        "x": min_x,
        "y": min_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def test_corrected_boxes_restore_and_train_word_boundary_model(client):
    set_learned_gap_threshold(None)
    content = _four_stroke_document()
    uploaded = client.post(
        "/api/upload",
        files={"file": ("annotated.txt", content, "text/plain")},
    ).json()
    assert len(uploaded["words"]) == 2

    corrections = [
        {
            "id": uploaded["words"][0]["id"],
            "line_index": 0,
            "word_index": 0,
            "stroke_ids": [0, 1],
            "bbox": _box_for_strokes(uploaded["strokes"], [0, 1]),
        },
        {
            "id": uploaded["words"][1]["id"],
            "line_index": 0,
            "word_index": 1,
            "stroke_ids": [2, 3],
            "bbox": _box_for_strokes(uploaded["strokes"], [2, 3]),
        },
    ]
    corrected = client.put(
        f"/api/sessions/{uploaded['session_id']}/words",
        json={"words": corrections},
    )
    assert corrected.status_code == 200

    trained = client.post("/api/train")
    assert trained.status_code == 200
    result = trained.json()["segmentation_model"]
    assert result["trained"] is True
    assert result["training_gaps"] == 3

    restored = client.post(
        "/api/upload",
        files={"file": ("annotated-again.txt", content, "text/plain")},
    ).json()
    assert restored["stats"]["restored"] is True
    assert restored["session_id"] == uploaded["session_id"]
    assert [word["stroke_ids"] for word in restored["words"]] == [[0, 1], [2, 3]]

    set_learned_gap_threshold(None)
    with TestClient(app) as restarted_client:
        status = restarted_client.get("/api/train/status").json()
        assert status["segmentation_model"]["trained"] is True
        scaled = restarted_client.post(
            "/api/upload",
            files={
                "file": (
                    "new-document.txt",
                    _four_stroke_document(scale=2),
                    "text/plain",
                )
            },
        ).json()
    assert len(scaled["words"]) == 2
    assert [word["stroke_ids"] for word in scaled["words"]] == [[0, 1], [2, 3]]


def test_upload_reports_files_without_valid_rows(client):
    response = client.post(
        "/api/upload",
        files={"file": ("empty.txt", "not pen data", "text/plain")},
    )

    assert response.status_code == 422
    assert "No valid" in response.json()["detail"]


def test_save_bad_sample_train_and_predict(client):
    content = (
        "x,y,pen\n"
        "0,0,1\n2,5,1\n4,10,1\n6,5,1\n8,0,1\n0,0,0\n"
    )
    upload_response = client.post(
        "/api/upload",
        files={"file": ("sample.csv", content, "text/csv")},
    )
    assert upload_response.status_code == 200
    upload = upload_response.json()
    word = upload["words"][0]
    points = upload["strokes"][0]["points"]

    save_response = client.post(
        "/api/samples",
        json={
            "session_id": upload["session_id"],
            "word_id": word["id"],
            "text": "ক",
            "chars": ["ক"],
            "points": points,
        },
    )
    assert save_response.status_code == 200
    assert save_response.json()["saved"] is True

    bad_response = client.post(
        "/api/samples/bad",
        json={
            "session_id": upload["session_id"],
            "word_id": word["id"],
            "points": points,
        },
    )
    assert bad_response.status_code == 200

    train_response = client.post("/api/train")
    assert train_response.status_code == 200
    assert train_response.json()["trained_samples"] >= 1
    assert train_response.json()["model"] == "DTW template matcher"
    assert train_response.json()["yolo_character_detector"]["status"] == "unavailable"

    predictor.train_template_matcher([])
    assert predictor.trained_sample_count == 0
    with TestClient(app) as restarted_client:
        status_response = restarted_client.get("/api/train/status")
        assert status_response.status_code == 200
        assert status_response.json()["trained_samples"] >= 1
        assert status_response.json()["status"] == "complete"

        prediction_response = restarted_client.post(
            f"/api/predict/{word['id']}",
            json={"points": points},
        )
        assert prediction_response.status_code == 200
        assert prediction_response.json()["text"] == "ক"


def test_word_box_corrections_merge_split_and_persist(client):
    content = (
        "x,y,pen\n"
        "0,0,1\n2,5,1\n4,10,1\n6,5,1\n8,0,1\n0,0,0\n"
        "100,0,1\n102,5,1\n104,10,1\n106,5,1\n108,0,1\n0,0,0\n"
    )
    uploaded = client.post(
        "/api/upload",
        files={"file": ("sample.csv", content, "text/csv")},
    ).json()
    first, second = uploaded["words"]
    merged_box = {
        "x": first["bbox"]["x"],
        "y": min(first["bbox"]["y"], second["bbox"]["y"]),
        "width": second["bbox"]["x"] + second["bbox"]["width"] - first["bbox"]["x"],
        "height": max(first["bbox"]["y"] + first["bbox"]["height"],
                      second["bbox"]["y"] + second["bbox"]["height"])
        - min(first["bbox"]["y"], second["bbox"]["y"]),
    }

    merged_response = client.put(
        f"/api/sessions/{uploaded['session_id']}/words",
        json={
            "words": [
                {
                    "id": first["id"],
                    "line_index": 0,
                    "word_index": 0,
                    "stroke_ids": first["stroke_ids"] + second["stroke_ids"],
                    "bbox": merged_box,
                }
            ]
        },
    )
    assert merged_response.status_code == 200

    split_response = client.put(
        f"/api/sessions/{uploaded['session_id']}/words",
        json={
            "words": [
                {
                    **first,
                    "word_index": 0,
                },
                {
                    **second,
                    "id": None,
                    "word_index": 1,
                },
            ]
        },
    )
    assert split_response.status_code == 200
    assert len(split_response.json()["words"]) == 2

    persisted = client.get(f"/api/sessions/{uploaded['session_id']}")
    assert persisted.status_code == 200
    assert [word["stroke_ids"] for word in persisted.json()["words"]] == [
        first["stroke_ids"],
        second["stroke_ids"],
    ]

    duplicate_stroke_response = client.put(
        f"/api/sessions/{uploaded['session_id']}/words",
        json={
            "words": [
                {
                    "id": first["id"],
                    "line_index": 0,
                    "word_index": 0,
                    "stroke_ids": first["stroke_ids"],
                    "bbox": first["bbox"],
                },
                {
                    "id": second["id"],
                    "line_index": 0,
                    "word_index": 1,
                    "stroke_ids": first["stroke_ids"],
                    "bbox": second["bbox"],
                },
            ]
        },
    )
    assert duplicate_stroke_response.status_code == 422
