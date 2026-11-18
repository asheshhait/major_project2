from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .config import APP_CONFIG
from .core.segmentation import set_learned_gap_threshold
from .db import SegmentationModel, SessionLocal, initialize_database
from .ml import predictor
from .ml.yolo_character import load_saved_character_detector
from .routers import predict, samples, train, upload, words


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    predictor.reload_saved_templates()
    load_saved_character_detector()
    with SessionLocal() as db:
        model = db.get(SegmentationModel, 1)
        set_learned_gap_threshold(model.gap_threshold_factor if model else None)
    yield


app = FastAPI(
    title="Online Bangla Handwriting Word Detection and Incremental Learning",
    version=APP_CONFIG.app_version,
    description="A FastAPI + PyTorch project scaffold for incremental Bangla handwriting learning.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (upload.router, words.router, samples.router, train.router, predict.router):
    app.include_router(router)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def index():
    """Serve the frontend entry page."""
    return {"message": "Bangla OCR app is running. Open the frontend at /static/index.html"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "app": APP_CONFIG.app_name}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)
