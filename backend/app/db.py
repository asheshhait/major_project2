from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, DateTime, JSON, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import APP_CONFIG, DATA_DIR


class Base(DeclarativeBase):
    pass


DATA_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(
    APP_CONFIG.database_url,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    words: Mapped[list["Word"]] = relationship(back_populates="session")
    samples: Mapped[list["Sample"]] = relationship(back_populates="session")
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="session")


class UploadDocument(Base):
    __tablename__ = "upload_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), unique=True, nullable=False)
    strokes_json: Mapped[str] = mapped_column(Text, nullable=False)
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    corrected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class SegmentationModel(Base):
    __tablename__ = "segmentation_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    gap_threshold_factor: Mapped[float] = mapped_column(nullable=False)
    training_gaps: Mapped[int] = mapped_column(Integer, nullable=False)
    corrected_documents: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class Word(Base):
    __tablename__ = "words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    text: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    line_index: Mapped[int] = mapped_column(Integer, default=0)
    word_index: Mapped[int] = mapped_column(Integer, default=0)
    bbox_json: Mapped[dict] = mapped_column(JSON, default=dict)
    stroke_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session: Mapped[Session] = relationship(back_populates="words")
    samples: Mapped[list["Sample"]] = relationship(back_populates="word")


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    word_id: Mapped[Optional[int]] = mapped_column(ForeignKey("words.id"), nullable=True)
    text: Mapped[str] = mapped_column(String(255), nullable=False)
    chars: Mapped[str] = mapped_column(String(255), nullable=False)
    stroke_to_char: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    raw_points: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_bad: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session: Mapped[Session] = relationship(back_populates="samples")
    word: Mapped[Optional[Word]] = relationship(back_populates="samples")


class BadSample(Base):
    __tablename__ = "bad_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"), nullable=False)
    raw_points: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    checkpoint_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session: Mapped[Session] = relationship(back_populates="model_versions")
