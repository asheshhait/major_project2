from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw

from ..config import BASE_DIR, CHECKPOINT_DIR, DATA_DIR
from ..core.parser import parse_csv_like
from ..core.segmentation import (
    WordBoxData,
    get_learned_gap_threshold,
)


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 256
MAX_INFERENCE_WIDTH = 4096
MAX_INFERENCE_HEIGHT = 2048
CHARACTER_HEIGHT = 118
TRAIN_COMPOSITES = 2400
VALIDATION_COMPOSITES = 480
YOLO_IMAGE_SIZE = 640
YOLO_EPOCHS = 30
MAX_SAMPLES_PER_CLASS = 80
BENGALI_RANGE = range(0x0980, 0x0A00)
DEFAULT_DATASET_DIR = BASE_DIR.parent / "Canvas" / "OHR_Bangla_Characters_Checked"
DATASET_DIR = Path(os.environ.get("BANGLA_CHARACTER_DATASET_DIR", DEFAULT_DATASET_DIR))
YOLO_DATASET_DIR = DATA_DIR / "yolo11_character_dataset"
YOLO_MODEL_PATH = CHECKPOINT_DIR / "yolo11-character.pt"
YOLO_METRICS_PATH = CHECKPOINT_DIR / "yolo11-character-metrics.json"

_detector: Any = None
_detector_lock = threading.Lock()
_training_lock = threading.Lock()
_training_state: dict[str, Any] = {
    "status": "idle",
    "error": None,
    "metrics": None,
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharacterSample:
    label: str
    strokes: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class CharacterDetection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_index: int


def bengali_character_classes(dataset_dir: Path = DATASET_DIR) -> list[str]:
    if not dataset_dir.is_dir():
        return []
    return sorted(
        folder.name
        for folder in dataset_dir.iterdir()
        if folder.is_dir() and any(ord(character) in BENGALI_RANGE for character in folder.name)
    )


def load_character_samples(
    dataset_dir: Path = DATASET_DIR,
) -> tuple[list[str], dict[str, list[CharacterSample]]]:
    classes = bengali_character_classes(dataset_dir)
    samples: dict[str, list[CharacterSample]] = {}
    for class_index, label in enumerate(classes):
        class_samples = []
        paths = sorted((dataset_dir / label).glob("*.txt"))
        if len(paths) > MAX_SAMPLES_PER_CLASS:
            random.Random(173 + class_index).shuffle(paths)
            paths = paths[:MAX_SAMPLES_PER_CLASS]
        for path in paths:
            text = path.read_text(encoding="utf-8-sig")
            points, _ = parse_csv_like(text)
            strokes = _split_trace(points)
            if sum(len(stroke) for stroke in strokes) >= 3:
                class_samples.append(CharacterSample(label, strokes))
        if len(class_samples) >= 3:
            samples[label] = class_samples
    retained_classes = [label for label in classes if label in samples]
    return retained_classes, samples


def build_yolo_character_dataset(
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = YOLO_DATASET_DIR,
    train_images: int = TRAIN_COMPOSITES,
    validation_images: int = VALIDATION_COMPOSITES,
    seed: int = 173,
) -> dict[str, Any]:
    classes, samples_by_class = load_character_samples(dataset_dir)
    if len(classes) < 2:
        raise ValueError(
            f"Need at least two Bengali character folders with usable trace files in {dataset_dir}."
        )

    train_samples: dict[str, list[CharacterSample]] = {}
    validation_samples: dict[str, list[CharacterSample]] = {}
    split_rng = random.Random(seed)
    for label, samples in samples_by_class.items():
        shuffled = list(samples)
        split_rng.shuffle(shuffled)
        split_at = max(1, int(len(shuffled) * 0.8))
        split_at = min(split_at, len(shuffled) - 1)
        train_samples[label] = shuffled[:split_at]
        validation_samples[label] = shuffled[split_at:]

    if output_dir.exists():
        output_dir = output_dir.with_name(f"{output_dir.name}-{uuid4().hex[:8]}")
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    _write_composites(output_dir, "train", classes, train_samples, train_images, rng)
    _write_composites(output_dir, "val", classes, validation_samples, validation_images, rng)
    data_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": classes,
    }
    (output_dir / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {json.dumps(data_yaml['path'], ensure_ascii=False)}",
                "train: images/train",
                "val: images/val",
                "names: " + json.dumps(classes, ensure_ascii=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "classes": classes,
        "training_images": train_images,
        "validation_images": validation_images,
        "dataset_yaml": str(output_dir / "dataset.yaml"),
    }


def _write_composites(
    output_dir: Path,
    split: str,
    classes: list[str],
    samples_by_class: dict[str, list[CharacterSample]],
    image_count: int,
    rng: random.Random,
) -> None:
    class_indices = {label: index for index, label in enumerate(classes)}
    labels = list(samples_by_class)
    for image_index in range(image_count):
        image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
        draw = ImageDraw.Draw(image)
        annotations: list[str] = []
        cursor_x = 28.0
        word_gap = rng.uniform(60.0, 82.0)
        character_count = rng.randint(5, 10)
        word_lengths = _random_word_lengths(character_count, rng)

        for word_index, word_length in enumerate(word_lengths):
            if word_index:
                cursor_x += word_gap
            for _ in range(word_length):
                label = rng.choice(labels)
                sample = rng.choice(samples_by_class[label])
                scale = rng.uniform(0.82, 1.14) * CHARACTER_HEIGHT / _sample_height(sample)
                strokes = [
                    [(x * scale, y * scale) for x, y in stroke]
                    for stroke in sample.strokes
                ]
                points = [point for stroke in strokes for point in stroke]
                min_x = min(x for x, _ in points)
                max_x = max(x for x, _ in points)
                min_y = min(y for _, y in points)
                max_y = max(y for _, y in points)
                offset_x = cursor_x - min_x
                baseline = 188 + rng.uniform(-5, 5)
                offset_y = baseline - max_y
                strokes = [
                    [(x + offset_x, y + offset_y) for x, y in stroke]
                    for stroke in strokes
                ]
                points = [point for stroke in strokes for point in stroke]
                for stroke in strokes:
                    for start, end in zip(stroke, stroke[1:]):
                        draw.line((start, end), fill="black", width=3)
                box_x_min = min(x for x, _ in points)
                box_x_max = max(x for x, _ in points)
                box_y_min = min(y for _, y in points)
                box_y_max = max(y for _, y in points)
                center_x = (box_x_min + box_x_max) / (2 * IMAGE_WIDTH)
                center_y = (box_y_min + box_y_max) / (2 * IMAGE_HEIGHT)
                box_width = (box_x_max - box_x_min) / IMAGE_WIDTH
                box_height = (box_y_max - box_y_min) / IMAGE_HEIGHT
                annotations.append(
                    f"{class_indices[label]} {center_x:.6f} {center_y:.6f} "
                    f"{box_width:.6f} {box_height:.6f}"
                )
                cursor_x = box_x_max + rng.uniform(5.0, 18.0)
                if cursor_x > IMAGE_WIDTH - 100:
                    break
            if cursor_x > IMAGE_WIDTH - 100:
                break

        image_path = output_dir / "images" / split / f"{image_index:06d}.jpg"
        image.save(image_path, quality=94)
        label_path = output_dir / "labels" / split / f"{image_index:06d}.txt"
        label_path.write_text("\n".join(annotations) + "\n", encoding="utf-8")


def _sample_height(sample: CharacterSample) -> float:
    points = [point for stroke in sample.strokes for point in stroke]
    height = max(y for _, y in points) - min(y for _, y in points)
    return max(height, 1.0)


def _split_trace(points) -> tuple[tuple[tuple[float, float], ...], ...]:
    strokes: list[tuple[tuple[float, float], ...]] = []
    current: list[tuple[float, float]] = []
    for point in points:
        if point.pen == 1:
            current.append((point.x, point.y))
        elif current:
            strokes.append(tuple(current))
            current = []
    if current:
        strokes.append(tuple(current))
    return tuple(stroke for stroke in strokes if len(stroke) >= 2)


def _random_word_lengths(character_count: int, rng: random.Random) -> list[int]:
    lengths: list[int] = []
    remaining = character_count
    while remaining:
        maximum = min(4, remaining)
        minimum = 1 if not lengths else 2
        if maximum < minimum:
            lengths[-1] += remaining
            break
        length = rng.randint(minimum, maximum)
        lengths.append(length)
        remaining -= length
    return lengths


def train_yolo_character_detector(
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = YOLO_DATASET_DIR,
    checkpoint_path: Path = YOLO_MODEL_PATH,
    runs_dir: Path = DATA_DIR / "yolo11_runs",
    train_images: int = TRAIN_COMPOSITES,
    validation_images: int = VALIDATION_COMPOSITES,
    epochs: int = YOLO_EPOCHS,
) -> dict[str, Any]:
    from ultralytics import YOLO

    use_cuda = _cuda_available()
    if not use_cuda:
        train_images = min(train_images, 800)
        validation_images = min(validation_images, 160)
        epochs = min(epochs, 12)
    image_size = YOLO_IMAGE_SIZE if use_cuda else 512
    dataset = build_yolo_character_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        train_images=train_images,
        validation_images=validation_images,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    initial_weights = checkpoint_path.with_name("yolo11n.pt")
    if initial_weights.is_file():
        model = YOLO(str(initial_weights))
    else:
        model = YOLO("yolo11n.pt")
        downloaded_weights = Path("yolo11n.pt")
        if downloaded_weights.is_file():
            shutil.copy2(downloaded_weights, initial_weights)
    model.train(
        data=dataset["dataset_yaml"],
        epochs=epochs,
        imgsz=image_size,
        batch=8 if use_cuda else 4,
        workers=0,
        device=0 if use_cuda else "cpu",
        project=str(runs_dir.resolve()),
        name=f"bangla-character-detector-{uuid4().hex[:8]}",
        exist_ok=False,
        patience=8,
        seed=173,
        plots=False,
        verbose=False,
    )
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise RuntimeError(f"YOLO training did not produce best weights at {best_weights}.")
    shutil.copy2(best_weights, checkpoint_path)

    validation = model.val(
        data=dataset["dataset_yaml"],
        imgsz=image_size,
        batch=8 if use_cuda else 4,
        workers=0,
        device=0 if use_cuda else "cpu",
        plots=False,
        verbose=False,
    )
    metrics = {
        **dataset,
        "epochs": epochs,
        "image_size": image_size,
        "model": "YOLOv11n character detector",
        "validation_map50": float(validation.box.map50),
        "validation_map50_95": float(validation.box.map),
        "validation_note": "Metrics are on synthetic layouts, not real handwritten sentences.",
        "checkpoint": str(checkpoint_path),
    }
    metrics_path = checkpoint_path.with_name("yolo11-character-metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    load_character_detector(checkpoint_path)
    return metrics


def start_yolo_training() -> dict[str, Any]:
    with _training_lock:
        if _training_state["status"] == "running":
            return dict(_training_state)
        if not bengali_character_classes():
            _training_state.update(
                status="unavailable",
                error=f"No Bengali character dataset found at {DATASET_DIR}.",
                metrics=None,
            )
            return dict(_training_state)
        _training_state.update(status="running", error=None, metrics=None)
        thread = threading.Thread(target=_train_in_background, daemon=True)
        thread.start()
        return dict(_training_state)


def _train_in_background() -> None:
    try:
        metrics = train_yolo_character_detector()
    except Exception as error:
        logger.exception("YOLOv11 character detector training failed")
        with _training_lock:
            _training_state.update(
                status="failed",
                error=str(error),
                metrics=None,
            )
        return
    with _training_lock:
        _training_state.update(status="complete", error=None, metrics=metrics)


def yolo_training_status() -> dict[str, Any]:
    with _training_lock:
        state = dict(_training_state)
    if state["status"] in {"idle", "unavailable"} and YOLO_METRICS_PATH.is_file():
        return {"status": "complete", "error": None, "metrics": yolo_metrics()}
    return state


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def load_character_detector(checkpoint_path: Path = YOLO_MODEL_PATH) -> bool:
    global _detector
    if not checkpoint_path.is_file():
        return False
    from ultralytics import YOLO

    with _detector_lock:
        _detector = YOLO(str(checkpoint_path))
    return True


def yolo_metrics() -> dict[str, Any]:
    if not YOLO_METRICS_PATH.is_file():
        return {"trained": False}
    try:
        metrics = json.loads(YOLO_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Saved YOLO metrics are invalid.") from error
    return {"trained": True, **metrics}


def segment_strokes_with_yolo(
    strokes: list[dict[str, Any]],
) -> list[WordBoxData] | None:
    if _detector is None or not strokes or any(not stroke.get("points") for stroke in strokes):
        return None

    image, scale, min_x, min_y = _rasterize_strokes(strokes)
    with _detector_lock:
        prediction = _detector.predict(
            source=image,
            imgsz=YOLO_IMAGE_SIZE,
            conf=0.15,
            iou=0.5,
            verbose=False,
            device=0 if _cuda_available() else "cpu",
        )[0]
    if prediction.boxes is None or len(prediction.boxes) == 0:
        return None

    detections = []
    for values, confidence, class_index in zip(
        prediction.boxes.xyxy.cpu().tolist(),
        prediction.boxes.conf.cpu().tolist(),
        prediction.boxes.cls.cpu().tolist(),
    ):
        detections.append(
            CharacterDetection(
                x_min=values[0] / scale + min_x,
                y_min=values[1] / scale + min_y,
                x_max=values[2] / scale + min_x,
                y_max=values[3] / scale + min_y,
                confidence=confidence,
                class_index=int(class_index),
            )
        )
    return group_character_detections(detections, strokes)


def _rasterize_strokes(
    strokes: list[dict[str, Any]],
) -> tuple[Image.Image, float, float, float]:
    all_points = [
        point
        for stroke in strokes
        for point in stroke.get("points", [])
    ]
    min_x = min(float(point["x"]) for point in all_points)
    max_x = max(float(point["x"]) for point in all_points)
    min_y = min(float(point["y"]) for point in all_points)
    max_y = max(float(point["y"]) for point in all_points)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    scale = min(
        1.0,
        (MAX_INFERENCE_WIDTH - 32) / width,
        (MAX_INFERENCE_HEIGHT - 32) / height,
    )
    image_width = max(32, math.ceil(width * scale + 32))
    image_height = max(32, math.ceil(height * scale + 32))
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)
    offset_x = (image_width - width * scale) / 2
    offset_y = (image_height - height * scale) / 2
    for stroke in strokes:
        points = [
            (
                (float(point["x"]) - min_x) * scale + offset_x,
                (float(point["y"]) - min_y) * scale + offset_y,
            )
            for point in stroke.get("points", [])
        ]
        for start, end in zip(points, points[1:]):
            draw.line((start, end), fill="black", width=max(2, round(scale * 2)))
    return image, scale, min_x - offset_x / scale, min_y - offset_y / scale


def group_character_detections(
    detections: list[CharacterDetection],
    strokes: list[dict[str, Any]],
) -> list[WordBoxData]:
    """Combine character detections into non-overlapping word regions by line gaps."""
    if not detections or not strokes:
        return []
    median_height = median(max(item.y_max - item.y_min, 1.0) for item in detections)
    lines: list[list[CharacterDetection]] = []
    for detection in sorted(detections, key=lambda item: (item.y_min + item.y_max) / 2):
        center_y = (detection.y_min + detection.y_max) / 2
        candidates = [
            line for line in lines
            if abs(
                center_y
                - median((item.y_min + item.y_max) / 2 for item in line)
            ) <= median_height * 0.65
        ]
        if candidates:
            candidates[0].append(detection)
        else:
            lines.append([detection])

    bounds = []
    for line_index, line in enumerate(lines):
        line.sort(key=lambda item: item.x_min)
        char_heights = [max(item.y_max - item.y_min, 1.0) for item in line]
        learned_factor = get_learned_gap_threshold()
        gap_threshold = median(char_heights) * (
            learned_factor if learned_factor is not None else 0.38
        )
        groups: list[list[CharacterDetection]] = []
        for character in line:
            if not groups or character.x_min - groups[-1][-1].x_max > gap_threshold:
                groups.append([character])
            else:
                groups[-1].append(character)

        for word_index, group in enumerate(groups):
            x_min = min(item.x_min for item in group)
            x_max = max(item.x_max for item in group)
            y_min = min(item.y_min for item in group)
            y_max = max(item.y_max for item in group)
            padding = median_height * 0.06
            bounds.append(
                (
                    line_index,
                    word_index,
                    x_min - padding,
                    y_min - padding,
                    x_max + padding,
                    y_max + padding,
                )
            )

    assigned_strokes: list[list[int]] = [[] for _ in bounds]
    stroke_centers = {
        int(stroke["id"]): (
            (float(stroke["x_min"]) + float(stroke["x_max"])) / 2,
            (float(stroke["y_min"]) + float(stroke["y_max"])) / 2,
        )
        for stroke in strokes
    }
    for stroke_id, (center_x, center_y) in stroke_centers.items():
        distances = [
            _point_to_rect_distance(center_x, center_y, x_min, y_min, x_max, y_max)
            for _, _, x_min, y_min, x_max, y_max in bounds
        ]
        assigned_strokes[min(range(len(distances)), key=distances.__getitem__)].append(
            stroke_id
        )

    word_boxes: list[WordBoxData] = []
    for word_index, (line_index, group_index, x_min, y_min, x_max, y_max) in enumerate(bounds):
        stroke_ids = sorted(assigned_strokes[word_index])
        if not stroke_ids:
            continue
        word_boxes.append(
            WordBoxData(
                id=len(word_boxes),
                line_index=line_index,
                word_index=group_index,
                stroke_ids=stroke_ids,
                bbox={
                    "x": x_min,
                    "y": y_min,
                    "width": max(x_max - x_min, 1.0),
                    "height": max(y_max - y_min, 1.0),
                },
            )
        )
    return word_boxes


def _point_to_rect_distance(
    x: float,
    y: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> float:
    dx = max(x_min - x, 0.0, x - x_max)
    dy = max(y_min - y, 0.0, y - y_max)
    return dx * dx + dy * dy


def detect_word_boxes(strokes: list[dict[str, Any]]) -> list[WordBoxData] | None:
    result = segment_strokes_with_yolo(strokes)
    return result if result else None


def load_saved_character_detector() -> bool:
    return load_character_detector(YOLO_MODEL_PATH)
