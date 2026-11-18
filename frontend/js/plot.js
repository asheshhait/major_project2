export class PlotCanvas {
  static MIN_ZOOM = 0.5;
  static MAX_ZOOM = 16;

  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.session = null;
    this.selectedWordIndex = 0;
    this.selectedWordIndices = [];
    this.viewMode = "page";
    this.editMode = "select";
    this.callbacks = {};
    this.pointerInteraction = null;
    this.draftBox = null;
    this.viewTransform = null;
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);

    const zoomSurface = canvas.parentElement || canvas;
    zoomSurface.addEventListener("wheel", (event) => {
      if (event.cancelable) event.preventDefault();
      const bounds = canvas.getBoundingClientRect();
      const lineHeight = Number.parseFloat(getComputedStyle(canvas).lineHeight) || 16;
      const deltaY = event.deltaMode === WheelEvent.DOM_DELTA_LINE
        ? event.deltaY * lineHeight
        : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
          ? event.deltaY * canvas.clientHeight
          : event.deltaY;
      this.zoomAt(
        Math.max(0, Math.min(canvas.clientWidth, event.clientX - bounds.left)),
        Math.max(0, Math.min(canvas.clientHeight, event.clientY - bounds.top)),
        Math.exp(-deltaY * 0.002),
      );
    }, { passive: false });

    canvas.addEventListener("pointerdown", (event) => this._pointerDown(event));
    canvas.addEventListener("pointermove", (event) => this._pointerMove(event));
    canvas.addEventListener("pointerup", (event) => this._pointerFinish(event));
    canvas.addEventListener("pointercancel", (event) => this._pointerFinish(event));
  }

  setCallbacks(callbacks) {
    this.callbacks = callbacks;
  }

  setSession(session) {
    this.session = session;
    this.selectedWordIndex = 0;
    this.selectedWordIndices = [];
    this.viewMode = "word";
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.draw();
  }

  setEditMode(mode) {
    this.editMode = mode;
    this.viewMode = "page";
    this.draftBox = null;
    this.canvas.classList.toggle("is-box-creating", mode === "create");
    this.draw();
  }

  setSelectedWordIndices(indices) {
    this.selectedWordIndices = [...indices];
    this.draw();
  }

  zoomAt(x, y, factor) {
    const nextZoom = Math.min(
      PlotCanvas.MAX_ZOOM,
      Math.max(PlotCanvas.MIN_ZOOM, this.zoom * factor),
    );
    const actualFactor = nextZoom / this.zoom;
    const centerX = this.canvas.clientWidth / 2;
    const centerY = this.canvas.clientHeight / 2;
    this.panX = (x - centerX) - (x - centerX - this.panX) * actualFactor;
    this.panY = (y - centerY) - (y - centerY - this.panY) * actualFactor;
    this.zoom = nextZoom;
    this.draw();
  }

  zoomBy(factor) {
    this.zoomAt(this.canvas.clientWidth / 2, this.canvas.clientHeight / 2, factor);
  }

  resetView() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.draw();
  }

  selectWord(index) {
    this.selectedWordIndex = index;
    this.selectedWordIndices = [index];
    this.draw();
  }

  setViewMode(mode) {
    this.viewMode = mode;
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.draw();
  }

  draw() {
    const { ctx, canvas, session } = this;
    const width = Math.max(canvas.clientWidth, 1);
    const height = Math.max(canvas.clientHeight, 1);
    const zoomLabel = document.querySelector("#zoom-level");
    if (zoomLabel) zoomLabel.textContent = `${Math.round(this.zoom * 100)}%`;
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--panel-alt").trim() || "#edf3ff";
    ctx.fillRect(0, 0, width, height);

    if (!session?.strokes?.length) {
      this.viewTransform = null;
      ctx.fillStyle = "#72809a";
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Uploaded handwriting will appear here", width / 2, height / 2);
      return;
    }

    const selectedWord = session.words[this.selectedWordIndex];
    const selectedStrokeIds = new Set(selectedWord?.stroke_ids || []);
    const visibleStrokes = this.viewMode === "word" && selectedWord
      ? session.strokes.filter((stroke) => selectedStrokeIds.has(stroke.id))
      : session.strokes;
    const visibleWords = this.viewMode === "word" && selectedWord
      ? [selectedWord]
      : session.words;
    const coordinates = visibleStrokes.flatMap((stroke) => stroke.points);
    const minX = Math.min(...coordinates.map((point) => point.x));
    const maxX = Math.max(...coordinates.map((point) => point.x));
    const minY = Math.min(...coordinates.map((point) => point.y));
    const maxY = Math.max(...coordinates.map((point) => point.y));
    const padding = 28;
    const fitScale = Math.min(
      (width - padding * 2) / Math.max(maxX - minX, 1),
      (height - padding * 2) / Math.max(maxY - minY, 1),
    );
    const scale = fitScale * this.zoom;
    const offsetX = width / 2 - ((maxX - minX) * scale) / 2 + this.panX;
    const offsetY = height / 2 - ((maxY - minY) * scale) / 2 + this.panY;
    this.viewTransform = { minX, minY, scale, offsetX, offsetY };
    const toCanvas = (x, y) => ({
      x: offsetX + (x - minX) * scale,
      y: offsetY + (y - minY) * scale,
    });

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 2;
    visibleStrokes.forEach((stroke, strokeIndex) => {
      if (stroke.points.length < 2) return;
      ctx.beginPath();
      stroke.points.forEach((point, pointIndex) => {
        const mapped = toCanvas(point.x, point.y);
        if (pointIndex === 0) ctx.moveTo(mapped.x, mapped.y);
        else ctx.lineTo(mapped.x, mapped.y);
      });
      ctx.strokeStyle = `hsl(${(strokeIndex * 47) % 360} 65% 43%)`;
      ctx.stroke();
    });

    visibleWords.forEach((word, visibleIndex) => {
      const wordIndex = this.viewMode === "word" ? this.selectedWordIndex : visibleIndex;
      const { x, y, width: boxWidth, height: boxHeight } = word.bbox;
      const topLeft = toCanvas(x, y);
      const selected = this.selectedWordIndices.includes(wordIndex);
      ctx.strokeStyle = selected ? "#5277ff" : "#ef6262";
      ctx.lineWidth = selected ? 3.5 : 2;
      ctx.setLineDash(selected ? [] : [7, 5]);
      ctx.strokeRect(topLeft.x, topLeft.y, boxWidth * scale, boxHeight * scale);
      ctx.setLineDash([]);
      ctx.fillStyle = selected ? "#5277ff" : "#ef6262";
      ctx.beginPath();
      ctx.arc(topLeft.x + 13, topLeft.y - 9, 13, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 14px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(wordIndex + 1), topLeft.x + 13, topLeft.y - 9);
    });

    if (this.draftBox) {
      ctx.fillStyle = "rgba(82, 119, 255, 0.12)";
      ctx.fillRect(this.draftBox.x, this.draftBox.y, this.draftBox.width, this.draftBox.height);
      ctx.strokeStyle = "#5277ff";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(this.draftBox.x, this.draftBox.y, this.draftBox.width, this.draftBox.height);
      ctx.setLineDash([]);
    }
  }

  _canvasPosition(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  _toData(x, y) {
    if (!this.viewTransform) return null;
    return {
      x: (x - this.viewTransform.offsetX) / this.viewTransform.scale + this.viewTransform.minX,
      y: (y - this.viewTransform.offsetY) / this.viewTransform.scale + this.viewTransform.minY,
    };
  }

  _wordAt(x, y) {
    if (!this.session || !this.viewTransform) return -1;
    const point = this._toData(x, y);
    for (let index = this.session.words.length - 1; index >= 0; index -= 1) {
      const { x: boxX, y: boxY, width, height } = this.session.words[index].bbox;
      if (point.x >= boxX && point.x <= boxX + width && point.y >= boxY && point.y <= boxY + height) {
        return index;
      }
    }
    return -1;
  }

  _pointerDown(event) {
    const position = this._canvasPosition(event);
    if (this.editMode === "create") {
      this.pointerInteraction = { kind: "create", pointerId: event.pointerId, start: position };
      this.draftBox = { x: position.x, y: position.y, width: 0, height: 0 };
    } else {
      const wordIndex = this._wordAt(position.x, position.y);
      if (wordIndex !== -1) {
        this.pointerInteraction = {
          kind: "word",
          pointerId: event.pointerId,
          wordIndex,
          startX: event.clientX,
          startY: event.clientY,
        };
      } else {
        this.pointerInteraction = {
          kind: "pan",
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          panX: this.panX,
          panY: this.panY,
        };
        this.canvas.classList.add("is-dragging");
      }
    }
    this.canvas.setPointerCapture(event.pointerId);
  }

  _pointerMove(event) {
    const interaction = this.pointerInteraction;
    if (!interaction || interaction.pointerId !== event.pointerId) return;
    const position = this._canvasPosition(event);
    if (interaction.kind === "pan") {
      this.panX = interaction.panX + event.clientX - interaction.startX;
      this.panY = interaction.panY + event.clientY - interaction.startY;
    } else if (interaction.kind === "create") {
      this.draftBox = {
        x: Math.min(interaction.start.x, position.x),
        y: Math.min(interaction.start.y, position.y),
        width: Math.abs(position.x - interaction.start.x),
        height: Math.abs(position.y - interaction.start.y),
      };
    } else if (
      interaction.kind === "word"
      && Math.hypot(event.clientX - interaction.startX, event.clientY - interaction.startY) > 5
    ) {
      this.pointerInteraction = {
        kind: "pan",
        pointerId: interaction.pointerId,
        startX: interaction.startX,
        startY: interaction.startY,
        panX: this.panX,
        panY: this.panY,
      };
      this.canvas.classList.add("is-dragging");
    }
    this.draw();
  }

  _pointerFinish(event) {
    const interaction = this.pointerInteraction;
    if (!interaction || interaction.pointerId !== event.pointerId) return;
    const position = this._canvasPosition(event);
    this.pointerInteraction = null;

    if (interaction.kind === "word") this.callbacks.onWordClick?.(interaction.wordIndex);
    if (interaction.kind === "create" && this.draftBox) {
      const first = this._toData(this.draftBox.x, this.draftBox.y + this.draftBox.height);
      const second = this._toData(this.draftBox.x + this.draftBox.width, this.draftBox.y);
      if (first && second) {
        this.callbacks.onCreateBox?.({
          x: Math.min(first.x, second.x),
          y: Math.min(first.y, second.y),
          width: Math.abs(second.x - first.x),
          height: Math.abs(second.y - first.y),
        });
      }
      this.draftBox = null;
    }

    this.canvas.classList.remove("is-dragging");
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    this.draw();
  }
}
