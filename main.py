import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import InferenceEngine

app = FastAPI(title="TrichAI API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine("model/phytolens_v1.onnx")

CONTRIB_DIR = "contributions"
VALID_LABELS = {"bud", "hash", "other", "plant"}

os.makedirs(CONTRIB_DIR, exist_ok=True)
for label in VALID_LABELS:
    os.makedirs(os.path.join(CONTRIB_DIR, label), exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok", "model": "trichai_v1"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Formato no soportado. Usa JPG o PNG.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen demasiado grande. Máximo 10MB.")
    result = engine.predict(contents)
    return {"success": True, "result": result}

@app.post("/contribute")
async def contribute(file: UploadFile = File(...), label: str = Form(...)):
    if label not in VALID_LABELS:
        raise HTTPException(400, f"Etiqueta inválida. Usa: {', '.join(VALID_LABELS)}")
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Formato no soportado. Usa JPG o PNG.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen demasiado grande. Máximo 10MB.")

    ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(CONTRIB_DIR, label, filename)

    with open(path, "wb") as f:
        f.write(contents)

    return {"success": True, "saved": filename, "label": label}

@app.get("/contribute/stats")
def contribute_stats():
    stats = {}
    for label in VALID_LABELS:
        folder = os.path.join(CONTRIB_DIR, label)
        stats[label] = len(os.listdir(folder)) if os.path.exists(folder) else 0
    stats["total"] = sum(stats.values())
    return stats
