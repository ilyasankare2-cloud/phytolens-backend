from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import InferenceEngine

app = FastAPI(title="PhytoLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine("model/phytolens_v1.onnx")

@app.get("/health")
def health():
    return {"status": "ok", "model": "phytolens_v1"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg","image/png","image/webp"]:
        raise HTTPException(400, "Formato no soportado. Usa JPG o PNG.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen demasiado grande. Máximo 10MB.")
    result = engine.predict(contents)
    return {"success": True, "result": result}