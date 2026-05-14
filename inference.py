import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

# Cap megapixels to mitigate decompression bombs (PNG/WebP can be tiny but
# expand to GBs in RAM). Pillow raises DecompressionBombError above this.
Image.MAX_IMAGE_PIXELS = 50_000_000

CLASS_INFO = {
    0: {
        "label":      "bud",
        "display":    "Cogollo seco",
        "thc_min":    15,
        "thc_max":    30,
        "description": "Flor seca, lista para consumir. La forma más común de cannabis.",
        "varieties":  ["OG Kush", "Amnesia Haze", "Purple Haze", "White Widow", "Gorilla Glue", "Gelato", "Fruta Prohibida", "Static"],
    },
    1: {
        "label":      "hash",
        "display":    "Hachís / Resina",
        "thc_min":    20,
        "thc_max":    60,
        "description": "Resina prensada. Más concentrado que la flor, suele moverse en rangos altos de THC.",
        "varieties":  ["Hash marroquí", "Charas", "Bubble hash", "Dry sift", "Hash libanés"],
    },
    2: {
        "label":      "other",
        "display":    "No detectado",
        "thc_min":    0,
        "thc_max":    0,
        "description": "No se ha detectado cannabis en la imagen.",
        "varieties":  [],
    },
    3: {
        "label":      "plant",
        "display":    "Planta viva",
        "thc_min":    10,
        "thc_max":    25,
        "description": "Planta viva. El THC se desarrolla durante la floración, el potencial real depende de la fase.",
        "varieties":  ["Cannabis sativa", "Cannabis indica", "Cannabis ruderalis", "Híbrido"],
    },
}


def ensure_model_local(local_path: str, r2_key: str, s3_client, bucket: str):
    """Download the model from R2 to local disk if not present.
    Raises RuntimeError if download fails — backend cannot start without a model."""
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"[model] Using cached {local_path} ({size_mb:.1f} MB)")
        return local_path

    if s3_client is None:
        raise RuntimeError(
            f"Model not found at {local_path} and R2 client unavailable. "
            "Set R2_ENDPOINT/R2_ACCESS_KEY/R2_SECRET_KEY env vars."
        )

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"[model] Downloading {r2_key} from R2 -> {local_path} ...")
    try:
        s3_client.download_file(bucket, r2_key, local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model {r2_key} from R2: {e}")
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"[model] Downloaded {size_mb:.1f} MB")
    return local_path


class InferenceEngine:
    def __init__(self, model_path: str):
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr  = (arr - mean) / std
        return arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def predict(self, image_bytes: bytes) -> dict:
        tensor = self._preprocess(image_bytes)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        e      = np.exp(logits - logits.max())
        probs  = e / e.sum()
        idx    = int(probs.argmax())
        conf   = float(probs[idx])
        info   = CLASS_INFO[idx]

        thc_estimate = round(info["thc_min"] + (info["thc_max"] - info["thc_min"]) * conf)

        return {
            "label":        info["label"],
            "display":      info["display"],
            "confidence":   round(conf, 4),
            "quality":      "Alta" if conf>=0.85 else "Media" if conf>=0.65 else "Baja",
            "thc_min":      info["thc_min"],
            "thc_max":      info["thc_max"],
            "thc_estimate": thc_estimate,
            "description":  info["description"],
            "varieties":    info["varieties"],
            "all_probs":    {CLASS_INFO[i]["label"]: round(float(p),4) for i,p in enumerate(probs)},
        }
