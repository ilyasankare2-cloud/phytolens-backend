import onnxruntime as ort
import numpy as np
from PIL import Image
import io, torch, torch.nn as nn
from torchvision import models

CLASS_NAMES = {
    0: {"label": "bud",   "display": "Cogollo seco"},
    1: {"label": "hash",  "display": "Hachís / Resina"},
    2: {"label": "other", "display": "Otro producto"},
    3: {"label": "plant", "display": "Planta viva"},
}

class InferenceEngine:
    def __init__(self, model_path: str):
        import torch
        from torchvision import models
        import torch.nn as nn

        self.device = "cpu"
        model = models.efficientnet_v2_s(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 256),
            nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 4),
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        self.model = model

    def predict(self, image_bytes: bytes) -> dict:
        import torch
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = tf(img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)[0]
            probs  = torch.softmax(logits, dim=0).numpy()

        pred_idx   = int(probs.argmax())
        confidence = float(probs[pred_idx])
        cfg        = CLASS_NAMES[pred_idx]

        return {
            "label":       cfg["label"],
            "display":     cfg["display"],
            "confidence":  round(confidence, 4),
            "quality":     "Alta" if confidence >= 0.85 else "Media" if confidence >= 0.65 else "Baja",
            "all_probs":   {CLASS_NAMES[i]["label"]: round(float(p), 4) for i, p in enumerate(probs)},
        }
    