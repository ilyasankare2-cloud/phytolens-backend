# TrichAI Backend

REST API for cannabis image classification. Built with FastAPI, runs inference locally using ONNX — no external AI calls.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green.svg)](https://fastapi.tiangolo.com)

## How it works

You send an image. The API runs it through a fine-tuned EfficientNetV2-S model (ONNX format) and returns the classification + a visual analysis computed directly from the pixels (trichome coverage, texture, cure level).

Four categories: `bud` · `hash` · `other` · `plant`

## Endpoints

```
POST /analyze          → classify an image
POST /contribute       → submit a labeled image
GET  /stats            → analytics (requires x-api-key header)
GET  /contribute/stats → contribution counts
GET  /health           → service status
```

### Example

```bash
curl -X POST https://your-api.railway.app/analyze -F "file=@photo.jpg"
```

```json
{
  "success": true,
  "result": {
    "label": "bud",
    "confidence": 0.924,
    "quality": "Alta",
    "thc_estimate": 24,
    "visual_traits": {
      "trichomes": "Alta",
      "trichome_coverage": 14.2,
      "texture": "Cristalina",
      "cure": "Bien curada",
      "dominant_color": [142, 98, 61]
    }
  }
}
```

## Running locally

```bash
pip install -r requirements.txt
# put your model at model/phytolens_v1.onnx
uvicorn main:app --reload
```

Interactive docs at `http://localhost:8000/docs`

## Environment variables

| Variable | Description |
|---|---|
| `STATS_API_KEY` | Restricts access to `/stats` — set this |
| `UPSTASH_REDIS_URL` | Persistent analytics (`rediss://...`). In-memory fallback if not set |
| `R2_ENDPOINT` / `R2_ACCESS_KEY` / `R2_SECRET_KEY` | Cloudflare R2 for contribution storage. Local filesystem fallback if not set |
| `R2_BUCKET` | Bucket name (default: `trichai-contributions`) |

## Security

Requests are validated at three levels: Content-Length header (before the body is read), declared MIME type, and actual magic bytes. Rate limited to 20 requests/minute per IP.
