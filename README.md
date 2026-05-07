# TrichAI Backend

API REST para clasificación de imágenes de cannabis. Construida con FastAPI, la inferencia corre localmente en ONNX — sin llamadas a APIs externas de IA.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green.svg)](https://fastapi.tiangolo.com)

## Cómo funciona

Envías una imagen. La API la pasa por un modelo EfficientNetV2-S afinado (formato ONNX) y devuelve la clasificación junto con un análisis visual calculado directamente de los píxeles (cobertura de tricomas, textura, nivel de curación).

Cuatro categorías: `bud` · `hash` · `other` · `plant`

## Endpoints

```
POST /analyze          → clasifica una imagen
POST /contribute       → envía una imagen etiquetada
GET  /stats            → analytics (requiere header x-api-key)
GET  /contribute/stats → conteo de contribuciones
GET  /health           → estado del servicio
```

### Ejemplo

```bash
curl -X POST https://tu-api.railway.app/analyze -F "file=@foto.jpg"
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

## Ejecutar en local

```bash
pip install -r requirements.txt
# coloca tu modelo en model/phytolens_v1.onnx
uvicorn main:app --reload
```

Documentación interactiva en `http://localhost:8000/docs`

## Variables de entorno

| Variable | Descripción |
|---|---|
| `STATS_API_KEY` | Protege el acceso a `/stats` — configúrala siempre |
| `UPSTASH_REDIS_URL` | Analytics persistentes (`rediss://...`). Sin ella, usa memoria |
| `R2_ENDPOINT` / `R2_ACCESS_KEY` / `R2_SECRET_KEY` | Cloudflare R2 para almacenar contribuciones. Sin ellas, guarda en disco local |
| `R2_BUCKET` | Nombre del bucket (por defecto: `trichai-contributions`) |

## Seguridad

Las peticiones se validan en tres niveles: cabecera Content-Length (antes de leer el body), tipo MIME declarado y magic bytes reales del archivo. Límite de 20 peticiones por minuto por IP.
