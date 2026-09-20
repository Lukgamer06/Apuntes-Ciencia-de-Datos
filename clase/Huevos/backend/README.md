# Backend de Huevos

Servicio FastAPI que recibe frames JPEG en vivo, ejecuta YOLO y devuelve las cajas detectadas.

## Instalar y ejecutar

```bash
cd Huevos/backend
python -m pip install -r requirements.txt
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

El backend espera el modelo entrenado en `Huevos/models/best.pt`. Si el peso tiene otro nombre o esta en otra ruta:

```bash
EGG_MODEL=/ruta/al/modelo.pt uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

Variables opcionales:

- `YOLO_CONFIDENCE=0.35`
- `CORS_ORIGINS=http://localhost:5173`
- `ESP32_URL=http://192.168.1.50`

El ESP32 debe aceptar `POST /motor` con `{ "enabled": true }` o `{ "enabled": false }`. Si no se configura `ESP32_URL`, el endpoint funciona en modo simulacion.
