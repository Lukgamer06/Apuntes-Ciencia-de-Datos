# Backend de Huevos

Servicio FastAPI que recibe frames JPEG en vivo, ejecuta YOLO y devuelve las cajas detectadas.

## Instalar y ejecutar

```bash
cd Huevos/backend
python -m pip install -r requirements.txt
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

El backend usa `Huevos/models/best.pt` unicamente para localizar huevos y recortar cada uno. `best1.pt` determina si el huevo es sano o roto; cada inferencia válida reinicia el temporizador. Mientras el huevo siga detectado, si pasan 10 segundos sin una nueva inferencia válida de `best1.pt`, la app muestra **roto** hasta que llegue otra inferencia. La salida por timeout no lleva confianza. Si los pesos estan en otras rutas:

```bash
EGG_MODEL=/ruta/best.pt EGG_CONDITION_MODEL=/ruta/best1.pt uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

La clasificacion con `best1.pt` tiene un limite de 10 segundos por frame. El mismo valor configura el tiempo sin inferencias válidas antes del estado de fallback "roto"; se ajusta con `CONDITION_TIMEOUT_SECONDS` (10 por defecto). El backend interpreta la clase 0 como sano y la clase 1 como roto cuando el checkpoint tiene nombres genéricos.

Variables opcionales:

- `YOLO_CONFIDENCE=0.35`
- `EGG_CONDITION_MODEL=/ruta/best1.pt`
- `CONDITION_TIMEOUT_SECONDS=10`
- `CORS_ORIGINS=http://localhost:5173`
- `SERVO_HOME_ANGLE=90`
- `SERVO_HEALTHY_ANGLE=0`
- `SERVO_DAMAGED_ANGLE=180`

El ESP32 usa GPIO 19 y consulta `GET /servo` cada segundo. La respuesta es un JSON con el angulo actual, por ejemplo `{ "angle": 90 }` al iniciar.
Cuando el modelo detecta un huevo sano, el backend actualiza el angulo a `0`; cuando detecta uno roto, lo actualiza a `180`. El ESP32 lee el nuevo valor en su siguiente consulta, sin que AWS tenga que conocer la IP del ESP32.
`SERVO_HOME_ANGLE`, `SERVO_HEALTHY_ANGLE` y `SERVO_DAMAGED_ANGLE` solo admiten `90`, `0` y `180`, respectivamente.
