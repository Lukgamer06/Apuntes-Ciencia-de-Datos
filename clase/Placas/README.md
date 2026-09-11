# Detector y lector de placas

## Backend

Requiere Python 3.13. Desde `Placas`:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/WSL: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn backend.backend:app --host 0.0.0.0 --port 8000
```

El modelo activo es `models/best.pt`; también puede cambiarse con `PLATE_MODEL`. El endpoint es `POST /predict` con un campo multipart llamado `file`. Para que un celular en la misma red llegue al PC, usa la IP LAN del PC en `app/.env` y permite el puerto 8000 en el firewall.

## App movil

Instala Node.js y ejecuta desde `Placas/app`:

```bash
npm install
# copia .env.example a .env y cambia la IP del backend
npx expo start
```

Abre el QR con Expo Go o usa `npx expo run:android` para un APK local. La app solicita permiso de cámara, captura JPEG y lo envía al backend.

## Entrenamiento y Google Drive

Construye el dataset normalizado y ejecuta la búsqueda aleatoria:

```bash
python models/random_search.py --trials 20 --epochs 100 --device 0
```

Se guardan `models/top_models/model_1..3/best.pt`, cada `configuration.txt` y `search_results.json`. Cada `.pt` conserva pesos y bias aprendidos; `model_1/best.pt` se copia como `models/best.pt` para la API. Las augmentations, `weight_decay`, `AdamW`, `patience`, validación separada y early stopping reducen el sobreajuste.

Para copiar los artefactos a Drive instala y configura `rclone` una sola vez con `rclone config`, y luego:

```bash
python models/upload_to_drive.py --remote gdrive
```
