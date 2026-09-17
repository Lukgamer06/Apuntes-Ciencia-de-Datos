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

El backend consulta SIMIT después de reconocer una placa. En el servidor configura la clave sin ponerla en la app móvil ni en el repositorio:

```bash
export CROMA_API_KEY='tu_clave_nueva'
export DATABASE_URL='postgresql://usuario:clave@tu-rds.amazonaws.com:5432/placas?sslmode=require'
export MAX_INFERENCE_DIMENSION=1600
export CROMA_TIMEOUT_SECONDS=20
uvicorn backend.backend:app --host 0.0.0.0 --port 8000
```

La base de datos del historial es PostgreSQL. Crea una instancia RDS accesible
desde el servidor de la API, configura su grupo de seguridad para permitir el
puerto 5432 solo desde ese servidor y define `DATABASE_URL` con `sslmode=require`.
La tabla `plate_history` se crea automáticamente al iniciar el backend. El
endpoint `/predict` responde la placa reconocida inmediatamente; la consulta
SIMIT se ejecuta en segundo plano y queda guardada en esa tabla.
Se aceptan placas de carro (`ABC123`) y de moto (`ABC12D`). Comprueba en
`/health` que tanto `database_configured` como `database_connected` sean `true`.

Endpoints del historial:

```text
GET  /history
POST /history/refresh
DELETE /history/{placa}
```

Para `systemd`, crea un archivo que solo pueda leer root, por ejemplo `/etc/placas-backend.env`:

```bash
sudo install -m 600 /dev/null /etc/placas-backend.env
sudo nano /etc/placas-backend.env
```

Incluye `CROMA_API_KEY=tu_clave_nueva` sin comillas ni espacios alrededor del `=` y referencia el archivo desde la unidad:

```ini
[Service]
EnvironmentFile=/etc/placas-backend.env
```

Después ejecuta `sudo systemctl daemon-reload && sudo systemctl restart placas-backend`. Comprueba la configuración sin mostrar la clave con `curl http://127.0.0.1:8000/health`; debe aparecer `"croma_api_key_configured":true`. La imagen se reduce a un máximo de 1600 px y el OCR solo acepta candidatos con formato colombiano de seis caracteres (`ABC123` o `ABC12D`).

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
