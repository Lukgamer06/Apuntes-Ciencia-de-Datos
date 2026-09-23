# App de vision en vivo

Aplicacion web que accede a la webcam del PC con `getUserMedia`. No toma fotografias manuales ni guarda frames: envia JPEG temporales al backend cada 180 ms y dibuja las detecciones sobre el video en vivo.

## Ejecutar

Desde la carpeta `Huevos/app`:

```bash
python -m http.server 5173
```

Abre <http://localhost:5173>. El navegador pedira permiso para la webcam.

Si el backend corre en otra maquina, edita `API_URL` en `app.js` o abre la pagina con una variable global antes de cargar el script:

```html
<script>window.EGG_API_URL = "http://IP_DEL_PC:8000";</script>
```
