import html
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "models" / "Data"
IMAGES_DIR = DATA_ROOT / "images"
LABELS_DIR = DATA_ROOT / "labels"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ClassLabeler:
    def __init__(self):
        self.items = self.find_pairs()
        self.position = 0
        self.lock = threading.Lock()
        self.history = []
        self.saved_count = 0

    def find_pairs(self):
        if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
            return []
        return [
            (image, LABELS_DIR / f"{image.stem}.txt")
            for image in sorted(IMAGES_DIR.iterdir())
            if image.suffix.lower() in IMAGE_EXTENSIONS
            and (LABELS_DIR / f"{image.stem}.txt").exists()
        ]

    def current_item(self):
        with self.lock:
            return self.items[self.position] if self.position < len(self.items) else None

    def assign_class(self, class_id):
        with self.lock:
            if self.position >= len(self.items):
                return
            _, label_path = self.items[self.position]
            original = label_path.read_text(encoding="utf-8")
            updated = []
            for line in original.splitlines():
                values = line.split()
                if values:
                    values[0] = str(class_id)
                    updated.append(" ".join(values))
            label_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
            if label_path.read_text(encoding="utf-8") != "\n".join(updated) + "\n":
                raise IOError(f"No se pudo verificar el guardado de {label_path}")
            self.history.append((self.position, label_path, original))
            self.saved_count += 1
            self.position += 1

    def undo(self):
        with self.lock:
            if not self.history:
                return False
            previous_position, label_path, original = self.history.pop()
            label_path.write_text(original, encoding="utf-8")
            self.position = previous_position
            return True

    def page(self):
        item = self.current_item()
        if item is None:
            return """<!doctype html><html lang="es"><meta charset="utf-8">
<title>Proceso terminado</title><body><h1>Proceso terminado</h1>
<p>Todas las imágenes fueron etiquetadas.</p>
<button onclick="undo()">Volver a la anterior</button>
<script>async function undo() { await fetch('/undo', {method:'POST'}); location.reload(); }</script>
</body></html>"""

        image_path, _ = item
        filename = html.escape(image_path.name)
        progress = f"{self.position + 1} / {len(self.items)}"
        return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Ajustar clases</title>
<style>
body {{ margin:0; background:#202124; color:white; font-family:Arial; text-align:center; }}
main {{ max-width:1100px; margin:20px auto; padding:0 16px; }}
canvas {{ max-width:100%; max-height:72vh; background:#111; }}
.progress {{ margin:12px; font-size:18px; }} .buttons {{ display:flex; gap:16px; margin-top:16px; }}
button {{ flex:1; min-height:70px; border:0; border-radius:8px; font-size:24px; font-weight:bold; cursor:pointer; }}
.healthy {{ background:#8fd694; }} .damaged {{ background:#e58b8b; }}
.undo {{ width:100%; margin-top:16px; min-height:48px; background:#d8d8d8; color:#222; }}
</style></head><body><main>
<div class="progress">Imagen {progress}<br><small>{filename}</small></div>
<canvas id="preview" aria-label="{filename}"></canvas><div class="buttons">
<button class="healthy" onclick="assign(0)">Sano (S)<br><small>Clase 0</small></button>
<button class="damaged" onclick="assign(1)">Dañado (D)<br><small>Clase 1</small></button>
</div><button class="undo" onclick="undo()">Volver a la imagen anterior</button></main><script>
async function assign(id) {{ await fetch('/class/' + id, {{method:'POST'}}); location.reload(); }}
async function undo() {{ await fetch('/undo', {{method:'POST'}}); location.reload(); }}
document.addEventListener('keydown', e => {{ if (e.key.toLowerCase()==='s') assign(0); if (e.key.toLowerCase()==='d') assign(1); }});
        const canvas = document.getElementById('preview');
        const context = canvas.getContext('2d');
        const image = new Image();
        image.onload = async () => {{
            const labels = await fetch('/label').then(response => response.text());
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;
            context.drawImage(image, 0, 0);
            labels.split('\\n').forEach(line => {{
                const values = line.trim().split(/\\s+/).map(Number);
                if (values.length !== 5 || values.some(Number.isNaN)) return;
                const [classId, centerX, centerY, width, height] = values;
                const boxWidth = width * canvas.width;
                const boxHeight = height * canvas.height;
                const x = (centerX * canvas.width) - boxWidth / 2;
                const y = (centerY * canvas.height) - boxHeight / 2;
                context.strokeStyle = classId === 1 ? '#ff3b30' : '#34c759';
                context.lineWidth = Math.max(3, canvas.width / 400);
                context.strokeRect(x, y, boxWidth, boxHeight);
                context.fillStyle = context.strokeStyle;
                context.font = `bold ${{Math.max(16, canvas.width / 50)}}px Arial`;
                context.fillText(classId === 1 ? 'Dañado' : 'Sano', x + 4, Math.max(18, y - 6));
            }});
        }};
        image.src = '/image';
</script></body></html>"""

        


class RequestHandler(BaseHTTPRequestHandler):
    labeler = None

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            content = self.labeler.page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif path == "/image":
            item = self.labeler.current_item()
            if item is None:
                self.send_error(404)
                return
            image_path, _ = item
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(image_path.name)[0] or "image/jpeg")
            self.end_headers()
            self.wfile.write(image_path.read_bytes())
        elif path == "/label":
            item = self.labeler.current_item()
            if item is None:
                self.send_error(404)
                return
            _, label_path = item
            content = label_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/undo":
            self.labeler.undo()
            self.send_response(204)
            self.end_headers()
            return
        if path not in ("/class/0", "/class/1"):
            self.send_error(404)
            return
        self.labeler.assign_class(int(path[-1]))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args):
        pass


def main():
    labeler = ClassLabeler()
    if not labeler.items:
        raise RuntimeError(f"No hay pares imagen/etiqueta en {IMAGES_DIR}")
    RequestHandler.labeler = labeler
    server = ThreadingHTTPServer(("127.0.0.1", 8765), RequestHandler)
    url = "http://127.0.0.1:8765"
    print(f"Abriendo {url}. S=Sano, D=Dañado. Ctrl+C para salir.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nProceso pausado. Guardados verificados: {labeler.saved_count}/{len(labeler.items)}.")
        print(f"Las etiquetas están en: {LABELS_DIR}")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
