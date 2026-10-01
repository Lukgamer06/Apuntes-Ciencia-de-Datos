# Proyecto Huevos

Este proyecto combina visión artificial y aprendizaje profundo para detectar huevos, identificar si están sanos o rotos y controlar una respuesta automatizada mediante un backend y una interfaz de hardware o aplicación.

## Estructura

- `app/`: interfaz web o cliente de la solución.
- `backend/`: servicio FastAPI que recibe imágenes y ejecuta inferencias con YOLO.
- `models/`: pesos del modelo entrenado (`best.pt`, `best1.pt` y otros artefactos).
- `runs_colab/`: resultados de experimentos y pruebas realizadas en Colab.
- `top_models/`: mejores modelos obtenidos durante la búsqueda de hiperparámetros.

## Objetivo

El sistema busca reconocer huevos en imágenes, separar los sanos de los dañados y, en caso de detectar un daño, activar una lógica de control o mostrar un estado de alerta. La solución usa modelos YOLO para detección y clasificación de objetos.

## Flujo general

1. El backend recibe una imagen o frame desde la aplicación.
2. Un modelo YOLO detecta los huevos presentes.
3. Un segundo modelo clasifica si el huevo está sano o roto.
4. La lógica de negocio determina el estado final y puede enviar una señal a hardware externo o devolver el resultado al cliente.

## Componentes principales

### Backend

El backend está implementado en FastAPI y se ejecuta con Python. Recibe peticiones, procesa imágenes y devuelve resultados de inferencia.

### Modelos

Los modelos entrenados se guardan en la carpeta `models/` y se usan para:

- localización de huevos
- clasificación de condición (sano/roto)
- control de fallbacks por timeout o ausencia de inferencias válidas

### Aplicación

La carpeta `app/` contiene la parte visual o cliente sobre la que se integra la detección y la respuesta del sistema.

## Tecnologías

- Python
- FastAPI
- YOLO
- OpenCV
- PyTorch
- JavaScript / web app (según la implementación del cliente)
- ESP32 / control de actuadores (según el caso de uso)

## Uso recomendado

```bash
cd Huevos/backend
python -m pip install -r requirements.txt
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

## Nota

Este proyecto está orientado a una solución práctica de visión artificial aplicada a una inspección automatizada de huevos, con énfasis en detección, clasificación y control de estado en tiempo real.
