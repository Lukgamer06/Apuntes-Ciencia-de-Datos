# Prendas

Este proyecto está enfocado en la clasificación de prendas de ropa usando aprendizaje profundo.

## ¿Qué contiene?

- `app.py`: archivo principal con la lógica de la aplicación o inferencia.
- `fashion_mnist_model.keras`: modelo entrenado en Keras para clasificar prendas.
- cuadernos de exploración y entrenamiento.
- `requirements.txt`: dependencias del proyecto.

## Objetivo

El propósito principal es reconocer distintas categorías de prendas (como camisetas, pantalones, zapatos, etc.) a partir de imágenes, utilizando un modelo basado en redes neuronales y el conjunto Fashion MNIST.

## Contexto del dataset

Fashion MNIST es un conjunto de imágenes en escala de grises que representa diferentes tipos de ropa. El modelo entrenado intenta aprender patrones visuales que permitan distinguir estas clases con buena precisión.

## Tecnologías

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## Flujo de uso

1. Instalar las dependencias listadas en `requirements.txt`.
2. Cargar el modelo `fashion_mnist_model.keras`.
3. Ejecutar la aplicación o la inferencia sobre nuevas imágenes.
4. Revisar la salida de la clasificación para identificar la prenda.

## Uso sugerido

```bash
cd Prendas
python -m pip install -r requirements.txt
python app.py
```

## Resultado esperado

El sistema debe ser capaz de identificar la categoría de una prenda a partir de una imagen de entrada, mostrando un ejemplo práctico de clasificación supervisada con redes neuronales.
