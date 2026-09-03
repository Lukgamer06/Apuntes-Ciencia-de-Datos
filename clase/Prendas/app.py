from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
	page_title="Predictor de prendas con Tensorflow",
	page_icon="👕",
	layout="centered",
)

MODEL_PATH = Path(__file__).parent / "fashion_mnist_model.keras"
CLASS_NAMES = [
	"T-shirt/top",
	"Trouser",
	"Pullover",
	"Dress",
	"Coat",
	"Sandal",
	"Shirt",
	"Sneaker",
	"Bag",
	"Ankle boot",
]


@st.cache_resource
def load_model():
	from tensorflow import keras

	return keras.models.load_model(MODEL_PATH, compile=False)


def prepare_image(image: Image.Image, invert: bool = False) -> np.ndarray:
	grayscale = ImageOps.grayscale(image)
	if invert:
		grayscale = ImageOps.invert(grayscale)
	resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)
	return np.asarray(resized, dtype=np.float32) / 255.0


def show_prediction(image_array: np.ndarray) -> None:
	model = load_model()
	tensor = image_array[np.newaxis, ..., np.newaxis]
	probabilities = np.asarray(model.predict(tensor, verbose=0))[0]
	predicted_index = int(np.argmax(probabilities))

	st.subheader(f"Predicción: {CLASS_NAMES[predicted_index]}")
	st.metric("Confianza", f"{probabilities[predicted_index] * 100:.2f}%")

	results = sorted(
		zip(CLASS_NAMES, probabilities), key=lambda item: item[1], reverse=True
	)
	st.write("Probabilidades por clase")
	for class_name, probability in results:
		st.progress(float(probability), text=f"{class_name}: {probability * 100:.2f}%")


st.title("Predictor de prendas con Tensorflow")
st.caption("Clasificación de imágenes con un modelo entrenado sobre Fashion-MNIST")

input_mode = st.radio("Fuente de la imagen", ["Dibujar en canvas", "Subir imagen"], horizontal=True)
invert_colors = False
image_array = None

if input_mode == "Dibujar en canvas":
	canvas_result = st_canvas(
		fill_color="rgba(0, 0, 0, 1)",
		stroke_width=20,
		stroke_color="#FFFFFF",
		background_color="#000000",
		height=280,
		width=280,
		drawing_mode="freedraw",
		key="clothing_canvas",
	)
	if canvas_result.image_data is not None:
		canvas_image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
		image_array = prepare_image(canvas_image)
else:
	uploaded_file = st.file_uploader(
		"Selecciona una imagen",
		type=["png", "jpg", "jpeg", "bmp", "webp"],
	)
	invert_colors = st.checkbox(
		"Invertir colores (si la imagen tiene fondo blanco)", value=False
	)
	if uploaded_file is not None:
		image_array = prepare_image(Image.open(uploaded_file), invert_colors)

if image_array is not None:
	st.image(image_array, caption="Imagen procesada a 28 x 28 píxeles", width=224)
	if st.button("Predecir prenda", type="primary", use_container_width=True):
		if not MODEL_PATH.exists():
			st.error(f"No se encontró el modelo: {MODEL_PATH.name}")
		else:
			with st.spinner("Analizando la imagen..."):
				show_prediction(image_array)

st.divider()
st.subheader("Instrucciones")
st.markdown(
	"""
	- Dibuja una sola prenda centrada en el canvas negro usando un lápiz de ancho medio, o carga una imagen.
	- La imagen se convierte automáticamente a escala de grises, se ajusta a 28 x 28 píxeles y se normaliza dividiendo entre 255.
	- El modelo espera un fondo negro y la prenda en tonos de gris hacia el blanco.
	- Para obtener mejores resultados, usa imágenes similares a las imágenes empleadas durante el entrenamiento: una prenda, centrada y con poco ruido.
	- Si la imagen cargada tiene fondo blanco, activa la opción de invertir colores antes de predecir.
	"""
)
