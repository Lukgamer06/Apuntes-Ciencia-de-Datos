import math
from pathlib import Path

import joblib
import streamlit as st


def forward(x1, x2):
	"""Realiza la propagacion hacia adelante de la red neuronal."""
	x1_x2 = x1 * x2
	a1 = max(0, -0.064 - 0.57 * x1 - 0.44 * x2 + 0.90 * x1_x2)
	a2 = max(0, 0.43 - 0.18 * x1 + 0.63 * x2 - 0.21 * x1_x2)
	a3 = max(0, 0.41 + 0.59 * x1 + 0.70 * x2 + 0.80 * x1_x2)
	a4 = max(0, -0.13 + 0.91 * x1 + 0.26 * x2 + x1_x2)
	a5 = max(0, 0.15 + 0.078 * x1 + 0.88 * x2 - 0.12 * x1_x2)
	a6 = max(0, 0.31 + 0.080 * x1 + 0.11 * x2 - 0.85 * x1_x2)
	a7 = max(0, 0.47 + 0.40 * x1 - 0.13 * x2 + 0.41 * x1_x2)
	a8 = max(0, 0.29 - 0.82 * x1 + 0.47 * x2 - 0.59 * x1_x2)
	a9 = max(0, 0.57 - 0.56 * a1 - 0.54 * a2 - 0.34 * a3 - 0.38 * a4 + 0.27 * a5 - 0.18 * a6 - 0.37 * a7 - 0.44 * a8)
	a10 = max(0, 0.096 + 0.041 * a1 - 0.26 * a2 + 0.44 * a3 + 0.60 * a4 - 0.035 * a5 - 0.28 * a6 - 0.23 * a7 + 0.28 * a8)
	a11 = max(0, 0.57 + 0.65 * a1 - 0.0033 * a2 + 0.47 * a3 + 0.17 * a4 + 0.48 * a5 - 0.24 * a6 - 0.27 * a7 - 0.68 * a8)
	a12 = max(0, 0.11 + 0.33 * a1 - 0.078 * a2 - 0.53 * a3 + 1.3 * a4 - 0.58 * a5 + 0.69 * a6 - 0.0029 * a7 + 0.95 * a8)
	a13 = max(0, -0.062 - 0.67 * a1 + 0.47 * a2 + 0.75 * a3 - 0.015 * a4 + 0.27 * a5 + 0.37 * a6 + 0.29 * a7 + 0.43 * a8)
	a14 = max(0, 0.76 - 0.63 * a1 - 0.33 * a2 - 0.37 * a3 + 0.0099 * a4 - 0.48 * a5 - 0.92 * a6 + 0.69 * a7 + 0.074 * a8)
	a15 = max(0, 0.52 - 0.30 * a9 + 0.56 * a10 - 0.91 * a11 + 0.66 * a12 + 0.37 * a13 - 0.98 * a14)
	a16 = max(0, 0.28 - 0.21 * a9 - 0.46 * a10 - 0.36 * a11 - 0.39 * a12 - 0.21 * a13 + 0.98 * a14)
	a17 = max(0, -0.035 + 0.83 * a9 + 0.14 * a10 - 0.58 * a11 + 1.2 * a12 - 0.65 * a13 + 0.55 * a14)
	a18 = max(0, -0.085 + 0.038 * a9 - 0.066 * a10 - 0.42 * a11 + 0.95 * a12 - 0.42 * a13 + 0.44 * a14)
	a19 = max(0, 0.35 + 0.45 * a15 + 0.72 * a16 - 0.26 * a17 - 0.43 * a18)
	a20 = max(0, 0.21 - 1.5 * a15 - 0.91 * a16 + 1.4 * a17 + 0.77 * a18)
	a21 = max(0, 0.22 + 0.36 * a15 + 0.61 * a16 - 0.19 * a17 - 0.28 * a18)
	return math.tanh(-0.037 + 0.88 * a19 - 2.3 * a20 + 0.55 * a21)


st.set_page_config(page_title="Prediccion de problemas cardiacos", page_icon="❤")

IMG_TITULO = "https://th.bing.com/th/id/OIP.N_rTu2rKFurUctglIzkfyQHaHB?w=197&h=187&c=7&r=0&o=7&dpr=1.3&pid=1.7&rm=3"
IMG_SALUDABLE = "https://tse1.explicit.bing.net/th/id/OIP._S4IljqqVZvK7ipI_HUEVgHaE7?r=0&rs=1&pid=ImgDetMain&o=7&rm=3"
IMG_RIESGO = "https://st2.depositphotos.com/4297405/10474/v/950/depositphotos_104747216-stock-illustration-illustration-is-first-aid-person.jpg"

st.title("Prediccion de problemas cardiacos")
st.image(IMG_TITULO, width=260)
st.subheader("Objetivo de la aplicacion")
st.write("Estimar, de forma experimental, si una persona puede presentar problemas cardiacos a partir de su edad y nivel de colesterol, usando una red neuronal.")

st.subheader("Instrucciones de uso")
st.markdown("1. Seleccione la edad entre 20 y 100 años.\n2. Seleccione el colesterol entre 200 y 500 mg/dL.\n3. Pulse **Predecir** y revise el resultado.\n4. Esta herramienta no reemplaza la valoracion de un profesional de la salud.")

with st.form("datos_paciente"):
	edad = st.slider("Edad (años)", min_value=20, max_value=100, value=40)
	colesterol = st.slider("Colesterol (mg/dL)", min_value=200, max_value=500, value=250)
	predecir = st.form_submit_button("Predecir", type="primary")

if predecir:
	scaler_path = Path(__file__).with_name("modelo_estandarizacion.joblib")
	scaler = joblib.load(scaler_path)
	valores_normalizados = scaler.transform([[edad, colesterol]]) * 2
	salida = forward(valores_normalizados[0][0], valores_normalizados[0][1])
	porcentaje = max(0.0, min(100.0, (salida + 1) * 50))
	clase = 1 if salida >= 0 else -1

	st.subheader("Resultado de la prediccion")
	if clase == -1:
		st.image(IMG_SALUDABLE, width=420)
		st.success("no sufrira del corazon")
		st.write(f"Confianza estimada de esta prediccion: **{100 - porcentaje:.2f}%**")
		st.markdown("**Recomendaciones:** mantenga una alimentacion equilibrada, realice actividad fisica con regularidad, controle periodicamente su colesterol y presion arterial, y evite fumar.")
	else:
		st.image(IMG_RIESGO, width=520)
		st.error("Sufrira de problemas del corazon")
		st.write(f"Porcentaje de la prediccion: **{porcentaje:.2f}%**")
		recomendacion = "Reduzca grasas saturadas, aumente la actividad fisica y consulte a un profesional de la salud para un control cardiovascular."
		if edad >= 60:
			recomendacion += " Por su edad, priorice chequeos medicos periodicos."
		if colesterol >= 240:
			recomendacion += " Su colesterol es elevado: solicite orientacion profesional para reducirlo."
		st.markdown(f"**Recomendacion:** {recomendacion}")

st.divider()
st.markdown("**® Lucas Ardila**")
st.caption("Esto es un trabajo experimental, UNAB 2026")
