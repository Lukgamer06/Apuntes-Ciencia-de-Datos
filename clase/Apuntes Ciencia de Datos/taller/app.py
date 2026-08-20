
import math

import joblib
import streamlit as st


def forward(X1, X2):
    """Compute a forward pass of the network."""
    X1X2 = X1 * X2
    a1 = max(0, -0.064 + (-0.57 * X1) + (-0.44 * X2) + (0.90 * X1X2))
    a2 = max(0, 0.43 + (-0.18 * X1) + (0.63 * X2) + (-0.21 * X1X2))
    a3 = max(0, 0.41 + (0.59 * X1) + (0.70 * X2) + (0.80 * X1X2))
    a4 = max(0, -0.13 + (0.91 * X1) + (0.26 * X2) + (1.0 * X1X2))
    a5 = max(0, 0.15 + (0.078 * X1) + (0.88 * X2) + (-0.12 * X1X2))
    a6 = max(0, 0.31 + (0.080 * X1) + (0.11 * X2) + (-0.85 * X1X2))
    a7 = max(0, 0.47 + (0.40 * X1) + (-0.13 * X2) + (0.41 * X1X2))
    a8 = max(0, 0.29 + (-0.82 * X1) + (0.47 * X2) + (-0.59 * X1X2))
    a9 = max(0, 0.57 + (-0.56 * a1) + (-0.54 * a2) + (-0.34 * a3) + (-0.38 * a4) + (0.27 * a5) + (-0.18 * a6) + (-0.37 * a7) + (-0.44 * a8))
    a10 = max(0, 0.096 + (0.041 * a1) + (-0.26 * a2) + (0.44 * a3) + (0.60 * a4) + (-0.035 * a5) + (-0.28 * a6) + (-0.23 * a7) + (0.28 * a8))
    a11 = max(0, 0.57 + (0.65 * a1) + (-0.0033 * a2) + (0.47 * a3) + (0.17 * a4) + (0.48 * a5) + (-0.24 * a6) + (-0.27 * a7) + (-0.68 * a8))
    a12 = max(0, 0.11 + (0.33 * a1) + (-0.078 * a2) + (-0.53 * a3) + (1.3 * a4) + (-0.58 * a5) + (0.69 * a6) + (-0.0029 * a7) + (0.95 * a8))
    a13 = max(0, -0.062 + (-0.67 * a1) + (0.47 * a2) + (0.75 * a3) + (-0.015 * a4) + (0.27 * a5) + (0.37 * a6) + (0.29 * a7) + (0.43 * a8))
    a14 = max(0, 0.76 + (-0.63 * a1) + (-0.33 * a2) + (-0.37 * a3) + (0.0099 * a4) + (-0.48 * a5) + (-0.92 * a6) + (0.69 * a7) + (0.074 * a8))
    a15 = max(0, 0.52 + (-0.30 * a9) + (0.56 * a10) + (-0.91 * a11) + (0.66 * a12) + (0.37 * a13) + (-0.98 * a14))
    a16 = max(0, 0.28 + (-0.21 * a9) + (-0.46 * a10) + (-0.36 * a11) + (-0.39 * a12) + (-0.21 * a13) + (0.98 * a14))
    a17 = max(0, -0.035 + (0.83 * a9) + (0.14 * a10) + (-0.58 * a11) + (1.2 * a12) + (-0.65 * a13) + (0.55 * a14))
    a18 = max(0, -0.085 + (0.038 * a9) + (-0.066 * a10) + (-0.42 * a11) + (0.95 * a12) + (-0.42 * a13) + (0.44 * a14))
    a19 = max(0, 0.35 + (0.45 * a15) + (0.72 * a16) + (-0.26 * a17) + (-0.43 * a18))
    a20 = max(0, 0.21 + (-1.5 * a15) + (-0.91 * a16) + (1.4 * a17) + (0.77 * a18))
    a21 = max(0, 0.22 + (0.36 * a15) + (0.61 * a16) + (-0.19 * a17) + (-0.28 * a18))
    a22 = math.tanh(-0.037 + (0.88 * a19) + (-2.3 * a20) + (0.55 * a21))
    return a22


# ---------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------
st.set_page_config(page_title="Predicción de Riesgo Cardíaco", page_icon="❤️", layout="centered")

IMG_GENERAL = "https://clinicarisso.com/wp-content/uploads/2025/08/colesterol.jpg"
IMG_SIN_RIESGO = "https://farmaciajonuriarte.com/wp-content/uploads/2021/02/hipercolesterolemia.jpg"
IMG_CON_RIESGO = "https://blobcore.pulsoslp.com.mx/images/2022/01/17/colesterol-alto-focus-0-0-1044-675.jpg"

st.title("❤️ Predicción de Riesgo de Problemas Cardíacos")
st.image(IMG_GENERAL, use_container_width=True)

st.header("🎯 Objetivo")
st.write(
    "Esta aplicación utiliza una red neuronal entrenada para estimar la probabilidad de que una "
    "persona sufra un problema cardíaco a partir de su edad y su nivel de colesterol."
)

st.header("📋 Instrucciones")
st.write(
    """
    1. Ingrese la **edad** y el **nivel de colesterol** usando los deslizadores de la izquierda.
    2. El sistema estandarizará automáticamente los valores con el mismo proceso usado en el entrenamiento
       (eliminación de nulos, filtrado de rangos válidos, estandarización Z-score y multiplicación por 2).
    3. Presione **"Predecir"** para obtener el resultado y una recomendación preventiva.
    """
)

# Escalador entrenado en procesamiento.py (mismo preprocesamiento del entrenamiento)
scaler = joblib.load("modelo_estandarizacion.joblib")

st.sidebar.header("🩺 Datos del paciente")
edad = st.sidebar.slider("Edad (años)", min_value=0, max_value=120, value=45, step=1)
colesterol = st.sidebar.slider("Colesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)

predecir = st.sidebar.button("Predecir", type="primary")

if predecir:
    # Mismo preprocesamiento que en el entrenamiento: estandarizar y multiplicar por 2
    X_scaled = scaler.transform([[edad, colesterol]]) * 2
    x1, x2 = X_scaled[0][0], X_scaled[0][1]

    salida = forward(x1, x2)
    clase = 1 if salida >= 0 else -1
    prob_riesgo = (salida + 1) / 2 * 100
    prob_sin_riesgo = 100 - prob_riesgo

    st.header("🔎 Resultado de la Predicción")

    if clase == 1:
        st.image(IMG_CON_RIESGO, use_container_width=True)
        st.error(f"⚠️ Riesgo de sufrir un problema cardíaco (Clase: {clase})")
        st.error(f"Probabilidad de pertenecer a la clase de riesgo: **{prob_riesgo:.2f}%**")

        st.subheader("💡 Recomendaciones preventivas")
        st.markdown(
            """
            - Reducir el consumo de grasas saturadas y colesterol dietético.
            - Realizar actividad física regular (mínimo 150 min/semana).
            - Controlar el peso corporal y evitar el sobrepeso.
            - Evitar el consumo de tabaco y limitar el alcohol.
            - Consultar a un cardiólogo para un chequeo y posible tratamiento con estatinas.
            - Mantener una dieta rica en fibra, frutas, verduras y pescado.
            """
        )
    else:
        st.image(IMG_SIN_RIESGO, use_container_width=True)
        st.success(f"✅ Sin riesgo de sufrir un problema cardíaco (Clase: {clase})")
        st.success(f"Probabilidad de no pertenecer a la clase de riesgo: **{prob_sin_riesgo:.2f}%**")

        st.subheader("💡 Recomendaciones para mantener la salud cardíaca")
        st.markdown(
            """
            - Mantener una alimentación balanceada baja en grasas saturadas.
            - Continuar con actividad física regular.
            - Realizar chequeos médicos periódicos de colesterol y presión arterial.
            - Evitar el sedentarismo y el consumo de tabaco.
            """
        )

st.divider()
st.caption("Ejemplo usando modelos playground.scienxlab.org con UNAB 2026")
st.caption("Realizado por Alfredo Diaz")