import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Cargar datos, eliminar nulos y filtrar por rangos válidos
# ---------------------------------------------------------
df = pd.read_csv("pacientes.csv")

# Eliminar nulos
df = df.dropna()

# Filtrar por rangos de edad (0 a 120) y colesterol (100 a 600)
df = df[(df["edad"] >= 0) & (df["edad"] <= 120)]
df = df[(df["colesterol"] >= 100) & (df["colesterol"] <= 600)]
df = df.reset_index(drop=True)

# ---------------------------------------------------------
# 2. Modificar la columna objetivo (label)
# ---------------------------------------------------------
df["problema_cardiaco"] = df["problema_cardiaco"].replace(0, -1)

# ---------------------------------------------------------
# 3. Features (x, y) y Estandarización Z-score (* 2)
# ---------------------------------------------------------
features = ["edad", "colesterol"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features]) * 2

# Guardar características estandarizadas en el DataFrame
df["x1"] = X_scaled[:, 0]
df["x2"] = X_scaled[:, 1]

# ---------------------------------------------------------
# 4. Guardar modelo de estandarización
# ---------------------------------------------------------
joblib.dump(scaler, "modelo_estandarizacion.joblib")

# ---------------------------------------------------------
# 5. Generación del Archivo JSON
# ---------------------------------------------------------
json_data = [
    {
        "x": float(round(row["x1"], 4)),
        "y": float(round(row["x2"], 4)),
        "label": int(row["problema_cardiaco"]),
    }
    for _, row in df.iterrows()
]

with open("datos_procesados.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2)

# ---------------------------------------------------------
# 6. Generar Gráfico de Dispersión (Scatter Plot)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

# Mapa de colores según la etiqueta: -1 (rojo), 1 (azul)
colores = {-1: "red", 1: "blue"}

for label, color in colores.items():
    subset = df[df["problema_cardiaco"] == label]
    plt.scatter(
        subset["x1"],
        subset["x2"],
        c=color,
        label=f"Label: {label}",
        s=90,
        alpha=0.8,
        edgecolors="k",
    )

plt.title("Gráfico de Dispersión $x_1$ vs $x_2$ por Etiqueta", fontsize=14)
plt.xlabel("$x_1$ (Edad estandarizada $\\times 2$)", fontsize=12)
plt.ylabel("$x_2$ (Colesterol estandarizado $\\times 2$)", fontsize=12)
plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(title="Clase", loc="best")
plt.tight_layout()

# Guardar la imagen en disco y mostrarla
plt.savefig("grafico_dispersion.png", dpi=300)
plt.show()