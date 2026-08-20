# A. Guía Completa de Redes Neuronales: Del Perceptrón al Descenso del Gradiente

![](https://raw.githubusercontent.com/adiacla/Apuntes-Ciencia-de-Datos/refs/heads/main/imagenes/perceptro.jpeg)

Este repositorio contiene una explicación detallada, matemática y práctica sobre cómo aprenden las redes neuronales a través del ajuste de pesos ($w$), sesgos ($b$), el cálculo del error, la tasa de aprendizaje ($\eta$) y el descenso del gradiente. Además, incluye un análisis del notebook interactivo `Avanzada_Cuaderno_1_ANN_El_Perceptron.ipynb`.

---

## 1. La Anatomía Básica del Perceptrón

El **perceptrón** es la unidad de aprendizaje más simple en inteligencia artificial (propuesto por Frank Rosenblatt en 1958). Toma varias entradas, las multiplica por sus respectivos pesos, añade un ajuste general (sesgo) y aplica una función de activación para producir una salida binaria ($0$ o $1$).

### Componentes Principales:
* **Entradas ($x_1, x_2, \dots, x_n$):** Las características o datos de entrada.
* **Pesos ($w_1, w_2, \dots, w_n$):** La "fuerza" o importancia asignada a cada entrada.
* **Sesgo / Bias ($b$):** Un valor constante que permite desplazar la frontera de decisión, dándole flexibilidad a la neurona para activarse incluso si las entradas son cero.
* **Suma Ponderada ($z$):**
  $$z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + \dots + (x_n \cdot w_n) + b = \sum_{i=1}^{n} w_i x_i + b$$
* **Función de Activación Escalón:**
* 
![step](https://raw.githubusercontent.com/adiacla/Apuntes-Ciencia-de-Datos/refs/heads/main/imagenes/step.png)
---

## 2. Reglas de Ajuste y Métricas de Error

### Regla de Actualización del Perceptrón Clásico
Cuando el perceptrón realiza una predicción incorrecta, calculamos el error simple:
$$\text{Error} = y_{real} - y_{pred}$$

Usando la **Tasa de Aprendizaje ($\eta$)**, ajustamos los parámetros:
* **Nuevo peso:** $w_{nuevo} = w_{viejo} + \eta \cdot \text{Error} \cdot x$
* **Nuevo sesgo:** $b_{nuevo} = b_{viejo} + \eta \cdot \text{Error}$

> **¿Por qué interviene $x$ en el ajuste del peso?**
> Si $x_i = 0$, esa característica no influyó en el resultado final de la suma ponderada, por lo que su peso no debe modificarse.

---

### SSE vs. MSE: ¿El error va al cuadrado o no?

* **Suma de Errores al Cuadrado (SSE):**
  $$\text{SSE} = \sum_{i=1}^{N} (y_{real}^{(i)} - y_{pred}^{(i)})^2$$
  En el perceptrón clásico con función escalón, como $y_{real} - y_{pred} \in \{-1, 0, 1\}$, elevar al cuadrado convierte los $-1$ en $1$. Por lo tanto, la SSE representa el **conteo total de clasificaciones erróneas** en una época.

* **Error Cuadrático Medio (MSE):**
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{real}^{(i)} - y_{pred}^{(i)})^2$$
  Mide el promedio de los errores al cuadrado. Elevar al cuadrado evita que los errores positivos y negativos se cancelen, penaliza los errores grandes y ofrece una función de pérdida suave y derivable para el descenso del gradiente:
  $$L = \frac{1}{2}(y_{real} - y_{pred})^2 \implies \frac{\partial L}{\partial w} = -(y_{real} - y_{pred}) \cdot x$$

---

## 3. Dinámica de Entrenamiento: Épocas, Instancias y Estrategias

* **Instancia / Muestra:** Una sola fila del conjunto de datos.
* **Época (Epoch):** Una pasada completa sobre todo el conjunto de entrenamiento.

### Métodos de Actualización de Pesos:

| Estrategia | Descripción | Actualizaciones (Ejemplo: 10 datos, 5 épocas) |
| :--- | :--- | :--- |
| **Estocástico (SGD / Dato por Dato)** | Ajusta pesos y bias inmediatamente después de evaluar cada instancia. | **50 actualizaciones** |
| **Lote Completo (Batch GD)** | Procesa todos los datos, promedia los errores y actualiza una sola vez por época. | **5 actualizaciones** |
| **Mini-Lote (Mini-Batch GD)** | Divide los datos en bloques (ej. tamaño de lote = 2) y actualiza por cada bloque. | **25 actualizaciones** |

---

## 4. Ejemplo Numérico Paso a Paso

### Configuración Inicial:
* **Entradas:** $x_1 = 1$, $x_2 = 0$ | **Clase real:** $y_{real} = 1$
* **Pesos iniciales:** $w_1 = 0.3$, $w_2 = -0.2$ | **Sesgo inicial:** $b = -0.5$
* **Tasa de aprendizaje:** $\eta = 0.1$

### Paso 1: Predicción (Forward Pass)
$$z = (1 \cdot 0.3) + (0 \cdot -0.2) + (-0.5) = 0.3 + 0 - 0.5 = -0.2$$
$$y_{pred} = 0 \quad (\text{ya que } z = -0.2 < 0)$$

### Paso 2: Cálculo del Error
$$\text{Error} = y_{real} - y_{pred} = 1 - 0 = 1$$

### Paso 3: Actualización de Parámetros
* $w_1 = 0.3 + 0.1 \cdot (1) \cdot 1 = \mathbf{0.4}$
* $w_2 = -0.2 + 0.1 \cdot (1) \cdot 0 = \mathbf{-0.2}$ (sin cambio)
* $b = -0.5 + 0.1 \cdot (1) = \mathbf{-0.4}$

### Verificación:
$$z_{nuevo} = (1 \cdot 0.4) + (0 \cdot -0.2) + (-0.4) = 0.0 \implies y_{pred} = 1 \quad \text{(¡Correcto!)}$$

---

## 5. Código en Python Paso a Paso (Compuerta OR)

El siguiente script en Python puro demuestra cómo evoluciona el perceptrón época por época:

```python
# 1. Conjunto de datos de entrenamiento (Compuerta OR)
datos = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)
]

# 2. Inicialización de parámetros
w1, w2, b = 0.3, -0.2, -0.1
tasa_aprendizaje = 0.1
epocas = 4

def activacion(z):
    return 1 if z >= 0 else 0

print("--- ESTADO INICIAL ---")
print(f"w1 = {w1:.2f}, w2 = {w2:.2f}, bias = {b:.2f}\n" + "="*65)

# 3. Bucle de entrenamiento por épocas
for epoca in range(1, epocas + 1):
    print(f"ÉPOCA {epoca}")
    ajustes = 0
    
    for (x1, x2), y_real in datos:
        z = (x1 * w1) + (x2 * w2) + b
        y_pred = activacion(z)
        error = y_real - y_pred
        
        w1_old, w2_old, b_old = w1, w2, b
        
        if error != 0:
            w1 += tasa_aprendizaje * error * x1
            w2 += tasa_aprendizaje * error * x2
            b  += tasa_aprendizaje * error
            ajustes += 1
            
            print(f"  Entrada ({x1},{x2}) | Real: {y_real} | Pred: {y_pred} | Error: {error:+d}")
            print(f"    └─ Ajuste: w1: {w1_old:.2f}→{w1:.2f} | w2: {w2_old:.2f}→{w2:.2f} | b: {b_old:.2f}→{b:.2f}")
        else:
            print(f"  Entrada ({x1},{x2}) | Real: {y_real} | Pred: {y_pred} | Error:  0 (Correcto)")

    print(f"--> Fin de Época {epoca}: {ajustes} correcciones realizadas.\n")

print("="*65)
print(f"ENTRENAMIENTO FINALIZADO: w1 = {w1:.2f}, w2 = {w2:.2f}, bias = {b:.2f}")
```

# El Salto al Descenso del Gradiente (Gradient Descent)

El método del perceptrón simple solo funciona para problemas sencillos y funciones no continuas (de tipo sí/no). Para redes neuronales profundas con múltiples capas necesitaremos el **Descenso del Gradiente**.

---

## ¿B. Cómo funciona el Descenso del Gradiente?


### 1. La Superficie de Error (Función de Costo)

![gradiente](https://raw.githubusercontent.com/adiacla/Apuntes-Ciencia-de-Datos/refs/heads/main/imagenes/gradiente.jpg)

Imagina que el error de la red es una **montaña en 3D**:
* Las coordenadas $(x, y)$ en el piso son los pesos ($w$) y el sesgo ($b$).
* La altura de la montaña representa la **cantidad de error**.
* **Tu objetivo:** Llegar al valle más bajo (**Mínimo Global**).

---

### 2. El Gradiente ($\nabla L$)
Matemáticamente, la derivada (o gradiente) indica hacia dónde sube más rápido la pendiente.

---

### 3. El Descenso
Para reducir el error, caminas en la dirección **opuesta** al gradiente:

$$
w_{nuevo} = w_{viejo} - \eta \cdot \frac{\partial \text{Error}}{\partial w}
$$

$$
b_{nuevo} = b_{viejo} - \eta \cdot \frac{\partial \text{Error}}{\partial b}
$$

---

### 4. La Tasa de Aprendizaje ($\eta$)
La **tasa de aprendizaje** ($\eta$) equivale al **tamaño de tu paso** al bajar la montaña:

* **Si es muy grande:** Podrías dar un salto tan largo que te pases el valle y subas por la otra pared de la montaña.
* **Si es muy pequeña:** Te tomará miles de iteraciones llegar al fondo.

# Funciones de Activación en Redes Neuronales

Las **funciones de activación** son las encargadas de transformar la suma ponderada de las entradas ($z = w \cdot x + b$) en la salida final de una neurona. Determinan si la neurona debe "dispararse" (activarse) o no, y permiten que las redes neuronales aprendan patrones complejos.

---

## C. Función Escalón (Step Function)

![](https://raw.githubusercontent.com/adiacla/Apuntes-Ciencia-de-Datos/refs/heads/main/imagenes/funcion%20de%20activacion.jpeg)

### Fórmula

![step](https://raw.githubusercontent.com/adiacla/Apuntes-Ciencia-de-Datos/refs/heads/main/imagenes/step.png)


* **Objetivo:** Convertir la suma ponderada en una decisión binaria estricta ($0$ o $1$). Funciona como un interruptor de encendido/apagado.
* **Cuándo se usa:**
  * En el **Perceptrón clásico (Rosenblatt)** para resolver problemas sencillos y linealmente separables (ej. compuertas lógicas AND, OR).
* **Limitación clave:** Su derivada es $0$ en casi todas partes e indefinida en $z = 0$, lo que significa que **no permite el uso del Descenso del Gradiente**.

---

## 2. Función Sigmoide (Sigmoid)

### Fórmula
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

* **Objetivo:** Comprimir (*squash*) cualquier valor real en un rango acotado entre **0 y 1**. Esto permite interpretar la salida directamente como una **probabilidad**.
* **Cuándo se usa:**
  * En la **capa de salida** para problemas de **clasificación binaria** (ej. saber si un correo es spam o no, o si un paciente tiene una enfermedad).
* **Limitación clave:** Sufre del problema del *Gradiente Desvanecido* (Vanishing Gradient) en valores muy altos o muy bajos, por lo que hoy en día se evita usarla en las capas ocultas de redes profundas.

---

## 3. Función Lineal (Identity / Linear)

### Fórmula
$$
f(z) = z
$$

* **Objetivo:** Dejar pasar la suma ponderada directamente sin alterar su escala ni acotarla en un rango.
* **Cuándo se usa:**
  * En la **capa de salida** para problemas de **regresión** donde el valor a predecir es continuo (ej. estimar el precio de una vivienda, la temperatura del día o los ingresos futuros).
* **Limitación clave:** No añade no-linealidad. Usarla en las capas ocultas haría que toda la red neuronal funcione equivalentemente a una sola regresión lineal, sin importar cuántas capas tenga.

---

## 4. Función Tangente Hiperbólica (Tanh)

### Fórmula
$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

* **Objetivo:** Comprimir los valores continuos en un rango entre **-1 y 1**. A diferencia de la sigmoide, está **centrada en cero**, lo que ayuda a que las actualizaciones durante el entrenamiento sean más estables y rápidas.
* **Cuándo se usa:**
  * En **capas ocultas** de redes neuronales simples cuando se requiere una activación suave y centrada en cero.
  * Muy común en arquitecturas específicas como Redes Neuronales Recurrentes (RNN) y celdas LSTM.

---

## Nota Importante: Próximos Pasos

> Las funciones presentadas aquí son las bases históricas y fundamentales para entender las Redes Neuronales Artificiales (ANN). Sin embargo, en el aprendizaje profundo (*Deep Learning*) moderno, existen otras funciones de activación que superan varias de sus limitaciones, tales como:
>
> * **ReLU (Rectified Linear Unit):** La función por excelencia en capas ocultas de redes profundas.
> * **Leaky ReLU / ELU:** Variaciones diseñadas para evitar el "fallecimiento" de neuronas.
> * **Softmax:** Utilizada en la capa de salida para problemas de **clasificación multiclase**.
>
> **¡Todas estas funciones las veremos y analizaremos en detalle más adelante!**
