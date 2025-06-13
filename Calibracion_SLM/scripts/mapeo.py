import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo (reemplaza con tus datos reales)
# Curva experimental: Intensidad (I) -> Fase (P)
I_exp = np.linspace(0, 255, 256)  # Intensidades enviadas
P_medida = np.sqrt(I_exp) * 0.1  # Ejemplo de curva no-lineal (simulada)


def lineal(x, a, b):
    y = a * x + b
    return y


P_lineal = lineal(I_exp, 1, 0)
# Ajustá estos parámetros para el comportamiento que querés
k = 0.02
P_sintetica = np.exp(k * I_exp)

plt.scatter(I_exp, P_sintetica, label="Exponencial suave")
plt.scatter(I_exp, P_lineal, color="r")
plt.xlabel("Intensidad (0-255)")
plt.ylabel("Fase")
# plt.title("Datos sintéticos con crecimiento exponencial")
plt.grid(False)
plt.legend()
plt.show()
