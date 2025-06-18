import numpy as np
from scipy.interpolate import make_splrep, BSpline

import matplotlib.pyplot as plt

# Parámetros
n_points = 256
x = np.linspace(0, 2 * np.pi, n_points)
y_true = 0.3 * np.sin(x) + x  # Curva creciente
noise = np.random.normal(0, 0.2, n_points)
y_noisy = y_true + noise

# Ajuste con spline suavizante
k = 3  # Grado de la spline
spline = make_splrep(x, y_noisy, s=10, k=k)  # s controla la suavidad

# Acceso al grado, los knots y coeficientes
tck = spline.tck
# Reconstruir la spline a partir de knots y coeffs
# Crear la nueva spline usando BSpline
spline_reconstruida = BSpline(*tck)

# Graficar
plt.figure(figsize=(8, 4))
plt.plot(x, y_true, label='Curva verdadera', linewidth=2)
plt.scatter(x, y_noisy, color='red', s=10, label='Datos ruidosos')
# Graficar la spline ajustada
plt.plot(x, spline(x), label='Spline suavizante', color='green', linewidth=2)
# Graficar la spline reconstruida para comparar
plt.plot(x, spline_reconstruida(x), '--', label='Spline reconstruida', color='orange', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Curva creciente y ruidosa de 0 a 2π')
plt.tight_layout()
plt.show()

derivative = spline_reconstruida.derivative()
# Graficar la derivada de la spline
plt.figure(figsize=(8, 4))
plt.plot(x, derivative(x), label='Derivada de la spline', color='purple', linewidth=2)
plt.xlabel('x')
plt.ylabel('Derivada')
plt.legend()
plt.title('Derivada de la spline suavizante')
plt.tight_layout()
plt.show()


