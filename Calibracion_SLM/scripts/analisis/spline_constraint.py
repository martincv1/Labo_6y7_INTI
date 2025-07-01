import numpy as np
from scipy.interpolate import make_splrep, BSpline
from scipy.optimize import minimize

import matplotlib.pyplot as plt

# Parámetros
n_points = 256
amplitude_sin = 0.3
noise_std = 0.2
tolerance_knots = 0.1
k = 3   # Grado de la spline
s = 10  # s controla la suavidad de la spline
at_least_middle_knots = 2
plot_demo = False

assert n_points > 0
assert noise_std >= 0
assert k > 0
assert s > 0
assert at_least_middle_knots >= 0
assert 0 < tolerance_knots < 0.5

x = np.arange(n_points)
y_lin = np.linspace(0, 2 * np.pi, n_points)
y_true = amplitude_sin * np.sin(y_lin) + y_lin  # Curva creciente
noise = np.random.normal(0, noise_std, n_points)
y_noisy = y_true + noise

# Ajuste con spline suavizante
nest = 2 * k + 2 + at_least_middle_knots
spline = make_splrep(x, y_noisy, s=s, k=k, nest=nest)

# Acceso al grado, los knots y coeficientes
tck = spline.tck

n_knots = len(tck[0])
n_coeffs = len(tck[1])
print(f"Spline inicial. Grado: {k}. Número de knots: {n_knots}. Número de coeficientes: {n_coeffs}")

n_middle_knots = n_knots - (k + 1) * 2
if n_middle_knots < at_least_middle_knots:
    sep = (tck[0][-1] - tck[0][0]) / at_least_middle_knots
    middle_knots = tck[0][k] + sep / 2 + np.arange(at_least_middle_knots) * sep
    if n_middle_knots > 0:
        current_middle_knots = tck[0][k + 1:k + 1 + n_middle_knots]
        for cmk in current_middle_knots:
            idx_to_remove = np.argmin(np.abs(middle_knots - cmk))
            middle_knots = np.delete(middle_knots, idx_to_remove)
    for mk in middle_knots:
        spline = spline.insert_knot(mk)
    tck = spline.tck
    n_middle_knots = at_least_middle_knots

    n_knots = len(tck[0])
    n_coeffs = len(tck[1])
    print(f"Spline con knots adicionales. Grado: {k}. Número de knots: {n_knots}. Número de coeficientes: {n_coeffs}")

# Reconstruir la spline a partir de knots y coeffs
# Crear la nueva spline usando BSpline
spline_reconstruida = BSpline(*tck)

if plot_demo:
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


def get_spline(x, *args):
    t = args[0].copy()
    k = args[2]
    if args[1] > 0:
        t[k + 1:k + 1 + args[1]] = x[:args[1]]
    c = x[args[1]:]

    spline = BSpline(t, c, k)
    return spline


def fun_min(x, *args) -> float:
    spline = get_spline(x, *args)
    x_pos = args[3]
    values = args[4]
    return np.sum((spline(x_pos) - values) ** 2)


args = (tck[0], n_middle_knots, k, x, y_noisy)
if n_middle_knots > 0:
    middle_knots = tck[0][k + 1:k + 1 + n_middle_knots]
    x0 = np.concatenate((middle_knots, tck[1]))
    delta_knots = np.diff(tck[0][k:k + 2 + n_middle_knots])
    bounds = []
    for i in range(n_middle_knots):
        left = middle_knots[i] - tolerance_knots * delta_knots[i]
        right = middle_knots[i] + tolerance_knots * delta_knots[i + 1]
        bounds.append((left, right))
    for i in range(n_coeffs):
        bounds.append((None, None))
else:
    x0 = tck[1]
    bounds = None


def fun_constraint(x, *args) -> float:
    spline = get_spline(x, *args)
    derivative = spline.derivative()
    x_pos = args[3]
    return np.min(derivative(x_pos))


constraint = {"type": "ineq", "fun": fun_constraint, "args": args}

res = minimize(
    fun_min, x0, args=args, bounds=bounds, constraints=constraint, method="COBYLA", options={"maxiter": 1000}
)

spline_constrained = get_spline(res.x, *args)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig: plt.Figure = fig
axs: list[plt.Axes] = axs
axs[0].plot(x, y_true, label='Curva verdadera', linewidth=2)
axs[0].scatter(x, y_noisy, color='red', s=10, label='Datos ruidosos')
# Graficar la spline ajustada sin constraint
axs[0].plot(x, spline(x), label='Spline suavizante', color='green', linewidth=2)
# Graficar la spline con constraint para comparar
axs[0].plot(x, spline_constrained(x), '--', label='Spline constrained', color='orange', linewidth=2)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()
fig.suptitle('Resultado de spline suavizante con constraint')
axs[1].plot(x, spline_constrained.derivative()(x), label='Derivada de la spline', color='purple', linewidth=2)
plt.tight_layout()
plt.show()
