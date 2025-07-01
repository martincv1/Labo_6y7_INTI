import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_splrep, BSpline
from scipy.optimize import minimize
from typing import Union


class GammaCurver:

    def __init__(
        self,
        phase_measurement: np.ndarray,
        current_gamma_curve=None,
        intensities: Union[np.ndarray, None] = None,
        verbose: bool = False,
    ):
        self.intensities = intensities
        self.verbose = verbose
        self.phase_measurement = phase_measurement
        self.current_gamma_curve = current_gamma_curve
        self._initialize_intensities()
        self.splines_available = False
        self._tolerance_knots = 0.5
        self._k_spline = 3   # Grado de la spline
        self._s_spline = 10  # s controla la suavidad de la spline
        self._at_least_middle_knots = 2

        self.smooth_curve_monotonic()

    @property
    def tolerance_knots(self):
        return self._tolerance_knots

    @tolerance_knots.setter
    def tolerance_knots(self, value):
        assert 0 < value < 0.5
        self._tolerance_knots = value

    @property
    def k_spline(self):
        return self._k_spline

    @k_spline.setter
    def k_spline(self, value):
        assert value > 0 and isinstance(value, int)
        self._k_spline = value

    @property
    def s_spline(self):
        return self._s_spline

    @s_spline.setter
    def s_spline(self, value):
        assert value > 0
        self._s_spline = value

    @property
    def at_least_middle_knots(self):
        return self._at_least_middle_knots

    @at_least_middle_knots.setter
    def at_least_middle_knots(self, value):
        assert value >= 0 and isinstance(value, int)
        self._at_least_middle_knots = value

    def _print(self, message, *args, **kwargs):
        if self.verbose:
            print(message, *args, **kwargs)

    def _initialize_intensities(self):
        if self.intensities is None:
            assert len(self.phase_measurement) == 256
            self.intensities = np.arange(len(self.phase_measurement))

    def _add_knots(self):
        # Acceso al grado, los knots y coeficientes
        self.tck = self.initial_spline.tck

        self.n_knots = len(self.tck[0])
        self.n_coeffs = len(self.tck[1])
        self._print(
            f"Spline inicial. Grado: {self.k_spline}. Número de knots: {self.n_knots}. "
            f"Número de coeficientes: {self.n_coeffs}"
        )

        self.n_middle_knots = self.n_knots - (self.k_spline + 1) * 2
        if self.n_middle_knots < self.at_least_middle_knots:
            sep = (self.tck[0][-1] - self.tck[0][0]) / self.at_least_middle_knots
            middle_knots = self.tck[0][self.k_spline] + sep / 2 + np.arange(self.at_least_middle_knots) * sep
            if self.n_middle_knots > 0:
                current_middle_knots = self.tck[0][self.k_spline + 1:self.k_spline + 1 + self.n_middle_knots]
                for cmk in current_middle_knots:
                    idx_to_remove = np.argmin(np.abs(middle_knots - cmk))
                    middle_knots = np.delete(middle_knots, idx_to_remove)
            for mk in middle_knots:
                self.spline = self.initial_spline.insert_knot(mk)
            self.tck = self.spline.tck
            self.n_middle_knots = self.at_least_middle_knots

            self.n_knots = len(self.tck[0])
            self.n_coeffs = len(self.tck[1])
            self._print(
                f"Spline con knots adicionales. Grado: {self.k_spline}. Número de knots: {self.n_knots}. "
                f"Número de coeficientes: {self.n_coeffs}"
            )

    @staticmethod
    def _get_spline_from_fit(x, *args):
        t = args[0].copy()
        k = args[2]
        if args[1] > 0:
            t[k + 1:k + 1 + args[1]] = x[:args[1]]
        c = x[args[1]:]

        spline = BSpline(t, c, k)
        return spline

    @staticmethod
    def _fun_min(x, *args) -> float:
        spline = GammaCurver._get_spline_from_fit(x, *args)
        x_pos = args[3]
        values = args[4]
        return np.sum((spline(x_pos) - values) ** 2)

    def _get_seed_and_bounds(self):
        if self.n_middle_knots > 0:
            middle_knots = self.tck[0][self.k_spline + 1:self.k_spline + 1 + self.n_middle_knots]
            x0 = np.concatenate((middle_knots, self.tck[1]))
            delta_knots = np.diff(self.tck[0][self.k_spline:self.k_spline + 2 + self.n_middle_knots])
            bounds = []
            for i in range(self.n_middle_knots):
                left = middle_knots[i] - self.tolerance_knots * delta_knots[i]
                right = middle_knots[i] + self.tolerance_knots * delta_knots[i + 1]
                bounds.append((left, right))
            for i in range(self.n_coeffs):
                bounds.append((None, None))
        else:
            x0 = self.tck[1]
            bounds = None
        return x0, bounds

    def _fun_constraint(x, *args) -> float:
        spline = GammaCurver._get_spline_from_fit(x, *args)
        derivative = spline.derivative()
        x_pos = args[3]
        return np.min(derivative(x_pos))

    def smooth_curve_monotonic(self):
        nest = 2 * self.k_spline + 2 + self.at_least_middle_knots
        self.initial_spline = make_splrep(
            self.intensities, self.phase_measurement, s=self.s_spline, k=self.k_spline, nest=nest
        )

        self._add_knots()

        args = (self.tck[0], self.n_middle_knots, self.k_spline, self.intensities, self.phase_measurement)

        x0, bounds = self._get_seed_and_bounds()

        constraint = {"type": "ineq", "fun": GammaCurver._fun_constraint, "args": args}

        res = minimize(
            GammaCurver._fun_min, x0, args=args, bounds=bounds, constraints=constraint, method="COBYLA",
            options={"maxiter": 1000}
        )

        self.spline_constrained = self._get_spline_from_fit(res.x, *args)
        self.splines_available = True

    def get_initial_spline(self):
        return self.initial_spline(self.intensities)

    def get_spline_constrained(self):
        return self.spline_constrained(self.intensities)

    def get_spline_constrained_derivative(self):
        return self.spline_constrained.derivative()(self.intensities)

    def plot_result(self, y_true=None):
        if self.splines_available is False:
            print("Aun no se ha realizado el ajuste")
            return
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig: plt.Figure = fig
        axs: list[plt.Axes] = axs
        if y_true is not None:
            axs[0].plot(self.intensities, y_true, label='Curva verdadera', linewidth=2)
        axs[0].scatter(self.intensities, self.phase_measurement, color='red', s=10, label='Datos ruidosos')
        # Graficar la spline ajustada sin constraint
        axs[0].plot(self.intensities, self.get_initial_spline(), label='Spline suavizante', color='green', linewidth=2)
        # Graficar la spline con constraint para comparar
        axs[0].plot(self.intensities, self.get_spline_constrained(), '--', label='Spline constrained', color='orange',
                    linewidth=2)
        axs[0].set_xlabel('intensities')
        axs[0].set_ylabel('phase')
        axs[0].legend()
        fig.suptitle('Resultado de spline suavizante con constraint')
        axs[1].plot(self.intensities, self.get_spline_constrained_derivative(), label='Derivada de la spline',
                    color='purple', linewidth=2)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    n_points = 256
    amplitude_sin = 0.3
    noise_std = 0.2

    intensities = np.arange(n_points)
    y_lin = np.linspace(0, 2 * np.pi, n_points)
    curve = amplitude_sin * np.sin(y_lin) + y_lin  # Curva creciente
    noise = np.random.normal(0, noise_std, n_points)
    measured_curve = curve + noise
    gammer = GammaCurver(measured_curve, intensities=intensities)
    gammer.plot_result(y_true=curve)
