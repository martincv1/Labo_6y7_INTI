import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gamma_curver import GammaCurver
from scipy.interpolate import interp1d

## Este es el código para obtener una nueva curva_gamma a partir de la actual y de un
## set de mediciones de fase

# Importo la gamma curve actual

# df = pd.read_csv("/home/lorenzo/Labo_6/SLM_csvs/5-6_C20_lin2,1pi_532nm_274-148V.csv", header = None)
fase_medida = pd.read_csv(
    "Calibracion_SLM/data/fase_medida_corregida_1809_3.csv", header=None
)
fase_medida = np.array(fase_medida.iloc[:, 0])
fase_medida = fase_medida - min(fase_medida)
print(fase_medida)
gamma_en_slm = pd.read_csv("gamma_corregida_1809_3.csv", header=None)
print(gamma_en_slm)
# df = df.astype(np.uint16)
# print(df)
curva_gamma_1024 = gamma_en_slm[0].astype(int)
curva_gamma_1024 = curva_gamma_1024[0:]
print(curva_gamma_1024)
valores_gris = np.arange(0, 1024)
print(curva_gamma_1024)
print(type(curva_gamma_1024), len(curva_gamma_1024))
plt.plot(valores_gris, curva_gamma_1024)
plt.title("Curva gamma leida, usada para medir fase")
plt.xlabel("Valores de gris (repetidos hasta 1024)")
plt.ylabel("LUT")
plt.show()

curva_gamma_256 = curva_gamma_1024[::4]  ## tomo solo 1 valor cada 4
curva_gamma_256 = curva_gamma_256.ravel()


g = GammaCurver(
    phase_measurement=fase_medida,
    current_gamma_curve=curva_gamma_256,
    graylevels=np.arange(256),
    verbose=True,
)

g.plot_result()

plt.scatter(np.arange(383), g.spline_constrained(np.arange(383)))
plt.title("Fase medida interpolada por spline creciente")
plt.xlabel("LUT")
plt.ylabel("Fase[rad]")
plt.show()

tol = 0
fases = np.linspace(
    g.initial_spline((np.max(curva_gamma_256) - np.min(curva_gamma_256)) / 2)
    - (np.pi - (np.pi / 256))
    - tol,
    g.initial_spline((np.max(curva_gamma_256) - np.min(curva_gamma_256)) / 2)
    + (np.pi - (np.pi / 256))
    + tol,
    256,
)
nueva_gamma_256 = np.zeros(256)

xx = np.linspace(0, 383, 5000)
yy = g.initial_spline(np.linspace(0, 383, 5000))

f = interp1d(yy, xx)
plt.plot(fases, f(fases))
plt.xlabel("Fase")
plt.ylabel("LUT")
plt.show()
for i in range(256):
    if i == 0:
        nueva_gamma_256[i] = np.floor(f(fases[i])).astype(np.uint16)
    elif i == 256:
        nueva_gamma_256[i] = np.ceil(f(fases[i])).astype(np.uint16)
    else:
        nueva_gamma_256[i] = np.round(f(fases[i])).astype(np.uint16)

plt.plot(nueva_gamma_256)
plt.plot(nueva_gamma_256)
plt.title("Nueva gamma 256")
plt.xlabel("Valor de gris")
plt.ylabel("LUT")
plt.show()

dif_256 = np.diff(nueva_gamma_256)
lut_repetidos = np.sum(dif_256 == 0)
print(lut_repetidos)

nueva_gamma_1024 = np.repeat(nueva_gamma_256, 4)

plt.plot(nueva_gamma_1024)
plt.title("Nueva gamma 1024")
plt.xlabel("Valor de gris extendido")
plt.ylabel("LUT")
plt.show()

# df_gamma = pd.DataFrame(nueva_gamma_1024)
# print(df_gamma)
# df_gamma.to_csv("gamma_corregida_1809_4.csv", index=False, header=None)
