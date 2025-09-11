import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gamma_curver import GammaCurver


## Este es el código para obtener una nueva curva_gamma a partir de la actual y de un
## set de mediciones de fase

#Importo la gamma curve actual

# df = pd.read_csv("/home/lorenzo/Labo_6/SLM_csvs/5-6_C20_lin2,1pi_532nm_274-148V.csv", header = None)
fase_medida = pd.read_csv('Calibracion_SLM/data/fase_medida_lineal_383_picoround.csv')
fase_medida = np.array(fase_medida.iloc[:,0])
fase_medida= fase_medida-min(fase_medida)
print(fase_medida)
df = pd.read_csv('../curva_gamma_lineal.csv', header = None)
df = df.astype(np.uint16)
curva_gamma_1024= df[0].values
valores_gris = np.arange(0, 1024)
print(curva_gamma_1024)
print(type(curva_gamma_1024), curva_gamma_1024.size)
plt.plot(valores_gris, curva_gamma_1024)
plt.show()

curva_gamma_256 = curva_gamma_1024[::4] ## tomo solo 1 valor cada 4


g = GammaCurver(phase_measurement= fase_medida, current_gamma_curve= curva_gamma_256, 
                graylevels=np.arange(0, 255, 5),
                verbose = True)

g.plot_result()

# plt.scatter(np.arange(330), g.spline_constrained(np.arange(330)))
# plt.show()

# Si esta ok la linearizo y guardo

g.linearize_gamma()

plt.scatter(np.arange(256), g.new_gamma_curve)
plt.xlabel('Valor de gris')
plt.ylabel('LUT')
plt.savefig('nueva_gamma.png')
plt.show()

g.new_gamma_curve[2] = 17

plt.scatter(np.arange(256), g.new_gamma_curve)
plt.xlabel('Valor de gris')
plt.ylabel('LUT')
plt.savefig('nueva_gamma.png')
plt.show()

new_gamma_1024 = np.repeat(g.new_gamma_curve, 4)

plt.scatter(np.arange(1024), new_gamma_1024)
plt.show()

df_gamma = pd.DataFrame(new_gamma_1024)
print(df_gamma)
df_gamma.to_csv('gamma_corregida.csv', index = False)

# g.plot_new_gamma()
# g.save_csv("gamma_curve_g.csv")

