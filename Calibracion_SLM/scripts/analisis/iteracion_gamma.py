import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gamma_curver import GammaCurver


## Este es el código para obtener una nueva curva_gamma a partir de la actual y de un
## set de mediciones de fase

#Importo la gamma curve actual

df = pd.read_csv("/home/lorenzo/Labo_6/SLM_csvs/5-6_C20_lin2,1pi_532nm_274-148V.csv", header = None)
curva_gamma_1024= df[0].values
valores_gris = np.arange(0, 1024)
print(curva_gamma_1024)
print(type(curva_gamma_1024), curva_gamma_1024.size)
plt.plot(valores_gris, curva_gamma_1024)
plt.show()

curva_gamma_256 = curva_gamma_1024[::4] ## tomo solo 1 valor cada 4


g = GammaCurver(phase_measurement= fase_medida, current_gamma_curve= curva_gamma_256, 
                graylevels=np.arange(256),
                verbose = True)

g.plot_result()

# Si esta ok la linearizo y guardo

g.linearize_gamma()
g.plot_new_gamma()
g.save_csv("gamma_curve_g.csv")

