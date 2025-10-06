import numpy as np
import matplotlib.pyplot as plt
from funciones import simular_imagen, fase_hilbert
from scipy.signal import find_peaks
import random
import tqdm

sigma_imperfecciones = 200
amplitud_imperfecciones = 4
angulo_franjas_max = 0.5

imagenes = simular_imagen(
    frecuencia=100, angulo_franjas_max=angulo_franjas_max, angulo_slm_max=0, fase2=[np.pi, 0],
    amplitud_imperfecciones=amplitud_imperfecciones,
    sigma_imperfecciones=sigma_imperfecciones,
)

altura_rec = 20
ancho_rec = 300
x_rec1 = 200
y_rec1 = 220
x_rec2 = 200
y_rec2 = 270

plt.imshow(np.array(imagenes[1]))
plt.show()

n_pruebas = 200
iteracion_incertidumbre = False
imagen = np.array(imagenes[0])
recorte1 = imagen[y_rec1 : y_rec1 + altura_rec, x_rec1 : x_rec1 + ancho_rec]
recorte2 = imagen[y_rec2 : y_rec2 + altura_rec, x_rec2 : x_rec2 + ancho_rec]
fig, ax = plt.subplots()
ax.imshow(imagen)
rect1 = plt.Rectangle(
    (x_rec1, y_rec1),
    ancho_rec,
    altura_rec,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
)
rect2 = plt.Rectangle(
    (x_rec2, y_rec2),
    ancho_rec,
    altura_rec,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
)
ax.add_patch(rect1)
ax.add_patch(rect2)
plt.show()

fases = np.linspace(0, 2 * np.pi, 256)

signal = np.sum(recorte1, axis=0)
signal = signal - np.mean(signal)
f_signal = np.fft.fft(signal)
frecuencias = np.fft.fftfreq(len(signal), d=1.0)
n = len(f_signal)
indi = np.argmax(np.abs(f_signal[: n // 2]))
pico_frec = frecuencias[indi]
print(pico_frec)
plt.plot(frecuencias[: n // 2], np.abs(f_signal[: n // 2]))
plt.show()

if iteracion_incertidumbre:
    diferencias = np.zeros(n_pruebas)
    iterator = tqdm.tqdm(range(n_pruebas))
    for i in iterator:
        fase = random.uniform(- np.pi, np.pi)
        imagenes = simular_imagen(
            frecuencia=100,
            angulo_franjas_max=angulo_franjas_max,
            angulo_slm_max=0,
            fase2=[0, fase],
            amplitud_imperfecciones=amplitud_imperfecciones,
            sigma_imperfecciones=sigma_imperfecciones,
        )
        # plt.imshow(imagen)
        # plt.pause(.01)
        # plt.cla()
        recorte1_0 = imagenes[0][y_rec1 : y_rec1 + altura_rec, x_rec1 : x_rec1 + ancho_rec]
        recorte2_0 = imagenes[0][y_rec2 : y_rec2 + altura_rec, x_rec2 : x_rec2 + ancho_rec]
        c1_0, c2_0 = fase_hilbert(recorte1_0, recorte2_0, filter_frec=pico_frec)
        recorte1_fase = imagenes[1][y_rec1 : y_rec1 + altura_rec, x_rec1 : x_rec1 + ancho_rec]
        recorte2_fase = imagenes[1][y_rec2 : y_rec2 + altura_rec, x_rec2 : x_rec2 + ancho_rec]
        c1_fase, c2_fase = fase_hilbert(recorte1_fase, recorte2_fase, filter_frec=pico_frec)
        diferencia_0 = np.angle(np.exp(1j * c1_0) / np.exp(1j * c2_0))
        diferencia_fase = np.angle(np.exp(1j * c1_fase) / np.exp(1j * c2_fase))
        diferencias[i] = np.angle(np.exp(1j * diferencia_fase) / np.exp(1j * diferencia_0) / np.exp(1j * fase))
    rmsd = np.sqrt(np.mean(diferencias**2))
    print(f"RMSD: {rmsd} rad")
    plt.hist(diferencias, bins=30)
    plt.show()
    plt.plot(diferencias)
    plt.show()
