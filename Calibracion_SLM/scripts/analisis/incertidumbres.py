import numpy as np
import matplotlib.pyplot as plt
from funciones import simular_imagen, fase_hilbert
from scipy.signal import find_peaks
import random


imagen = simular_imagen(
    frecuencia=100, angulo_franjas_max=0, angulo_slm_max=0, fase2=np.pi
)

altura_rec = 90
ancho_rec = 300
x_rec1 = 200
y_rec1 = 150
x_rec2 = 200
y_rec2 = 300

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
print(frecuencias[indi])
plt.plot(frecuencias[: n // 2], np.abs(f_signal[: n // 2]))
plt.show()

prueba = True
if prueba:
    diferencias = np.zeros(1000)
    for i in range(1000):
        fase = random.uniform(0, 2 * np.pi)
        imagen = simular_imagen(
            frecuencia=100,
            angulo_franjas_max=0,
            angulo_slm_max=0,
            fase2=fase,
            amplitud_imperfecciones=0,
        )
        # plt.imshow(imagen)
        # plt.pause(.01)
        # plt.cla()
        recorte1 = imagen[y_rec1 : y_rec1 + altura_rec, x_rec1 : x_rec1 + ancho_rec]
        recorte2 = imagen[y_rec2 : y_rec2 + altura_rec, x_rec2 : x_rec2 + ancho_rec]
        c1, c2 = fase_hilbert(recorte1, recorte2, filter_frec=0.15666666666666668)
        diferencias[i] = np.angle(np.exp(1j * c1) / np.exp(1j * c2)) - fase + np.pi
        print(i)
    rmsd = np.sqrt(np.mean(diferencias**2))
    print(f"RMSD: {rmsd} rad")
    plt.hist(diferencias, bins=30)
    plt.show()
    plt.plot(diferencias)
    plt.show()
