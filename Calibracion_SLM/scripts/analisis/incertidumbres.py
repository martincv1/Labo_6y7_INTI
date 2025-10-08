import numpy as np
import matplotlib.pyplot as plt
from funciones import simular_imagen, fase_hilbert
import random
import tqdm

Nx = 640
Ny = 512
visibilidad_franjas = 0.6
frecuencia_franjas = 120
hilbert_filter_band = 0.04  # Ancho de la ventana del filtro pasa banda en la transformada de Fourier (frecuencia normalizada)
sigma_imperfecciones = 180
imperfecciones_centradas = True
amplitud_imperfecciones = 5
angulo_franjas_max = 3
amplitud_parasitic_fringes = 0.5
frecuencia_parasitic_fringes = 0.3

frecuencia_franjas_norm = frecuencia_franjas / Nx
print(f"Frecuencia normalizada de las franjas: {frecuencia_franjas_norm} ciclos/pixel")

imagenes = simular_imagen(
    Nx=Nx, Ny=Ny,
    visibility=visibilidad_franjas,
    frecuencia=frecuencia_franjas,
    angulo_franjas_max=angulo_franjas_max,
    angulo_slm_max=0,
    fase2=[np.pi, 0],
    imperfecciones_centradas=imperfecciones_centradas,
    amplitud_imperfecciones=amplitud_imperfecciones,
    sigma_imperfecciones=sigma_imperfecciones,
    parasitic_fringes_amplitude=amplitud_parasitic_fringes,
    parasitic_fringes_frequency=frecuencia_parasitic_fringes,
)

altura_rec = 20
ancho_rec = 300
x_rec1 = 200
y_rec1 = 220
x_rec2 = 200
y_rec2 = 270
width_search_peak = 0.1     # Ancho de la ventana de búsqueda del pico en el espectro de Fourier (frecuencia normalizada)

plt.imshow(np.array(imagenes[1]))
plt.show()

n_pruebas = 1000
iteracion_incertidumbre = True

imagen = np.array(imagenes[0])
recorte1 = imagen[y_rec1: y_rec1 + altura_rec, x_rec1: x_rec1 + ancho_rec]
recorte2 = imagen[y_rec2: y_rec2 + altura_rec, x_rec2: x_rec2 + ancho_rec]
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
rango_busqueda = np.arange(int((frecuencia_franjas_norm - width_search_peak / 2) * n),
                           int((frecuencia_franjas_norm + width_search_peak / 2) * n))
indi = np.argmax(np.abs(f_signal[rango_busqueda])) + rango_busqueda[0]
pico_frec = frecuencias[indi]
plt.plot(frecuencias[: n // 2], np.abs(f_signal[: n // 2]))
plt.plot(frecuencias[rango_busqueda[0]], np.abs(f_signal[rango_busqueda[0]]), "gx")
plt.plot(frecuencias[rango_busqueda[-1]], np.abs(f_signal[rango_busqueda[-1]]), "gx")
plt.plot(pico_frec, np.abs(f_signal[indi]), "ro")
plt.xlabel("Frecuencia (ciclos/pixel)")
plt.ylabel("Amplitud")
plt.title("Espectro de Fourier del recorte")
plt.show()

if iteracion_incertidumbre:
    diferencias = np.zeros(n_pruebas)
    iterator = tqdm.tqdm(range(n_pruebas))
    for i in iterator:
        fase = random.uniform(-np.pi, np.pi)
        imagenes = simular_imagen(
            Nx=Nx, Ny=Ny,
            visibility=visibilidad_franjas,
            frecuencia=frecuencia_franjas,
            angulo_franjas_max=angulo_franjas_max,
            angulo_slm_max=0,
            fase2=[0, fase],
            imperfecciones_centradas=imperfecciones_centradas,
            amplitud_imperfecciones=amplitud_imperfecciones,
            sigma_imperfecciones=sigma_imperfecciones,
            parasitic_fringes_amplitude=amplitud_parasitic_fringes,
            parasitic_fringes_frequency=frecuencia_parasitic_fringes,
        )
        # plt.imshow(imagen)
        # plt.pause(.01)
        # plt.cla()
        recorte1_0 = imagenes[0][
            y_rec1: y_rec1 + altura_rec, x_rec1: x_rec1 + ancho_rec
        ]
        recorte2_0 = imagenes[0][
            y_rec2: y_rec2 + altura_rec, x_rec2: x_rec2 + ancho_rec
        ]
        c1_0, c2_0 = fase_hilbert(recorte1_0, recorte2_0, filter_frec=pico_frec, filter_band=hilbert_filter_band)
        recorte1_fase = imagenes[1][
            y_rec1: y_rec1 + altura_rec, x_rec1: x_rec1 + ancho_rec
        ]
        recorte2_fase = imagenes[1][
            y_rec2: y_rec2 + altura_rec, x_rec2: x_rec2 + ancho_rec
        ]
        c1_fase, c2_fase = fase_hilbert(
            recorte1_fase, recorte2_fase, filter_frec=pico_frec, filter_band=hilbert_filter_band
        )
        diferencia_0 = np.angle(np.exp(1j * c1_0) / np.exp(1j * c2_0))
        diferencia_fase = np.angle(np.exp(1j * c1_fase) / np.exp(1j * c2_fase))
        diferencias[i] = np.angle(np.exp(1j * (diferencia_fase - diferencia_0 - fase)))

        if diferencias[i] > 0.1:
            c1_0, c2_0 = fase_hilbert(
                recorte1_0, recorte2_0, filter_frec=pico_frec, debug=True, filter_band=hilbert_filter_band
            )
            c1_fase, c2_fase = fase_hilbert(
                recorte1_fase, recorte2_fase, filter_frec=pico_frec, debug=True, filter_band=hilbert_filter_band
            )
    rmsd = np.sqrt(np.mean(diferencias**2))
    print(f"RMSD: {rmsd} rad")
    plt.hist(diferencias, bins=30)
    plt.show()
    plt.plot(diferencias)
    plt.show()
