import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re
from datetime import datetime
from scipy.signal import hilbert, find_peaks
from scipy.optimize import curve_fit
import cv2
from funciones import fase_hilbert, rotate_bound
import pandas as pd
from scipy.interpolate import interp1d


dir_path = r"Calibracion_SLM\data\fase_tension_4.32-0.82"

file_path = r"Calibracion_SLM\data\fase_tension_4.32-0.82\1306_I150_T21.pkl"
# with open(file_path, mode = 'rb') as f:
#     imagen = pickle.load(f)
# imagen = rotate_bound(imagen, 2)
# plt.imshow(imagen)
# plt.show()
prueba = False
if prueba:
    with open(file_path, mode="rb") as f:
        imagen = pickle.load(f)

    imagen = rotate_bound(imagen, 2)

    altura_rec = 30
    ancho_rec = 300
    x_rec1 = 490
    y_rec1 = 440
    x_rec2 = 490
    y_rec2 = 510
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
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].imshow(recorte1, cmap="gray")
    ax[0].set_title("Recorte 1")
    ax[0].set_xlabel("Pixel")
    ax[1].imshow(recorte2, cmap="gray")
    ax[1].set_title("Recorte 2")
    ax[1].set_xlabel("Pixel")
    plt.tight_layout()
    plt.show()

    plt.plot(np.sum(recorte1, axis=0), label="Recorte 1")
    # plt.plot(np.sum(recorte2, axis=0), label='Recorte 2')
    plt.legend()
    plt.title("Señal de un recorte")
    plt.xlabel("Pixel")
    plt.ylabel("Intensidad")
    plt.show()

    signal1 = np.sum(recorte1, axis=0)
    signal2 = np.sum(recorte2, axis=0)

    signal1 = signal1 - np.mean(signal1)
    signal2 = signal2 - np.mean(signal2)

    f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
    f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
    n = len(f_signal1)
    freq1_ind = np.argmax(np.abs(f_signal1[: n // 2]))
    freq2_ind = np.argmax(np.abs(f_signal2[: n // 2]))
    frecuencias = np.fft.fftfreq(len(signal1), d=1)
    # freq1 = frecuencias[freq1_ind]
    # freq2 = frecuencias[freq2_ind]

    p1, _ = find_peaks(
        np.abs(f_signal1)[: n // 2], prominence=0.3 * np.max(np.abs(f_signal1))
    )
    p2, _ = find_peaks(
        np.abs(f_signal2)[: n // 2], prominence=0.3 * np.max(np.abs(f_signal2))
    )
    plt.plot(frecuencias[: n // 2], np.abs(f_signal1)[: n // 2])
    plt.scatter(frecuencias[p1], np.abs(f_signal1)[p1], color="r")
    plt.axhline(0.5 * np.max(np.abs(f_signal1)))
    plt.show()
    freq1 = np.max(frecuencias[p1])
    freq2 = np.max(frecuencias[p2])
    f_bandwidth = 0.01
    gauss_mask1 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth) ** 2) + np.exp(
        -0.5 * ((frecuencias + freq1) / f_bandwidth) ** 2
    )
    gauss_mask2 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth) ** 2) + np.exp(
        -0.5 * ((frecuencias + freq1) / f_bandwidth) ** 2
    )
    fft_filtrada1 = f_signal1 * gauss_mask1
    fft_filtrada2 = f_signal2 * gauss_mask2
    filtra1 = np.fft.ifft(fft_filtrada1).real
    filtra2 = np.fft.ifft(fft_filtrada2).real

    plt.plot(frecuencias, gauss_mask1)
    plt.plot(frecuencias, np.abs(f_signal1) / np.max(np.abs(f_signal1)))
    plt.show()
    plt.plot(frecuencias, gauss_mask2)
    plt.plot(frecuencias, np.abs(f_signal2) / np.max(np.abs(f_signal2)))
    plt.show()

    plt.plot(filtra1)
    plt.show()
    plt.plot(filtra2)
    plt.show()

    hil1 = hilbert(filtra1)
    hil2 = hilbert(filtra2)
    lin1 = np.unwrap(np.angle(hil1))
    lin2 = np.unwrap(np.angle(hil2))

    plt.plot(lin1)
    plt.plot(lin2)
    plt.show()
video = False
if video:
    for i in np.arange(0, 255, 5):
        file_path = os.path.join(dir_path, f"1306_I{i}_T21.pkl")
        if not os.path.exists(file_path):
            continue
        with open(file_path, mode="rb") as f:
            imag = pickle.load(f)
        imag = rotate_bound(imag, 2)
        plt.imshow(imag[150:800, 100:1300])
        plt.pause(0.05)
        plt.cla()
run = False
if run:

    def lineal(x, m, c):
        return m * x + c

    c1 = []
    c2 = []
    m1 = []
    m2 = []
    for i in np.arange(0, 255, 5):
        file_path = os.path.join(dir_path, f"1306_I{i}_T21.pkl")
        if not os.path.exists(file_path):
            continue
        with open(file_path, mode="rb") as f:
            imag = pickle.load(f)
        imag = rotate_bound(imag, 2)
        altura_rec = 30
        ancho_rec = 300
        x_rec1 = 490
        y_rec1 = 440
        x_rec2 = 490
        y_rec2 = 510
        recorte1 = imag[y_rec1 : y_rec1 + altura_rec, x_rec1 : x_rec1 + ancho_rec]
        recorte2 = imag[y_rec2 : y_rec2 + altura_rec, x_rec2 : x_rec2 + ancho_rec]

        signal1 = np.sum(recorte1, axis=0)
        signal2 = np.sum(recorte2, axis=0)

        signal1 = signal1 - np.mean(signal1)
        signal2 = signal2 - np.mean(signal2)

        f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
        f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
        n = len(f_signal1)
        freq1_ind = np.argmax(np.abs(f_signal1[: n // 2]))
        freq2_ind = np.argmax(np.abs(f_signal2[: n // 2]))
        frecuencias = np.fft.fftfreq(len(signal1), d=1)
        # freq1 = frecuencias[freq1_ind]
        # freq2 = frecuencias[freq2_ind]
        p1, _ = find_peaks(
            np.abs(f_signal1)[: n // 2], prominence=0.25 * np.max(np.abs(f_signal1))
        )
        p2, _ = find_peaks(
            np.abs(f_signal2)[: n // 2], prominence=0.25 * np.max(np.abs(f_signal2))
        )
        print(frecuencias[p1])
        # plt.plot(frecuencias[:n//2], np.abs(f_signal1)[:n//2])
        # plt.scatter(frecuencias[p1], np.abs(f_signal1)[p1], color ='r')
        # plt.title(f'I = {i}. 1: {len(p1)}, 2: {len(p2)}')
        # plt.pause(.05)
        # plt.cla()

        # freq1 = np.max(frecuencias[p1])
        # freq2 = np.max(frecuencias[p2])
        # freq1 = frecuencias[p1[-2]]
        # freq2 = frecuencias[p2[-2]]
        # print(freq1)
        freq1 = 0.08666667
        freq2 = freq1
        f_bandwidth = 0.01
        gauss_mask1 = np.exp(
            -0.5 * ((frecuencias - freq1) / f_bandwidth) ** 2
        ) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth) ** 2)
        gauss_mask2 = np.exp(
            -0.5 * ((frecuencias - freq1) / f_bandwidth) ** 2
        ) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth) ** 2)
        fft_filtrada1 = f_signal1 * gauss_mask1
        fft_filtrada2 = f_signal2 * gauss_mask2
        filtra1 = np.fft.ifft(fft_filtrada1).real
        filtra2 = np.fft.ifft(fft_filtrada2).real

        hil1 = hilbert(filtra1)
        hil2 = hilbert(filtra2)
        lin1 = np.unwrap(np.angle(hil1))
        lin2 = np.unwrap(np.angle(hil2))
        inf = 50
        sup = 250
        lin1_ajust = lin1[inf:sup]
        lin2_ajust = lin2[inf:sup]
        popt1, pcov1 = curve_fit(lineal, np.arange(inf, sup), lin1_ajust)
        popt2, pcov2 = curve_fit(lineal, np.arange(inf, sup), lin2_ajust)

        c1.append(popt1[1])  # guardo los valores de la ordenada al origen
        c2.append(popt2[1])  # guardo los valores de la ordenada al origen
        m1.append(popt1[0])
        m2.append(popt2[0])

        print(i)

    plt.plot(m1)
    plt.plot(m2)
    plt.show()
    c1 = np.array(c1)
    c2 = np.array(c2)
    exp_c1 = np.exp(1j * c1)
    exp_c2 = np.exp(1j * c2)
    plt.scatter(np.arange(0, 255, 5), c1, label="Ordenada al origen 1")
    plt.legend()
    plt.show()
    plt.scatter(np.arange(0, 255, 5), c2, label="Ordenada al origen 2")
    plt.legend()
    plt.show()
    fase1 = np.unwrap(c1)
    fase2 = np.unwrap(c2)
    diferencia = fase1 - fase2
    plt.plot(diferencia)
    plt.show()
    dif_compleja = np.angle(exp_c1 / exp_c2)
    plt.plot(dif_compleja)
    plt.show()
    plt.scatter(
        np.arange(0, 255, 5), np.unwrap(dif_compleja) - min(np.unwrap(dif_compleja))
    )
    # plt.title('Resultados 4.32-0.82')
    plt.xlabel("Valor de gris")
    plt.ylabel("Diferencia de fase")
    # plt.savefig('Calibracion_SLM/data/resultados_4.32-0.82.png')
    plt.show()
    datos = pd.DataFrame(np.unwrap(dif_compleja))
    datos.to_csv("Calibracion_SLM/data/0.82-4.32.csv")

intento = False
if intento:
    # imagen = cv2.imread('Calibracion_SLM/data/fotos_rot/3004_I200_1_T22_r.png')
    # imagen = rotate_bound(imagen, -27)
    # plt.imshow(imagen)
    # plt.show()
    # altura_rec = 30
    # ancho_rec = 120
    # x_rec1 = 520
    # y_rec1 = 480
    # x_rec2 = 520
    # y_rec2 = 585
    # recorte1 = imagen[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec, 0]
    # recorte2 = imagen[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec, 0]

    # fig, ax = plt.subplots()
    # ax.imshow(imagen)
    # rect1 = plt.Rectangle((x_rec1,y_rec1), ancho_rec, altura_rec, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    # rect2 = plt.Rectangle((x_rec2,y_rec2), ancho_rec, altura_rec, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    # ax.add_patch(rect1)
    # ax.add_patch(rect2)
    # plt.show()
    # senal = np.sum(recorte1, axis=0)
    # plt.plot(senal)
    # plt.show()
    # f_senal = np.fft.fft(senal-np.mean(senal))
    # n = len(f_senal)
    # plt.plot(np.abs(f_senal))
    # plt.show()
    # indi = np.argmax(np.abs(f_senal)[:n//2])
    # frecs = np.fft.fftfreq(len(senal), d=1)
    # print(frecs[indi])
    # print(indi)
    fases1 = np.zeros(256)
    fases2 = np.zeros(256)
    for i in range(256):
        imagen0 = cv2.imread(f"Calibracion_SLM/data/fotos_rot/3004_I{i}_0_T22_r.png")
        imagen0 = imagen0[:, :, 0]
        for j in range(9):
            imagen = cv2.imread(
                f"Calibracion_SLM/data/fotos_rot/3004_I{i}_{j+1}_T22_r.png"
            )
            imagen = imagen[:, :, 0]
            imagen0 = imagen0 + imagen
        imagen = imagen0
        imagen = rotate_bound(imagen, -27)
        altura_rec = 30
        ancho_rec = 120
        x_rec1 = 520
        y_rec1 = 480
        x_rec2 = 520
        y_rec2 = 585
        recorte1 = imagen[y_rec1 : y_rec1 + altura_rec, x_rec1 : x_rec1 + ancho_rec]
        recorte2 = imagen[y_rec2 : y_rec2 + altura_rec, x_rec2 : x_rec2 + ancho_rec]

        fase1, fase2 = fase_hilbert(recorte1, recorte2, 0.11666666666666667)
        fases1[i] = fase1
        fases2[i] = fase2
        print(i)

    exp_1 = np.exp(1j * fases1)
    exp_2 = np.exp(1j * fases2)
    dif = np.unwrap(np.angle(exp_1 / exp_2))
    plt.plot(np.arange(256), dif - min(dif))
    plt.show()

fasevolt = False
if fasevolt:
    datos = pd.read_csv("Calibracion_SLM/data/0.82-4.32.csv")
    datos = datos.iloc[:, 1]
    df = pd.read_csv("Calibracion_SLM/data/gamma_curve_0.csv")
    df = df.iloc[1:, 0]
    gamma = np.array(df.astype(float))
    print(gamma)
    plt.plot(gamma)
    plt.xlabel(' "Valor de gris" ')
    plt.ylabel("LUT")
    plt.show()
    print(max(gamma), min(gamma))
    v1 = 4.32
    v2 = 0.82
    max_gamma = 330
    tension = v2 + (v1 - v2) * gamma / max_gamma
    plt.plot(tension)
    plt.xlabel(' "Valor de gris" ')
    plt.ylabel("Voltaje")
    plt.show()
    tension_gris = np.zeros(256)
    for i in range(len(gamma)):
        if i % 4 == 0:
            tension_gris[i // 4] = tension[i]
    plt.scatter(np.arange(256), tension_gris)

    f_interp = interp1d(np.arange(256), tension_gris, kind="linear")
    plt.scatter(np.arange(0, 255, 5), f_interp(np.arange(0, 255, 5)), color="red")
    plt.show()
    print(len(datos), len(np.arange(0, 255, 5)))
    plt.scatter(np.arange(0, 255, 5), datos - min(datos))
    plt.xlabel("Valor de gris")
    plt.ylabel("Diferencia de fase (rad)")
    plt.show()
    plt.scatter(f_interp(np.arange(0, 255, 5)), datos - min(datos))
    plt.xlabel("Voltaje (V)")
    plt.ylabel("Diferencia de fase (rad)")
    plt.show()

show_ubicar_pico = False
# *******************************************************************
img_path_format = r"data\repetibilidad\12 0.69-1.42\imgs\3009_I{}_T21_{}.png"
indice_foto_prueba = 150
image_path_prueba = img_path_format.format(indice_foto_prueba, 0)
foto = cv2.imread(image_path_prueba)
dir_imgs = os.path.dirname(image_path_prueba)
dir_parent = os.path.dirname(dir_imgs)


inicio_x = 200
inicio_y = 470
gap_y = 50
ancho = 350
alto = 40
recorte1 = foto[inicio_y : inicio_y + alto, inicio_x : inicio_x + ancho, 0]
recorte2 = foto[
    inicio_y + gap_y + alto : inicio_y + gap_y + 2 * alto,
    inicio_x : inicio_x + ancho,
    0,
]

if show_ubicar_pico:
    plt.imshow(foto)
    plt.axhline(inicio_y + alto, color="r")
    plt.axhline(inicio_y + alto + gap_y, color="r")
    plt.axvline(inicio_x)
    plt.show()

sig = np.sum(recorte1, axis=0)
sig = sig - np.mean(sig)
if show_ubicar_pico:
    plt.plot(sig)
    plt.show()

fsig = np.fft.fft(sig)
freqsig = np.fft.fftfreq(len(sig), d=1)
nsig = len(np.abs(fsig))

indsig = np.argmax(np.abs(fsig)[nsig // 20 : nsig // 4]) + nsig // 20
print("descartamos el pico de frecuencias mayores a 1/4 de la de muestreo")
if show_ubicar_pico:
    plt.scatter(freqsig[indsig], np.abs(fsig)[indsig], color="r")
    plt.plot(freqsig[: nsig // 2], np.abs(fsig)[: nsig // 2])
    plt.axvline(freqsig[nsig // 20])
    plt.axvline(freqsig[nsig // 7])
    print(freqsig[indsig])
    plt.show()

fas1, fas2 = fase_hilbert(recorte1, recorte2, freqsig[indsig])
print(freqsig[indsig])
print(fas1, fas2)
# *******************************************************************
analizar = False

if analizar:
    fases1 = []
    fases2 = []
    for i in np.arange(0, 255, 5):
        this_image_path = img_path_format.format(i)
        imag = cv2.imread(this_image_path)
        recorte1 = imag[inicio_y : inicio_y + alto, inicio_x : inicio_x + ancho, 0]
        recorte2 = imag[
            inicio_y + gap_y + alto : inicio_y + gap_y + 2 * alto,
            inicio_x : inicio_x + ancho,
            0,
        ]

        fase1, fase2 = fase_hilbert(recorte1, recorte2, freqsig[indsig])
        fases1.append(fase1)
        fases2.append(fase2)
    fases1 = np.array(fases1)
    fases2 = np.array(fases2)

    exp_1 = np.exp(1j * fases1)
    exp_2 = np.exp(1j * fases2)
    dif = np.unwrap(np.angle(exp_1 / exp_2))

    datos = pd.DataFrame(np.unwrap(dif))
    datos.to_csv(
        "Calibracion_SLM/data/fase_medida_corregida_1809_1.csv",
        header=None,
        index=False,
    )
    max_int = 100
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(
        np.arange(0, 255, 5)[:max_int],
        dif[:max_int] - min(dif[:max_int]),
        label=f"{np.max(dif[:max_int]-min(dif[:max_int]))}",
    )
    ax[0].set_xlabel("Valor de gris")
    ax[0].set_ylabel("Fase [rad]")
    ax[0].axhline(2 * np.pi)
    ax[0].legend()
    ax[1].scatter(np.arange(0, 255, 5)[:-1], np.diff(dif))
    plot_path = os.path.join(dir_parent, "fasevsgris.pdf")
    plt.savefig(plot_path)
    plt.show()

chequeo_imgs_picos = False
if chequeo_imgs_picos:
    for i in np.arange(256):
        for j in range(10):
            foto = cv2.imread(
                f"data/repetibilidad/1 0.69-1.42/imgs/3009_I{i}_T21_{j}.png"
            )
            recorte1 = foto[inicio_y : inicio_y + alto, inicio_x : inicio_x + ancho, 0]
            recorte2 = foto[
                inicio_y + gap_y + alto : inicio_y + gap_y + 2 * alto,
                inicio_x : inicio_x + ancho,
                0,
            ]
            sig = np.sum(recorte1, axis=0)
            sig = sig - np.mean(sig)
            if j == 0:
                sig_0 = sig
            fsig = np.fft.fft(sig)
            freqsig = np.fft.fftfreq(len(sig), d=1)
            nsig = len(np.abs(fsig))
            indsig = np.argmax(np.abs(fsig)[nsig // 20 : nsig // 4]) + nsig // 20
            plt.scatter(
                freqsig[indsig],
                np.abs(fsig)[indsig],
                color="r",
                label=f"{i} {j}",
            )
            plt.plot(freqsig[: nsig // 2], np.abs(fsig)[: nsig // 2])
            plt.ylim(0, 25000)
            # plt.plot(sig, label=f"{i} {j}")
            plt.legend()
            print(np.max(np.abs(sig - sig_0)))
            plt.pause(0.05)
            plt.cla()


# ********************************
filtrado = False
cant_muestras = 10
j_img_buena = {}
if filtrado:
    for i in range(256):
        j_img_buena[i] = []
        for j in range(cant_muestras):
            print(i, j)
            foto = cv2.imread(
                f"data/seleccion_de_tensiones/24-c5 0.69-1.42/ims_gamma_lineal_pasos1/2209_I{i}_T21_{j}.png"
            )
            recorte1 = foto[inicio_y : inicio_y + alto, inicio_x : inicio_x + ancho, 0]
            recorte2 = foto[
                inicio_y + gap_y + alto : inicio_y + gap_y + 2 * alto,
                inicio_x : inicio_x + ancho,
                0,
            ]
            sig = np.sum(recorte1, axis=0)
            sig = sig - np.mean(sig)
            fsig = np.fft.fft(sig)
            freqsig = np.fft.fftfreq(len(sig), d=1)
            nsig = len(np.abs(fsig))
            fsig_normalized = np.abs(fsig) / np.max(np.abs(fsig))
            picos, props = find_peaks(
                fsig_normalized[0 : nsig // 4], height=0.5, prominence=0.3
            )
            indsig = np.argmax(np.abs(fsig)[nsig // 20 : nsig // 4]) + nsig // 20
            valor_pico1 = freqsig[indsig]

            if len(picos) == 1:
                prominences = props["prominences"]
                if prominences[0] > 0.5:
                    j_img_buena[i].append(j)

            # #plt.plot(
            #     freqsig[0 : nsig // 2],
            #     np.abs(fsig)[0 : nsig // 2],
            #     label=f"gris{i}muestra{j}",
            # )
            # #plt.scatter(freqsig[picos], np.abs(fsig)[picos])
            # #plt.ylim(0, 150000)
            # #plt.legend()
            # plt.pause(0.1)
            # plt.cla()

# # print(j_img_buena)
# foto_prueba = cv2.imread(f"data/repetibilidad/1 0.69-1.42/imgs/3009_I{10}_T21_{8}.png")
# recorte1 = foto_prueba[inicio_y : inicio_y + alto, inicio_x : inicio_x + ancho, 0]
# recorte2 = foto_prueba[
#     inicio_y + gap_y + alto : inicio_y + gap_y + 2 * alto,
#     inicio_x : inicio_x + ancho,
#     0,
# ]
# sig = np.sum(recorte1, axis=0)
# sig = sig - np.mean(sig)
# fsig = np.fft.fft(sig)
# freqsig = np.fft.fftfreq(len(sig), d=1)
# nsig = len(np.abs(fsig))
# fsig_normalized = np.abs(fsig) / np.max(np.abs(fsig))
# tope = nsig // 7
# picos, props = find_peaks(
#     fsig_normalized[nsig // 20 : tope], height=0.5, prominence=0.5
# )
# picos += nsig // 20
# print(props)
# plt.plot(freqsig[0 : nsig // 2], np.abs(fsig)[0 : nsig // 2])
# plt.scatter(freqsig[picos], np.abs(fsig)[picos])
# plt.axvline(freqsig[tope])
# plt.ylim(0, 20000)
# plt.show()
# indsig = np.argmax(np.abs(fsig)[nsig // 20 : nsig // 4]) + nsig // 20
# valor_pico1 = freqsig[indsig]


# *****************************
new_analisis = False
cant_muestras = 10


if new_analisis:
    exp_1 = []
    exp_2 = []
    indices = []
    for i in np.arange(256):
        fases_1j = []
        fases_2j = []
        for j in range(cant_muestras):
            print(i, j)
            this_image_path = img_path_format.format(i, j)
            imag = cv2.imread(this_image_path)
            recorte1 = imag[inicio_y : inicio_y + alto, inicio_x : inicio_x + ancho, 0]
            recorte2 = imag[
                inicio_y + gap_y + alto : inicio_y + gap_y + 2 * alto,
                inicio_x : inicio_x + ancho,
                0,
            ]

            sig = np.sum(recorte1, axis=0)
            sig = sig - np.mean(sig)
            fsig = np.fft.fft(sig)
            freqsig = np.fft.fftfreq(len(sig), d=1)
            nsig = len(np.abs(fsig))
            fsig_normalized = np.abs(fsig) / np.max(np.abs(fsig))
            picos, props = find_peaks(
                fsig_normalized[nsig // 20 : nsig // 7],
                height=0.5,
                prominence=0.3,
                distance=50,
            )
            indsig = np.argmax(np.abs(fsig)[nsig // 20 : nsig // 7]) + nsig // 20
            valor_pico1 = freqsig[indsig]

            if len(picos) == 1:
                prominences = props["prominences"]
                if prominences[0] > 0.5:
                    fase1j, fase2j = fase_hilbert(recorte1, recorte2, freqsig[indsig])
                    print(fase1j, fase2j)
                    fases_1j.append(fase1j)
                    fases_2j.append(fase2j)
            else:
                print("no hay un solo pico", len(picos))
        if fases_1j:
            fases_1j = np.array(fases_1j)
            fases_2j = np.array(fases_2j)

            exp_1j = np.mean(np.exp(1j * fases_1j))
            exp_2j = np.mean(np.exp(1j * fases_2j))
            exp_1.append(exp_1j)
            exp_2.append(exp_2j)
            indices.append(i)
    exp_1 = np.array(exp_1)
    exp_2 = np.array(exp_2)
    dif = np.unwrap(np.angle(exp_2 / exp_1))

    # datos = pd.DataFrame(np.unwrap(dif))
    # datos.to_csv(
    #     "Calibracion_SLM/data/fase_medida_corregida_1809_4.csv",
    #     header=None,
    #     index=False,
    # )

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(
        indices,
        dif - min(dif),
        label=f"{np.max(dif-min(dif))}",
    )
    ax[0].set_xlabel("Valor de gris")
    ax[0].set_ylabel("Fase [rad]")
    ax[0].axhline(2 * np.pi)
    ax[0].legend()
    print(indices)

    # ax[1].scatter(np.arange(256)[:-1], np.diff(dif))
    plot_path = os.path.join(dir_parent, "fasevsgris.pdf")
    # plt.savefig(plot_path)
    plt.show()
    # repe = pd.DataFrame(data=(indices, dif))
    # df = repe.transpose()
    # df.to_csv("data/repetibilidad/13 0.69-1.42/fasevsgris_13.csv")
    print(df)

cali = np.zeros((256, 11))
counter = 0
for i in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12]:  # 1 tiene 255 pero salio bien, 3 midio mal
    dato = pd.read_csv(f"data/repetibilidad/{i} 0.69-1.42/fasevsgris_{i}.csv")
    dato = dato.iloc[:, 1:]
    cali[:, counter] = dato["1"] - min(dato["1"])
    print(len(dato["0"]))
    plt.errorbar(
        dato["0"], dato["1"] - min(dato["1"]), fmt="o", markersize=2, alpha=0.6
    )
    counter += 1
plt.axhline(2 * np.pi)
plt.ylabel("Fase [rad]")
plt.xlabel("Valor de gris")
# plt.savefig("data/repetibilidad/fasevsgris_todas.png")
plt.show()

cali_mean = np.mean(cali, axis=1)
cali_std = np.std(cali, axis=1)
plt.errorbar(
    np.arange(256),
    cali_mean,
    yerr=cali_std,
    fmt="o",
    markersize=2,
    alpha=0.6,
)
plt.axhline(2 * np.pi)
plt.ylabel("Fase [rad]")
plt.xlabel("Valor de gris")
plt.show()
