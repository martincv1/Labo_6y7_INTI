import numpy as np
import cv2
from scipy.signal import hilbert, find_peaks
from scipy.optimize import curve_fit
import funciones
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

altura_rec = 20
ancho_rec = 200
x_rec1 = 535
y_rec1 = 435
x_rec2 = 535
y_rec2 = 510

dir_path = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo_corto0702'
with open(f"{dir_path}/0207_12-05-03-468_I230_202_T20.pkl", 'rb') as f:
    imag = pickle.load(f)  # Esto cargará el objeto Python que fue guardado
imag_rot = funciones.rotate_bound(imag, 1.7)
graficar = True                

recorte1 = imag_rot[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec]
recorte2 = imag_rot[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec]
if graficar:
    plt.imshow(imag)
    plt.show()
    plt.imshow(imag_rot)
    plt.show()
    fig, ax = plt.subplots()
    ax.imshow(imag_rot)
    rect1 = patches.Rectangle((x_rec1,y_rec1), ancho_rec, altura_rec, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    rect2 = patches.Rectangle((x_rec2,y_rec2), ancho_rec, altura_rec, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    plt.show()

    plt.imshow(recorte1)
    plt.show()
    plt.imshow(recorte2)
    plt.show()
buscar_freq = True
if buscar_freq:
        
    signal1 = np.sum(recorte1, axis=0)
    signal2 = np.sum(recorte2, axis=0)

    fft1 = np.fft.fft(signal1 - np.mean(signal1))
    fft2 = np.fft.fft(signal2 - np.mean(signal2))
    frecuencias1 = np.fft.fftfreq(len(signal1), d=1)
    frecuencias2 = np.fft.fftfreq(len(signal2), d=1)
    plt.plot(frecuencias1, np.abs(fft1))
    plt.xlabel("Freq [Hz]")
    plt.show()
    plt.plot(frecuencias2, np.abs(fft2))
    plt.xlabel("Freq [Hz]")
    plt.show()

    idx_pico = np.argmax(np.abs(fft1[:len(signal1)//2]))
    filter_frec = frecuencias1[idx_pico]
    print("Frecuencia dominante:", filter_frec)