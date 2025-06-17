import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import butter, firwin, filtfilt, hilbert
from scipy.optimize import curve_fit
import pickle

def lineal(x, m, c):
    return m*x + c

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

dir_path = '/home/lorenzo/Labo_6y7_INTI/Calibracion_SLM/data/fase_tiempo'
with open(f"{dir_path}/1206_13-54-48_I40_T21.pkl", 'rb') as f:
    imag = pickle.load(f)  # Esto cargará el objeto Python que fue guardado
imag_rot = rotate_bound(imag, 4)


#plt.imshow(imag)
#plt.imshow(imag_rot)
#plt.show()
graficar = False
altura_rec = 20
ancho_rec = 100
x_rec1 = 635
y_rec1 = 470
x_rec2 = 635
y_rec2 = 540
recorte1 = imag_rot[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec]
recorte2 = imag_rot[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec]
if graficar:
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
probar = True
if probar:
    # Señales promedio por columna
    signal1 = np.sum(recorte1, axis=0)
    signal2 = np.sum(recorte2, axis=0)

    # FFT para encontrar la frecuencia dominante
    f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
    f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
    n = len(f_signal1)
    freq1_ind = np.argmax(np.abs(f_signal1[:n//2]))
    freq2_ind = np.argmax(np.abs(f_signal2[:n//2]))
    frecuencias = np.fft.fftfreq(len(signal1), d=1)
    freq1 = frecuencias[freq1_ind]
    freq2 = frecuencias[freq2_ind]

    # Parámetros del filtro
    fs = 1.0  # muestras por píxel
    bandwidth = 0.005  # ancho de banda en ciclos/píxel
    lowcut1 = freq1 - bandwidth
    highcut1 = freq1 + bandwidth
    lowcut2 = freq2 - bandwidth
    highcut2 = freq2 + bandwidth

    # Diseño del filtro FIR
    order = 5
    nyq = 0.5 * fs
    low1 = lowcut1 / nyq
    high1 = highcut1 / nyq
    low2 = lowcut2 / nyq
    high2 = highcut2 / nyq

    numtaps = 20
    b1 = firwin(numtaps, [low1, high1], pass_zero=False)
    b2 = firwin(numtaps, [low2, high2], pass_zero=False)

    # Filtrado de las señales
    filtra1 = filtfilt(b1, [1.0], signal1)
    filtra2 = filtfilt(b2, [1.0], signal2)

    # Transformada de Hilbert
    hil1 = hilbert(filtra1)
    hil2 = hilbert(filtra2)
    lin1 = np.unwrap(np.angle(hil1))
    lin2 = np.unwrap(np.angle(hil2))

    # Ajuste lineal
    inf = 30
    sup = 70
    lin1_ajust = lin1[inf:sup]
    lin2_ajust = lin2[inf:sup]
    popt1, pcov1 = curve_fit(lineal, np.arange(inf, sup), lin1_ajust)
    popt2, pcov2 = curve_fit(lineal, np.arange(inf, sup), lin2_ajust)

    # Resultados
    print(f"Pendiente región 1: {popt1[0]}, Ordenada 1: {popt1[1]}")
    print(f"Pendiente región 2: {popt2[0]}, Ordenada 2: {popt2[1]}")

    # Visualización
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(imag_rot)
    rect1 = patches.Rectangle((x_rec1, y_rec1), ancho_rec, altura_rec, linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((x_rec2, y_rec2), ancho_rec, altura_rec, linewidth=1, edgecolor='b', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    plt.title('Imagen con regiones de interés')

    plt.subplot(2, 2, 2)
    plt.plot(signal1, 'r-', label='Señal región 1')
    plt.plot(signal2, 'b-', label='Señal región 2')
    plt.legend()
    plt.title('Señales originales')

    plt.subplot(2, 2, 3)
    plt.plot(filtra1, 'r-', label='Filtrada 1')
    plt.plot(filtra2, 'b-', label='Filtrada 2')
    plt.legend()
    plt.title('Señales filtradas')

    plt.subplot(2, 2, 4)
    plt.plot(np.arange(inf, sup), lin1_ajust, 'r-', label='Fase 1')
    plt.plot(np.arange(inf, sup), lin2_ajust, 'b-', label='Fase 2')
    plt.plot(np.arange(inf, sup), lineal(np.arange(inf, sup), *popt1), 'r--')
    plt.plot(np.arange(inf, sup), lineal(np.arange(inf, sup), *popt2), 'b--')
    plt.legend()
    plt.title('Fases desenvueltas y ajuste lineal')

    plt.tight_layout()
    plt.show()
    diferencia_fase = popt1[1] - popt2[1]  # Ordenada al origen 1 - Ordenada al origen 2
    print(f"Diferencia de fase entre regiones: {diferencia_fase} radianes")