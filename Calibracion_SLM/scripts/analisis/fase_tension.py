import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re
from datetime import datetime
from rotar_y_crop_img import rotate_bound
from scipy.signal import hilbert
from scipy.optimize import curve_fit
import cv2
from scipy.signal import find_peaks


dir_path = r'Calibracion_SLM\data\fase_tension_4.32-1.48'

file_path = r'Calibracion_SLM\data\fase_tension_4.32-0.82\1306_I150_T21.pkl'
with open(file_path, mode = 'rb') as f:
    imagen = pickle.load(f)
imagen = rotate_bound(imagen, 2)
plt.imshow(imagen)
plt.show()
prueba = False
if prueba:
    with open(file_path, mode = 'rb') as f:
        imagen = pickle.load(f)

    imagen = rotate_bound(imagen, 2)

    altura_rec = 30
    ancho_rec = 300
    x_rec1 = 490
    y_rec1 = 440
    x_rec2 = 490
    y_rec2 = 510
    recorte1 = imagen[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec]
    recorte2 = imagen[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec]
    fig, ax = plt.subplots()
    ax.imshow(imagen)
    rect1 = plt.Rectangle((x_rec1,y_rec1), ancho_rec, altura_rec, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    rect2 = plt.Rectangle((x_rec2,y_rec2), ancho_rec, altura_rec, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    plt.show()
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ax[0].imshow(recorte1, cmap='gray')
    ax[0].set_title('Recorte 1')
    ax[0].set_xlabel('Pixel')
    ax[1].imshow(recorte2, cmap='gray')
    ax[1].set_title('Recorte 2')
    ax[1].set_xlabel('Pixel')
    plt.tight_layout()
    plt.show()

    plt.plot(np.sum(recorte1, axis=0), label='Recorte 1')
    #plt.plot(np.sum(recorte2, axis=0), label='Recorte 2')
    plt.legend()
    plt.title('Señal de un recorte')
    plt.xlabel('Pixel')
    plt.ylabel('Intensidad')
    plt.show()

    signal1 = np.sum(recorte1, axis=0)
    signal2 = np.sum(recorte2, axis=0)

    signal1 = signal1 - np.mean(signal1)
    signal2 = signal2 - np.mean(signal2)
    
    f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
    f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
    n = len(f_signal1)
    freq1_ind = np.argmax(np.abs(f_signal1[:n//2]))
    freq2_ind = np.argmax(np.abs(f_signal2[:n//2]))
    frecuencias = np.fft.fftfreq(len(signal1), d=1)
    #freq1 = frecuencias[freq1_ind]
    #freq2 = frecuencias[freq2_ind]

    p1, _ = find_peaks(np.abs(f_signal1)[:n//2], prominence = 0.3*np.max(np.abs(f_signal1)))
    p2, _ = find_peaks(np.abs(f_signal2)[:n//2], prominence = 0.3*np.max(np.abs(f_signal2)))
    plt.plot(frecuencias[:n//2], np.abs(f_signal1)[:n//2])
    plt.scatter(frecuencias[p1], np.abs(f_signal1)[p1], color ='r')
    plt.axhline(0.5*np.max(np.abs(f_signal1)))
    plt.show()
    freq1 = np.max(frecuencias[p1])
    freq2 = np.max(frecuencias[p2])
    f_bandwidth = 0.01
    gauss_mask1 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth)**2) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth)**2)
    gauss_mask2 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth)**2) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth)**2)
    fft_filtrada1 = f_signal1 * gauss_mask1
    fft_filtrada2 = f_signal2 * gauss_mask2
    filtra1 = np.fft.ifft(fft_filtrada1).real
    filtra2 = np.fft.ifft(fft_filtrada2).real


    plt.plot(frecuencias, gauss_mask1)
    plt.plot(frecuencias, np.abs(f_signal1)/np.max(np.abs(f_signal1)))
    plt.show()
    plt.plot(frecuencias, gauss_mask2)
    plt.plot(frecuencias, np.abs(f_signal2)/np.max(np.abs(f_signal2)))
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
    for i in np.arange(0,255,5):
        file_path = os.path.join(dir_path, f'1306_I{i}_T21.pkl')
        if not os.path.exists(file_path):
            continue
        with open(file_path, mode = 'rb') as f:
            imag = pickle.load(f)
        imag = rotate_bound(imag, 2)
        plt.imshow(imag)
        plt.pause(.05)
        plt.cla()
run = True
if run:
    def lineal(x, m, c):
        return m*x+c
    c1 = []
    c2 = []
    m1 = []
    m2 = []
    for i in np.arange(0,255,5):
        file_path = os.path.join(dir_path, f'1306_I{i}_T21.pkl')
        if not os.path.exists(file_path):
            continue
        with open(file_path, mode = 'rb') as f:
            imag = pickle.load(f)
        imag = rotate_bound(imag, 2)
        altura_rec = 30
        ancho_rec = 300
        x_rec1 = 490
        y_rec1 = 440
        x_rec2 = 490
        y_rec2 = 510
        recorte1 = imag[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec]
        recorte2 = imag[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec]

        signal1 = np.sum(recorte1, axis=0)
        signal2 = np.sum(recorte2, axis=0)

        signal1 = signal1 - np.mean(signal1)
        signal2 = signal2 - np.mean(signal2)
        

        f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
        f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
        n = len(f_signal1)
        freq1_ind = np.argmax(np.abs(f_signal1[:n//2]))
        freq2_ind = np.argmax(np.abs(f_signal2[:n//2]))
        frecuencias = np.fft.fftfreq(len(signal1), d=1)
        #freq1 = frecuencias[freq1_ind]
        #freq2 = frecuencias[freq2_ind]
        p1, _ = find_peaks(np.abs(f_signal1)[:n//2], prominence = 0.25*np.max(np.abs(f_signal1)))
        p2, _ = find_peaks(np.abs(f_signal2)[:n//2], prominence = 0.25*np.max(np.abs(f_signal2)))

        plt.plot(frecuencias[:n//2], np.abs(f_signal1)[:n//2])
        plt.scatter(frecuencias[p1], np.abs(f_signal1)[p1], color ='r')
        plt.title(f'I = {i}. 1: {len(p1)}, 2: {len(p2)}')
        plt.pause(.1)
        plt.cla()
    
        #freq1 = np.max(frecuencias[p1])
        #freq2 = np.max(frecuencias[p2])
        #freq1 = frecuencias[p1[-2]]
        #freq2 = frecuencias[p2[-2]]
        #print(freq1)
        freq1 = 0.08666666666666667
        freq2 = freq1
        f_bandwidth = 0.01
        gauss_mask1 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth)**2) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth)**2)
        gauss_mask2 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth)**2) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth)**2)
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

    
        c1.append(popt1[1]) #guardo los valores de la ordenada al origen
        c2.append(popt2[1]) #guardo los valores de la ordenada al origen
        m1.append(popt1[0])
        m2.append(popt2[0])

        print(i)


    plt.plot(m1)
    plt.plot(m2)
    plt.show()
    c1 = np.array(c1)
    c2 = np.array(c2)
    exp_c1 = np.exp(1j*c1)
    exp_c2 = np.exp(1j*c2)
    plt.scatter(np.arange(0,255,5), c1, label='Ordenada al origen 1')
    plt.show()
    plt.scatter(np.arange(0,255,5), c2, label='Ordenada al origen 2')
    plt.show()
    fase1 = np.unwrap(c1)
    fase2 = np.unwrap(c2) 
    diferencia = fase1 - fase2
    plt.plot(diferencia)
    plt.show()
    dif_compleja = np.angle(exp_c1/exp_c2)
    plt.plot(dif_compleja)
    plt.show()
    plt.plot(np.unwrap(dif_compleja))
    plt.show()