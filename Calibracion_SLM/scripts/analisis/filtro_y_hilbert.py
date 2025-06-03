import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import butter, firwin, filtfilt, hilbert
from scipy.optimize import curve_fit

probar = False
if probar:
    imag = cv2.imread(f'../../../../fotos_rot_franjas/3004_I30_T22_r_p_final.png')

    altura_rec = 20
    ancho_rec = 100
    x_rec1 = 490
    y_rec1 = 520
    x_rec2 = 490
    y_rec2 = 600
    recorte1 = imag[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec, 0]
    recorte2 = imag[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec, 0]
    fig, ax = plt.subplots()
    ax.imshow(imag)
    rect1 = patches.Rectangle((x_rec1,y_rec1), ancho_rec, altura_rec, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    rect2 = patches.Rectangle((x_rec2,y_rec2), ancho_rec, altura_rec, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    plt.show()

    plt.imshow(recorte1)
    plt.show()
    plt.imshow(recorte2)
    plt.show()

    signal1 = np.sum(recorte1, axis=0)
    signal2 = np.sum(recorte2, axis=0)
    #signal1 = imag[y_rec1, x_rec1:x_rec1+ancho_rec, 0]
    #signal2 = imag[y_rec2, x_rec2:x_rec2+ancho_rec, 0]
    plt.plot(signal1)
    plt.plot(signal2)
    plt.show()
    f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
    f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
    n = len(f_signal1)
    freq1_ind = np.argmax(np.abs(f_signal1[:n//2]))
    freq2_ind = np.argmax(np.abs(f_signal2[:n//2]))

    frecuencias = np.fft.fftfreq(len(signal1), d=1)
    freq1 = frecuencias[freq1_ind]
    freq2 = frecuencias[freq2_ind]

    fs = 1.0  # muestras por píxel
    bandwidth = 0.005  # ancho de banda en ciclos/píxel, ajustable
    print(freq1)
    lowcut1 = freq1 - bandwidth
    highcut1 = freq1 + bandwidth

    lowcut2 = freq2 - bandwidth
    highcut2 = freq2 + bandwidth

    # Diseño del filtro Butterworth
    order = 5
    nyq = 0.5 * fs
    low1 = lowcut1 / nyq
    high1 = highcut1 / nyq
    low2 = lowcut2 / nyq
    high2 = highcut2 / nyq

    #b, a = butter(order, [low, high], btype='bandpass')

    numtaps = 20  # cantidad de coeficientes, más alto = más selectivo
    b1 = firwin(numtaps, [low1, high1], pass_zero=False)
    b2 = firwin(numtaps, [low2, high2], pass_zero=False)

    f_b1 = np.fft.fft(b1, len(signal1))
    plt.plot(np.abs(f_b1)/np.max((np.abs(f_b1))))
    plt.plot(np.abs(f_signal1)/np.max((np.abs(f_signal1))))
    plt.show()

    filtra1 = filtfilt(b1, [1.0], signal1-np.mean(signal1)) 
    filtra2 = filtfilt(b2, [1.0], signal2-np.mean(signal2))
    
    #plt.plot(signal1)
    #plt.plot(filtra1)
    plt.plot(signal2- np.mean(signal2))
    plt.plot(filtra2)
    plt.show()

    hil1 = hilbert(filtra1)
    hil2 = hilbert(filtra2)
    lin1 = np.unwrap(np.angle(hil1))
    lin2 = np.unwrap(np.angle(hil2))
    inf = 30
    sup = 70
    plt.plot(lin1)
    plt.plot(lin2)
    plt.show()
    plt.plot(np.diff(lin1))
    plt.plot(np.diff(lin2))
    plt.show()
    def lineal(x, m, c):
        return m*x+c
    lin1_ajust = lin1[inf:sup]
    lin2_ajust = lin2[inf:sup]
    popt1, pcov1 = curve_fit(lineal, np.arange(30,70), lin1_ajust)
    popt2, pcov2 = curve_fit(lineal, np.arange(30,70), lin2_ajust)
    plt.plot(np.arange(30,70), lin1_ajust)
    plt.plot(np.arange(30,70), lineal(np.arange(30,70), *popt1), color = 'r', linestyle = '--')
    plt.show()
    print(popt1[0], popt2[0])
    print(abs(popt1[1]-popt2[1]))
else:
    def lineal(x, m, c):
        return m*x+c
    dif_fase = np.zeros(256)
    c1 = np.zeros(256)
    c2 = np.zeros(256)
    for i in range(256):
        imag = cv2.imread(f'../../../../fotos_rot_franjas/3004_I{i}_T22_r_p_final.png')
        altura_rec = 20
        ancho_rec = 100
        x_rec1 = 490
        y_rec1 = 520
        x_rec2 = 490
        y_rec2 = 600
        recorte1 = imag[y_rec1:y_rec1+altura_rec, x_rec1:x_rec1+ancho_rec, 0]
        recorte2 = imag[y_rec2:y_rec2+altura_rec, x_rec2:x_rec2+ancho_rec, 0]

        signal1 = np.sum(recorte1, axis=0)
        signal2 = np.sum(recorte2, axis=0)

        f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
        f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
        n = len(f_signal1)
        freq1_ind = np.argmax(np.abs(f_signal1[:n//2]))
        freq2_ind = np.argmax(np.abs(f_signal2[:n//2]))
        frecuencias = np.fft.fftfreq(len(signal1), d=1)
        freq1 = frecuencias[freq1_ind]
        freq2 = frecuencias[freq2_ind]

        fs = 1.0  # muestras por píxel
        bandwidth = 0.005  # ancho de banda en ciclos/píxel, ajustable
        lowcut1 = freq1 - bandwidth
        highcut1 = freq1 + bandwidth

        lowcut2 = freq2 - bandwidth
        highcut2 = freq2 + bandwidth

        
        order = 5
        nyq = 0.5 * fs
        low1 = lowcut1 / nyq
        high1 = highcut1 / nyq
        low2 = lowcut2 / nyq
        high2 = highcut2 / nyq

        #b, a = butter(order, [low, high], btype='bandpass')

        numtaps = 20  # cantidad de coeficientes, más alto = más selectivo
        b1 = firwin(numtaps, [low1, high1], pass_zero=False)
        b2 = firwin(numtaps, [low2, high2], pass_zero=False)

        filtra1 = filtfilt(b1, [1.0], signal1)
        filtra2 = filtfilt(b2, [1.0], signal2)
        hil1 = hilbert(filtra1)
        hil2 = hilbert(filtra2)
        lin1 = np.unwrap(np.angle(hil1))
        lin2 = np.unwrap(np.angle(hil2))
        inf = 30
        sup = 70
        lin1_ajust = lin1[inf:sup]
        lin2_ajust = lin2[inf:sup]
        popt1, pcov1 = curve_fit(lineal, np.arange(inf,sup), lin1_ajust)
        popt2, pcov2 = curve_fit(lineal, np.arange(inf,sup), lin2_ajust)
    
        c1[i] = popt1[1] #guardo los valores de la ordenada al origen
        c2[i] = popt2[1]
        print(i)
        #plt.title(f'Intesidad = {i}')
        #plt.plot(np.arange(30,70), lin1_ajust)
        #plt.plot(np.arange(30,70), lin2_ajust)
        #plt.pause(.001)
        #plt.cla()
    plt.scatter(np.arange(256), np.unwrap(c1))
    plt.scatter(np.arange(256), np.unwrap(c2))
    plt.show()
    fase1 = np.unwrap(c1) - np.unwrap(c1)[0] #hago unwrap ahora que estan entre -pi y pi y resto 
    fase2 = np.unwrap(c2) - np.unwrap(c2)[0] # el valor inicial para que empiecen en cero
    diferencia = fase1 - fase2
    plt.plot(diferencia)
    plt.show()