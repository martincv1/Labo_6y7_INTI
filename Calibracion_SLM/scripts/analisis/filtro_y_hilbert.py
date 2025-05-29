import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, firwin, filtfilt, hilbert

imag = cv2.imread(f'../../../../fotos_rot_franjas/3004_I7_T22_r_p_final.png')
recorte1 = imag[495:555, 170:570, 0]
recorte2 = imag[690:750, 350:750, 0]

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

numtaps = 101  # cantidad de coeficientes, más alto = más selectivo
b1 = firwin(numtaps, [low1, high1], pass_zero=False)
b2 = firwin(numtaps, [low2, high2], pass_zero=False)

filtra1 = filtfilt(b1, [1.0], signal1)
filtra2 = filtfilt(b2, [1.0], signal2)
#plt.plot(signal1)
#plt.plot(filtra1)
plt.plot(signal2- np.mean(signal2))
plt.plot(filtra2)
plt.show()

hil1 = hilbert(filtra1)
hil2 = hilbert(filtra2)
plt.plot(np.unwrap(np.angle(hil1/np.abs(hil1))))
plt.plot(np.unwrap(np.angle(hil2/np.abs(hil2))))
plt.show()
