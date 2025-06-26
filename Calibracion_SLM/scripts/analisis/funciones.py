import numpy as np
import cv2
from scipy.signal import hilbert
from scipy.optimize import curve_fit

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

def fase_hilbert(recorte1, recorte2, filter_frec, filter_band = 0.01):
    signal1 = np.sum(recorte1, axis=0)
    signal2 = np.sum(recorte2, axis=0)
    signal1 = signal1 - np.mean(signal1)
    signal2 = signal2 - np.mean(signal2)

    f_signal1 = np.fft.fft(signal1)
    f_signal2 = np.fft.fft(signal2)
    frecuencias = np.fft.fftfreq(len(signal1), d=1)
    gauss_mask = np.exp(-0.5 * ((frecuencias - filter_frec) / filter_band)**2) + np.exp(-0.5 * ((frecuencias + filter_frec) / filter_band)**2)
    fft_filtrada1 = f_signal1 * gauss_mask
    fft_filtrada2 = f_signal2 * gauss_mask
    filtra1 = np.fft.ifft(fft_filtrada1).real
    filtra2 = np.fft.ifft(fft_filtrada2).real

    hil1 = hilbert(filtra1)
    hil2 = hilbert(filtra2)
    lin1 = np.unwrap(np.angle(hil1))
    lin2 = np.unwrap(np.angle(hil2))
    inf = int(round(0.1*len(lin1)))
    sup = int(round(0.9*len(lin1)))
    lin1_ajust = lin1[inf:sup]
    lin2_ajust = lin2[inf:sup]
    popt1, pcov1 = curve_fit(lambda x, m, c: m*x+c, np.arange(inf, sup), lin1_ajust)
    popt2, pcov2 = curve_fit(lambda x, m, c: m*x+c, np.arange(inf, sup), lin2_ajust)

    return popt1[1], popt2[1]
