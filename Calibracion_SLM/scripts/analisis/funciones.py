import numpy as np
import cv2
from scipy.signal import hilbert, find_peaks
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

def simular_imagen(Nx=640, Ny=512, angulo_slm_max=1, slm_ancho=500, slm_alto=350, angulo_franjas_max=5,
                   fase1=0, fase2=np.pi, frecuencia=10, fotones_por_cuenta=5, amplitud_imperfecciones=0.25):
    """
    Simula una imagen con un SLM (Spatial Light Modulator) rotados aleatoriamente y con franjas de interferencia con
    fase distinta para las mitades superior e inferior.

    Parámetros:
    Nx (int): Ancho de la imagen en píxeles. Default es 640.
    Ny (int): Altura de la imagen en píxeles. Default es 512.
    angulo_slm_max (float): Ángulo máximo de rotación del SLM en grados. Default es 1.
    slm_ancho (int): Ancho del SLM en píxeles. Default es 500.
    slm_alto (int): Altura del SLM en píxeles. Default es 350.
    angulo_franjas_max (float): Ángulo máximo de rotación de las franjas en grados. Default es 5.
    fase1 (float): Fase inicial para la primera mitad del SLM. Default es 0.
    fase2 (float): Fase inicial para la segunda mitad del SLM. Default es pi.
    frecuencia (float): Frecuencia espacial de las franjas en el SLM. Debe ser un valor positivo que represente el
                        número de ciclos que entra en un ancho de imagen. Default es 10.
    fotones_por_cuenta (int): Número de fotones que se generan por cada punto del SLM. Default es 5.
    amplitud_imperfecciones (float): Amplitud de las imperfecciones aleatorias en la imagen. Default es 0.25.

    Retorna:
    numpy.ndarray: Imagen simulada como un array de 2D numpy de tipo uint8.

    Lanza:
    ValueError: Si el SLM rotado no entra dentro de la imagen.
    """

    theta = np.radians(np.random.uniform(-angulo_slm_max, angulo_slm_max))
    theta_franjas = np.radians(90 + np.random.uniform(-angulo_franjas_max, angulo_franjas_max))
    rotacion = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = np.arange(Nx)
    y = np.arange(Ny)
    xy = np.meshgrid(x, y)
    xy = np.dot(np.column_stack((xy[0].flatten(), xy[1].flatten())), rotacion)
    x_rot = xy[:, 0].reshape((Ny, Nx))
    y_rot = xy[:, 1].reshape((Ny, Nx))
    x_izquierda = Nx // 2 - slm_ancho // 2
    x_derecha = Nx // 2 + slm_ancho // 2
    slm1 = np.logical_and(x_rot > x_izquierda, x_rot < x_derecha)
    slm1 = np.logical_and(slm1, y_rot > Ny // 2)
    slm1 = np.logical_and(slm1, y_rot < Ny // 2 + slm_alto // 2)
    slm2 = np.logical_and(x_rot > x_izquierda, x_rot < x_derecha)
    slm2 = np.logical_and(slm2, y_rot > Ny // 2 - slm_alto // 2)
    slm2 = np.logical_and(slm2, y_rot <= Ny // 2)
    # Calcular los valores de x e y del rectangulo girado
    center = np.array([Nx // 2, Ny // 2])
    esquina_1 = np.dot(rotacion, np.array([-slm_ancho // 2, -slm_alto // 2]) + center)
    esquina_2 = np.dot(rotacion, np.array([slm_ancho // 2, -slm_alto // 2]) + center)
    esquina_3 = np.dot(rotacion, np.array([slm_ancho // 2, slm_alto // 2]) + center)
    esquina_4 = np.dot(rotacion, np.array([-slm_ancho // 2, slm_alto // 2]) + center)
    xmin = np.min([esquina_1[0], esquina_2[0], esquina_3[0], esquina_4[0]])
    xmax = np.max([esquina_1[0], esquina_2[0], esquina_3[0], esquina_4[0]])
    ymin = np.min([esquina_1[1], esquina_2[1], esquina_3[1], esquina_4[1]])
    ymax = np.max([esquina_1[1], esquina_2[1], esquina_3[1], esquina_4[1]])
    if not (0 <= xmin < Nx and 0 <= xmax < Nx and 0 <= ymin < Ny and 0 <= ymax < Ny):
        raise ValueError('El slm girado no entra en la imagen')
    # Calcular fase de superficie imperfecta
    fase_imperfecta = gaussian_filter(np.random.randn(Ny, Nx), sigma=100, mode='reflect')
    fase_imperfecta = fase_imperfecta / np.max(fase_imperfecta) * amplitud_imperfecciones * 2 * np.pi

    # Generar la imagen
    imagen = np.zeros((Ny, Nx))
    imagen[slm1] = np.sin(2 * np.pi * frecuencia * (np.sin(theta_franjas) * x_rot[slm1] / Nx +
                                                    np.cos(theta_franjas) * y_rot[slm1] / Ny)
                          + fase1 + fase_imperfecta[slm1]) + 1
    imagen[slm2] = np.sin(2 * np.pi * frecuencia * (np.sin(theta_franjas) * x_rot[slm2] / Nx +
                                                    np.cos(theta_franjas) * y_rot[slm2] / Ny)
                          + fase2 + fase_imperfecta[slm2]) + 1
    imagen[slm1 | slm2] *= 120

    # Agregamos el contorno del SLM
    esquina_1 = np.round(esquina_1).astype(int)
    esquina_2 = np.round(esquina_2).astype(int)
    esquina_3 = np.round(esquina_3).astype(int)
    esquina_4 = np.round(esquina_4).astype(int)
    rr, cc = draw.polygon_perimeter([esquina_1[1], esquina_2[1], esquina_3[1], esquina_4[1]],
                                    [esquina_1[0], esquina_2[0], esquina_3[0], esquina_4[0]])
    imagen[rr, cc] = 255
    # Agregamos ruido de poisson
    imagen = np.random.poisson(imagen * fotones_por_cuenta) / fotones_por_cuenta
    imagen[imagen > 255] = 255
    imagen = imagen.astype(np.uint8)

    return imagen
