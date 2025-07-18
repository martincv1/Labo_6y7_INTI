import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import cv2
import interfranja as ifranja
import pandas as pd
from scipy.signal import butter, firwin, filtfilt, hilbert, lfilter
from scipy.optimize import curve_fit

MINIMUM_DISTANCE_PEAKS = 10
PROMINENCE_PEAKS = 1
SHOW_CROP1 = False
SHOW_INTERFRANJA = False


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


def get_shift_cross_correlation(imagen1, imagen2, padding=0):
    if padding > 0:
        imagen1 = np.pad(imagen1, padding)
        imagen2 = np.pad(imagen2, padding)
    cross_corr = np.real(np.fft.ifft2(np.fft.fft2(imagen1) * np.conj(np.fft.fft2(imagen2))))
    cross_corr = np.fft.fftshift(cross_corr)
    indices = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    shift = indices[0] - cross_corr.shape[0] // 2, indices[1] - cross_corr.shape[1] // 2
    return shift


def get_dphi_fft1d(imagen1, imagen2, get_shift=False):
    signal1 = np.sum(imagen1, axis=0)
    signal2 = np.sum(imagen2, axis=0)
    f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
    f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
    n = len(f_signal1)
    freq1 = np.argmax(np.abs(f_signal1[:n//2]))
    freq2 = np.argmax(np.abs(f_signal2[:n//2]))
    if abs(freq1-freq2)>2:
        print(freq1, freq2)
        raise ValueError("Las franjas no tienen la misma frecuencia espacial")
    dfase = np.angle(f_signal1[freq1] / f_signal2[freq2])

    if get_shift:
        N1 = len(signal1)
        shift = dfase / (2 * np.pi * freq1) * N1
        return shift
    return dfase


def get_interfranja(imagen, remove_Gaussian_profile=False, show=False):
    # Rotar la imagen para dejar las franjas más o menos verticales
    img_rotada = ifranja.rotate_image_to_max_frequency(imagen)

    intensity_profile = np.mean(img_rotada, axis=0)
    if remove_Gaussian_profile:
        intensity_profile = ifranja.remove_gaussian_from_curve(intensity_profile)

    if show:
        # Mostrar la imagen original y la rotada
        fig, axs = plt.subplots(1, 3, figsize=(8, 8))
        axs[0].title.set_text("Imagen Original")
        axs[0].imshow(imagen, cmap='gray')
        axs[1].title.set_text("Imagen Rotada")
        im = axs[1].imshow(img_rotada, cmap='gray')
        fig.colorbar(im, ax=axs[1])
        axs[2].plot(intensity_profile)
        plt.show()
    # Encontrar los mínimos en el perfil de intensidad
    peaks, _ = find_peaks(-intensity_profile, distance=MINIMUM_DISTANCE_PEAKS,
                          prominence=PROMINENCE_PEAKS)
    return np.mean(np.diff(peaks))


imagen = simular_imagen(frecuencia=50, angulo_franjas_max=0, angulo_slm_max= 0)
crop = imagen[90:220, 100:550]

SHOW_CROP1 = False
if SHOW_CROP1:
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    axs[0].imshow(imagen, cmap='gray')
    axs[1].imshow(crop, cmap='gray')
    plt.show()

if SHOW_INTERFRANJA:
    interfranja = get_interfranja(crop, remove_Gaussian_profile=False, show=True)
    print(f"Diferencia entre picos promedio: {interfranja}")

# Análisis de correlación cruzada
# crop2 = imagen[270:400, 100:550]
r_shift = 0
c_shift = 12
crop2 = imagen[90+r_shift:220+r_shift, 100+c_shift:550+c_shift]

#imag = cv2.imread('fotos_rot_franjas/3004_I0_0_T22_r_f.png')
#plt.imshow(imag)
#plt.show()
#recorte1 = imag[495:555, 170:570, 0]
#recorte2 = imag[690:750, 350:750, 0]
#fig, axs = plt.subplots(1, 2, figsize=(8, 8))
#axs[0].imshow(recorte1, cmap='gray')
#axs[1].imshow(recorte2, cmap='gray')
#plt.imshow(recorte1)
#plt.show()
#señal1 = np.sum(recorte1, axis = 0)
#señal2 = np.sum(recorte2, axis = 0)
#plt.plot(señal1[:100])
#plt.plot(señal2[:100])
#plt.show()
#fft1 = abs(np.fft.fft(señal1-np.mean(señal1)))
#fft2 = abs(np.fft.fft(señal2-np.mean(señal2)))
#plt.plot(fft1[:len(fft1)//2], label = '1')
#plt.plot(fft2[:len(fft2)//2], label = '2', linestyle = '--')
#plt.legend()
#plt.show()

#shift = get_dphi_fft1d(recorte2, recorte1, get_shift=True)
#print(f"Shift: {shift}")
run = False
if run:
    shifts = np.zeros(256)
    for i in range(256):
        print(i)
        imag = cv2.imread(f'fotos_rot_franjas/3004_I{i}_9_T22_r_f.png')
        recorte1 = imag[495:555, 170:570, 0]
        recorte2 = imag[690:750, 350:750, 0]
        shift = get_dphi_fft1d(recorte1, recorte2, get_shift=False)
        shifts[i] = shift
    shifts -= shifts[0]
    for i in range(256):
        while abs(shifts[i])>np.pi:
            if shifts[i]>0:
                shifts[i]-=np.pi
            else:
                shifts[i]+=np.pi
    plt.errorbar(np.arange(256), shifts, capsize = 3, markersize = 3, fmt = 'o', label= 'fase', color ='r')
    plt.xlabel('Valores de gris')
    plt.ylabel('Diferencia de fase')
    plt.legend()
    #plt.savefig('workdir/calibracion_curva.svg')
    plt.show()


#df = pd.DataFrame(data = shifts)
#df.to_csv('workdir/fases_9.csv')
#signals1 = {}
#signals2 = {}
fases = np.linspace(0,2*np.pi,256)
#for i in range(256):
#    imagen = simular_imagen(frecuencia=100, angulo_franjas_max=0, angulo_slm_max= 0, fase2 = fases[i], amplitud_imperfecciones= 0)
#    recorte1 = imagen[150:240, 200:500]
#    recorte2 = imagen[320:410, 200:500]
#    signal1 = np.sum(recorte1, axis=0)
#    signal2 = np.sum(recorte2, axis=0)
#    signals1[f'{i}'] = signal1
#    signals2[f'{i}'] = signal2
#    print(i)

#df_signal1 = pd.DataFrame(signals1)
#df_signal2 = pd.DataFrame(signals2)
#print(df_signal1)
#df_signal1.to_csv('Calibracion_SLM\\data\\signal1.csv')
#df_signal2.to_csv('Calibracion_SLM\\data\\signal2.csv')
signals1 = pd.read_csv('Calibracion_SLM\\data\\signal1.csv')
signals2 = pd.read_csv('Calibracion_SLM\\data\\signal2.csv')

prueba = False
if prueba:
    c1 = np.zeros(256)
    c2 = np.zeros(256)
    m1 = np.zeros(256)
    m2 = np.zeros(256)
    def lineal(x, m, c):
        return m*x+c
    for i in range(256):
        #imagen = simular_imagen(frecuencia=100, angulo_franjas_max=0, angulo_slm_max= 0, fase2 = fases[i], amplitud_imperfecciones= 0)
        #recorte1 = imagen[150:240, 200:500]
        #recorte2 = imagen[320:410, 200:500]
        

        #signal1 = np.sum(recorte1, axis=0)
        #signal2 = np.sum(recorte2, axis=0)
        signal1 = np.array(signals1[f'{i}'])
        signal2 = np.array(signals2[f'{i}'])
        signal1 = signal1 - np.mean(signal1)
        signal2 = signal2 - np.mean(signal2)
        #plt.plot(signal1)
        #plt.plot(signal2)
        #plt.pause(.01)
        #plt.cla()


        f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
        f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
        n = len(f_signal1)
        freq1_ind = np.argmax(np.abs(f_signal1[:n//2]))
        freq2_ind = np.argmax(np.abs(f_signal2[:n//2]))
        frecuencias = np.fft.fftfreq(len(signal1), d=1)
        freq1 = frecuencias[freq1_ind]
        freq2 = frecuencias[freq2_ind]

        fs = 1.0  # muestras por píxel
        bandwidth = 0.01  # ancho de banda en ciclos/píxel, ajustable
        lowcut1 = freq1 - bandwidth
        highcut1 = freq1 + bandwidth

        lowcut2 = freq2 - bandwidth
        highcut2 = freq2 + bandwidth

        
        nyq = 0.5 * fs
        low1 = lowcut1 / nyq
        high1 = highcut1 / nyq
        low2 = lowcut2 / nyq
        high2 = highcut2 / nyq

        #b, a = butter(order, [low, high], btype='bandpass')

        numtaps = 90   # cantidad de coeficientes, más alto = más selectivo
        b1 = firwin(numtaps, [low1, high1], pass_zero=False)
        b2 = firwin(numtaps, [low2, high2], pass_zero=False)

        filtra1 = filtfilt(b1, [1.0], signal1)
        filtra2 = filtfilt(b2, [1.0], signal2)
        f_bandwidth = 0.01
        gauss_mask1 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth)**2) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth)**2)
        gauss_mask2 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth)**2) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth)**2)
        fft_filtrada1 = f_signal1 * gauss_mask1
        fft_filtrada2 = f_signal2 * gauss_mask2
        #filtra1 = np.fft.ifft(fft_filtrada1).real
        #filtra2 = np.fft.ifft(fft_filtrada2).real


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
        m1[i] = popt1[0]
        m2[i] = popt2[0]
        print(i)
    plt.scatter(np.arange(256), c1)
    plt.show()
    plt.scatter(np.arange(256), c2)
    plt.show()
    plt.plot(c1-c2)
    plt.show()
    dif = np.unwrap(c1)-np.unwrap(c2)
    plt.plot(dif)
    plt.xlabel('Valores de gris')
    plt.ylabel('Diferencia de fase')
    plt.title('Diferencia de fase entre franjas')
    
    plt.show()
    plt.plot(m1-m2, label = f'{np.max(np.abs(m1-m2))}')
    plt.legend()
    plt.show()
    f_err = np.abs(np.fft.fft(m1-m2))
    nn = len(f_err)
    plt.plot(f_err[:n//2])
    print(np.argmax(f_err[:n//2]))
    plt.show()
else :
    imagen = simular_imagen(frecuencia=50, angulo_franjas_max=0, angulo_slm_max= 0, fase2 = fases[128], amplitud_imperfecciones=0)
    plt.imshow(imagen)
    plt.show()
    recorte1 = imagen[150:240, 200:500]
    recorte2 = imagen[320:410, 200:500]

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(recorte1)
    ax[1].imshow(recorte2)
    plt.show()

    signal1 = np.sum(recorte1, axis=0)
    signal2 = np.sum(recorte2, axis=0)
    print(len(signal1))

    fig, ax = plt.subplots(1,2)
    ax[0].plot(signal1)
    ax[1].plot(signal2)
    plt.show()

    f_signal1 = np.fft.fft(signal1 - np.mean(signal1))
    f_signal2 = np.fft.fft(signal2 - np.mean(signal2))
    n = len(np.abs(f_signal1))


    freq1_ind = np.argmax(np.abs(f_signal1[:n//2]))
    freq2_ind = np.argmax(np.abs(f_signal2[:n//2]))
    frecuencias = np.fft.fftfreq(len(signal1), d=1)
    freq1 = frecuencias[freq1_ind]
    freq2 = frecuencias[freq2_ind]


    fs = 1.0  # muestras por píxel
    bandwidth = 0.01  # ancho de banda en ciclos/píxel, ajustable
    lowcut1 = freq1 - bandwidth
    highcut1 = freq1 + bandwidth

    lowcut2 = freq2 - bandwidth
    highcut2 = freq2 + bandwidth


    order = 4
    nyq = 0.5 * fs
    low1 = lowcut1 / nyq
    high1 = highcut1 / nyq
    low2 = lowcut2 / nyq
    high2 = highcut2 / nyq



    numtaps = 40  # cantidad de coeficientes, más alto = más selectivo
    b1 = firwin(numtaps, [low1, high1], pass_zero=False)
    b2 = firwin(numtaps, [low2, high2], pass_zero=False)

    print('b1', np.linalg.norm(b1), 'b2', np.linalg.norm(b2))
    print('b1', np.sum(b1), 'b2', np.sum(b2))

    f_b1 = np.fft.fft(b1, len(signal1))
    f_b2 = np.fft.fft(b2, len(signal2))

    plt.plot(frecuencias[:n//2], np.abs(f_signal1)[:n//2]/max(np.abs(f_signal1)[:n//2]))
    plt.axvline(lowcut1, color ='r')
    plt.axvline(highcut1, color ='r')
    plt.plot(frecuencias[:n//2], np.abs(f_b1[:n//2])/max(np.abs(f_b1[:n//2])))
    plt.show()

    plt.plot(frecuencias[:n//2], np.abs(f_signal2)[:n//2]/max(np.abs(f_signal2)[:n//2]))
    plt.axvline(lowcut2, color ='r')
    plt.axvline(highcut2, color ='r')
    plt.plot(frecuencias[:n//2], np.abs(f_b2[:n//2])/max(np.abs((f_b2[:n//2]))))
    plt.show()

    f_bandwidth = 0.01

    gauss_mask1 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth)**2) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth)**2)
    gauss_mask2 = np.exp(-0.5 * ((frecuencias - freq1) / f_bandwidth)**2) + np.exp(-0.5 * ((frecuencias + freq1) / f_bandwidth)**2)
    plt.plot(frecuencias, gauss_mask1)
    plt.plot(frecuencias, np.abs(f_signal1)/max(np.abs(f_signal1)))
    plt.show()
    plt.plot(np.abs(f_signal1*gauss_mask1))
    plt.show()
    fft_filtrada1 = f_signal1 * gauss_mask1
    fft_filtrada2 = f_signal2 * gauss_mask2
    filtra1 = np.fft.ifft(fft_filtrada1).real
    filtra2 = np.fft.ifft(fft_filtrada2).real
    

    filtra1 = filtfilt(b1, [1.0], signal1-np.mean(signal1))
    filtra2 = filtfilt(b2, [1.0], signal2-np.mean(signal2))

    plt.plot(filtra1)
    plt.xlabel('Pixel')
    plt.ylabel('Intensidad')
    plt.title('Señal filtrada')
    
    plt.show()

    plt.plot(filtra2)
    plt.show()

    hil1 = hilbert(filtra1)
    hil2 = hilbert(filtra2)

    

    lin1 = np.unwrap(np.angle(hil1))

    fig, ax = plt.subplots(1,2, figsize=(10, 6))
    ax[0].plot(np.angle(hil1)[:100])
    ax[0].set_title('Fase hilbert')
    ax[1].plot(lin1[:100])
    ax[1].set_title('Fase hilbert unwrapped')
    plt.savefig('Calibracion_SLM\\data\\hilbert_fir.png', bbox_inches='tight')
    plt.show()


    lin2 = np.unwrap(np.angle(hil2))

    plt.plot(lin1)
    plt.plot(lin2)
    plt.show()