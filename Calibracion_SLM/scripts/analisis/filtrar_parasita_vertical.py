import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def notch_star(shape, vx, vy, base_width=20, alpha=0.5, beta=0.5):
    """
    Genera un filtro notch tipo 'estrella' centrado en (vx, vy).

    shape: (M,N) tamaño de la imagen
    vx, vy: coordenadas de la frecuencia notch (en unidades de frecuencia normalizada: -0.5 a 0.5)
    base_width: ancho base del notch
    alpha, beta: parámetros para variar la anisotropía angular
    """
    alpha_beta_sum = alpha + beta
    alpha = alpha / alpha_beta_sum
    beta = beta / alpha_beta_sum
    M, N = shape
    u = np.fft.fftfreq(N)
    v = np.fft.fftfreq(M)
    U, V = np.meshgrid(u, v)
    # Coordenadas relativas al notch
    dU = U - vx
    dV = V - vy

    # Distancia radial cuadrada
    R = dU**2 + dV**2
    # Ángulo
    theta = np.arctan2(dV, dU)
    # Ancho angular dependiente
    width = base_width * (alpha + beta * np.cos(4 * theta) ** 5)
    # Filtro notch (cercano a 0 dentro del notch, 1 fuera)
    arg_exponential = R / (2 * width**2)
    H = 1 - np.exp(- arg_exponential)
    H = (H + H[::-1, ::-1]) / 2  # Simetría respecto al origen
    return H


def notch_Lorentzian(shape, vx, vy, sigma=0.1):
    """
    Genera un notch simétrico (en (vx,vy) y (-vx,-vy)) con forma rectangular/Lorentziana.

    - shape: (M,N)
    - vx, vy: posición de la frecuencia objetivo
    - sigma: ancho del notch en x e y
    """
    M, N = shape
    u = np.fft.fftfreq(N)
    v = np.fft.fftfreq(M)
    U, V = np.meshgrid(u, v)

    def single_notch(vx0, vy0):
        dU = U - vx0
        dV = V - vy0
        Lu = 1.0 / (1.0 + (dU / sigma)**2)
        Lv = 1.0 / (1.0 + (dV / sigma)**2)
        return Lu * Lv

    B1 = single_notch(vx, vy)
    B2 = single_notch(-vx, -vy)
    B = np.maximum(B1, B2)  # unimos los dos picos notch
    H = 1.0 - B
    return H


def center_of_mass(X, Y, im):
    assert X.ndim == 1 and Y.ndim == 1, "X and Y must be one-dimensional arrays"
    assert X.size == im.shape[1] and Y.size == im.shape[0], \
           "X and Y must have length equal to the number of columns and rows of im, respectively"
    Y = Y[:, np.newaxis]  # Convertir a columna
    X = X[np.newaxis, :]  # Convertir a fila
    total = np.sum(im)
    y_cm = np.sum(Y * im) / total
    x_cm = np.sum(X * im) / total
    return y_cm, x_cm


def filtrar_parasita_vertical(imagen_path, vx_window_search=0.25, width_window_search=0.1, show_filter=False):
    imagen = cv2.imread(imagen_path)[:, :, 0]
    M, N = imagen.shape
    imagen = imagen.astype(np.float32)

    # Encontrar la frecuencia de la componente que tiene más energía cerca de 0 en vertical
    fft = np.fft.fft2(imagen)
    freq_v = np.fft.fftfreq(M)  # Frecuencias verticales (de -0.5 a 0.5)
    freq_h = np.fft.fftfreq(N)  # Frecuencias horizontales (de -0.5 a 0.5)

    componentes = np.abs(fft)
    componentes[np.abs(freq_v) > width_window_search / 2, :] = 0  # Filtrar frecuencias altas
    componentes[:, np.abs(freq_h - vx_window_search) > width_window_search / 2] = 0
    vy_notch, vx_notch = center_of_mass(freq_h, freq_v, componentes)

    # Crear un filtro notch centrado en la frecuencia encontrada
    # H = notch_star((M, N), vx_notch, vy_notch, base_width=0.5, alpha=0.6, beta=0.5)
    H = notch_Lorentzian((M, N), vx_notch, vy_notch, sigma=0.02)  # Alternativa: filtro Lorentziano

    # Aplicar el filtro notch
    fft_filtrada = fft * H
    imagen_filtrada = np.fft.ifft2(fft_filtrada).real

    if show_filter:
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        axs[0].imshow(np.fft.fftshift(np.log(np.abs(fft))), cmap="gray")
        axs[0].set_title("FFT de la imagen")
        axs[1].imshow(np.fft.fftshift(np.log(H)), cmap="gray")
        axs[1].set_title("Filtro notch tipo estrella")
        axs[2].imshow(np.fft.fftshift(np.log(np.abs(fft_filtrada))), cmap="gray")
        axs[2].set_title("FFT filtrada")
        plt.show()

    return imagen_filtrada


if __name__ == "__main__":
    path_im = r"/home/pablo/OneDrive/Documentos/Proyectos/IA Optics/SLM/2025_10_02 topografia/SLM en 0"
    name_im = "00000006_000000006B43B8F7.png"
    path_output = os.path.join(path_im, "Filtered")

    imagen_filtrada = filtrar_parasita_vertical(os.path.join(path_im, name_im), vx_window_search=0.2, show_filter=True)
    plt.imshow(imagen_filtrada, cmap="gray")
    plt.title("Imagen filtrada")
    plt.colorbar()
    plt.show()

    os.makedirs(path_output, exist_ok=True)
    cv2.imwrite(os.path.join(path_output, name_im), imagen_filtrada)
