import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import correlate2d
from skimage.registration import phase_cross_correlation
import time

imagen = cv2.imread('fotos_rot/3004_I230_0_T22_r.png')
imagen = imagen * 2
plt.imshow(imagen)
plt.show()

recorte1 = imagen[120:310,370:650,0]
recorte2 = imagen[330:520,370:650,0]

fig, axs = plt.subplots(2, 1, figsize=(10, 5))

# Mostrar región inferior
axs[0].imshow(recorte1, cmap='gray')
axs[0].set_title("Región inferior")
axs[0].axis('off')

# Mostrar región superior
axs[1].imshow(recorte2, cmap='gray')
axs[1].set_title("Región superior")
axs[1].axis('off')

plt.tight_layout()
plt.show()

corr2d = correlate2d(recorte1.astype(np.float32), recorte2.astype(np.float32), mode='full')

# Mostrar la correlación
plt.imshow(corr2d, cmap='viridis')
plt.colorbar(label='Correlación')
plt.title("Correlación cruzada 2D")
plt.axis('off')
plt.show()


correlacion = phase_cross_correlation(recorte1.astype(np.float32), recorte2.astype(np.float32), upsample_factor= 1)
dy = correlacion[0][0]
dx = correlacion[0][1]
modu = np.sqrt(dx**2+dy**2)
print(correlacion[0], modu)
mdl = np.zeros(256)
dd = np.zeros((2, 256))
medir = False
if medir:
    for i in range(256):
        imagen = cv2.imread(f'fotos_rot/3004_I{i}_0_T22_r.png')
        recorte1 = imagen[120:310,370:650,0]
        recorte2 = imagen[330:520,370:650,0]
        correlacion = phase_cross_correlation(recorte1.astype(np.float32), recorte2.astype(np.float32), upsample_factor= 1)
        if i == 0:
            dx0 = correlacion[0][1]
            dy0 = correlacion[0][0]
        dy = correlacion[0][0] - dy0
        dx = correlacion[0][1] - dx0
        modu = np.sqrt(dx**2+dy**2)
        mdl[i] = modu
        dd[:,i] = correlacion[0]
        print(i)

    #d = 10
    #for i in range(256):
    #    while(mdl[i]>d):
    #        mdl[i]-=d
    plt.scatter(np.arange(256), mdl)
    plt.show()
    plt.figure()
    plt.scatter(dd[0], dd[1])
    plt.show()


#for i in range(256):
#    imagen = cv2.imread(f'fotos_rot/3004_I{i}_0_T22_r.png')
#    cv2.imshow('foto', imagen)
#    if cv2.waitKey(200) & 0xFF == ord('q'):
#        break