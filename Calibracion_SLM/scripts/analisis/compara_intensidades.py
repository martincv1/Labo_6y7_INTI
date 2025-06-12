import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

img = cv2.imread(f'fotos_rot/3004_I200_2_T22_r.png')



print(np.arange(0, 255, 25))

ancho = 400
alto = 200
x1 = 400
y1 = 100
x2 = 200
y2 = 350

rec1 = img[y1:y1+alto, x1:x1+ancho]
rec2 = img[y2:y2+alto, x2:x2+ancho]
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')

rect1 = patches.Rectangle((x1, y1), ancho, alto, linewidth=2, edgecolor='r', facecolor='none')

rect2 = patches.Rectangle((x2, y2), ancho, alto, linewidth=2, edgecolor='r', facecolor='none')

ax.add_patch(rect1)
ax.add_patch(rect2)

plt.show()

corre = True
if corre:
    inten1 = np.zeros(256)
    inten2 = np.zeros(256)
    for i in range(256):
        imagen = cv2.imread(f'fotos_rot/3004_I{i}_2_T22_r.png')
        recorte1 = imagen[y1:y1+alto, x1:x1+ancho]
        recorte2 = imagen[y2:y2+alto, x2:x2+ancho]
        inten1[i] = np.mean(recorte1)
        inten2[i] = np.mean(recorte2)
        print(i)
    plt.plot(inten1)
    plt.plot(inten2)
    plt.show()  
    dif = inten1-inten2
    plt.plot(dif)
    plt.show()
    print(np.var(dif))
