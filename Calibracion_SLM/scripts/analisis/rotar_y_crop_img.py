import cv2
import numpy as np
import matplotlib.pyplot as plt

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



fecha = 3004
promediar = False
if promediar:
    for i in range(256):
        print(i)
        if i != 0:   #Excluyo la primer medición porque salió mal
            imag = cv2.imread(f'../../../../fotos_rot/{fecha}_I{i}_0_T22_r.png')                            # Sumo esta condición porque la primer imagen 
            for j in range(9):
                imag += cv2.imread(f'../../../../fotos_rot/{fecha}_I{i}_{j+1}_T22_r.png')
            imag = imag//10
            cv2.imwrite(f'../../../../fotos_rot_prom/{fecha}_I{i}_T22_r_p.png', imag)
        else:
            imag = cv2.imread(f'../../../../fotos_rot/{fecha}_I{i}_1_T22_r.png')
            for j in range(8):
                imag += cv2.imread(f'../../../../fotos_rot/{fecha}_I{i}_{j+2}_T22_r.png')
            imag = imag//9
            cv2.imwrite(f'../../../../fotos_rot_prom/{fecha}_I{i}_T22_r_p.png', imag)


contraste = False
if contraste:
    for i in range(256):
        print('i', i)
        image = cv2.imread(f'../../../../fotos_rot_prom/{fecha}_I{i}_T22_r_p.png')
        print(np.min(image))
        print(np.max(image))
        #image -= np.min(image)
        image *= 255//np.max(image)
        cv2.imwrite(f'../../../../fotos_inten/{fecha}_I{i}_T22_r_i.png', image) 

fecha = 3004
guardar = False
if guardar:
    for i in range(256):
        imagen = cv2.imread(f'../../../../fotos_rot_prom/{fecha}_I{i}_T22_r_p.png')
        rotated_image = rotate_bound(imagen, -27.5)
        #recorte = rotated_image[90:720,340:1380]
        cv2.imwrite(f'../../../../fotos_rot_franjas/{fecha}_I{i}_T22_r_p_final.png', rotated_image)

#imag = cv2.imread('fotos_rot/3004_I247_0_T22_r.png')
#rotated_image = rotate_bound(imag, -27.5)
#plt.imshow(rotated_image*2)
#plt.show()

#for i in range(256):
#    foto = cv2.imread(f'../../../../fotos_inten/{fecha}_I{i}_T22_r_i.png')
#    cv2.imshow('foto', foto)
#    if cv2.waitKey(30) & 0xFF == ord('q'):  # espera 30 ms, salir con 'q'
#        break