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
guardar = True
if guardar:
    for i in range(256):
        for j in range(10):
            imagen = cv2.imread(f'fotos_rot/{fecha}_I{i}_{j}_T22_r.png')
            rotated_image = rotate_bound(imagen, -27.5)
            #recorte = rotated_image[90:720,340:1380]
            cv2.imwrite(f'fotos_rot_franjas/{fecha}_I{i}_{j}_T22_r_f.png', rotated_image)

imag = cv2.imread('fotos_rot/3004_I247_0_T22_r.png')
rotated_image = rotate_bound(imag, -27.5)
plt.imshow(rotated_image*2)
plt.show()
