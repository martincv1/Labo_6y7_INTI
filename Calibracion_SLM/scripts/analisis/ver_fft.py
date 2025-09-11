import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

path_im = r"C:\Users\INTI\Documents\Labo67JM\SLM_topografia"
name_im = "00.png"

im = cv2.imread(os.path.join(path_im, name_im)).astype(np.float32)[:, :, 0]
fim = np.fft.fftshift(np.fft.fft2(im))

plt.imshow(np.log(np.abs(fim)))
plt.show()
