# Importo librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy

from acquisition_tools import jai_camera

"""----------------------------- Defino funciones de configuración -----------------------"""
BUFFER_COUNT = 16
WAIT_BETWEEN_FRAMES = 0.5  # segundos

camera = jai_camera(buffers=BUFFER_COUNT, verbose=True)
number_test_frames = 51
camera.reset_queue()
last_frame = None
mean_diffs = np.zeros(number_test_frames - 1)
times = np.zeros(number_test_frames)

for i in range(number_test_frames):
    t_ini = time.time()
    frame = camera.get_frame()
    times[i] = time.time() - t_ini
    if last_frame is not None:
        mean_diffs[i - 1] = np.mean(np.abs(frame - last_frame))
    time.sleep(WAIT_BETWEEN_FRAMES)
    last_frame = deepcopy(frame)

print(f"Average of Mean abs_diff with last frame: {np.mean(mean_diffs)}")
print(f"Std of Mean abs_diff with last frame: {np.std(mean_diffs)}")
print(f"Count of zero diffs: {np.sum(mean_diffs==0)}")
print(f"Average frame retrieval time: {np.mean(times)} s +- {np.std(times)} s")

try_multiple_frames = False
if try_multiple_frames:
    frames = camera.get_multiple_frame(10)
    plt.imshow(frames[9])
    plt.show()
    print(frames[0])
# # mean_diff = np.mean(np.abs(frame1 - frame2))
# print(mean_diff)
# plt.imshow(frame1 - frame2)
# plt.show()
camera.close()

# cv2.imshow("foto1", frame1)
# cv2.imshow("foto2", frame2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
