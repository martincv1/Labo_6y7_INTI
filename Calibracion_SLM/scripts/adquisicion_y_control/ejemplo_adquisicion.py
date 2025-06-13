# Importo librerias
import numpy as np
import eBUS as eb
import time
import pickle
import os
import warnings
import cv2

# En la misma ruta que donde esta el codigo tiene que estar la carpeta HEDS, hedslib y lib.
# Tmb hay que instalar con pip install el archivo .whl del ebus
import lib.PvSampleUtils as psu
import HEDS
from hedslib.heds_types import *

import Calibracion_SLM.scripts.adquisicion_y_control.acquisition_tools as acq_tools
import Calibracion_SLM.scripts.adquisicion_y_control.slm_tools as slm_tools


device, stream = acq_tools.initt()
# Termino de configurar algunas cosas
acq_tools.configure_stream(device, stream)
buffer_list = acq_tools.configure_stream_buffers(device, stream)
print('taman', len(buffer_list))
""" Empiezo a streamear """
# Primero agarro algunas propiedades de Genicam
# Get device parameters need to control streaming
device_params = device.GetParameters()
# Map the GenICam AcquisitionStart and AcquisitionStop commands
start = device_params.Get("AcquisitionStart")
stop = device_params.Get("AcquisitionStop")
# Get stream parameters
stream_params = stream.GetParameters()
# Map a few GenICam stream stats counters
frame_rate = stream_params.Get("AcquisitionRate")
bandwidth = stream_params["Bandwidth"]
# Enable streaming and send the AcquisitionStart command
print("Enabling streaming and sending AcquisitionStart command.")
device.StreamEnable()
start.Execute()
# Por las dudas pongo un sleep
time.sleep(2)

doodle = "|\\-|-/"
doodle_index = 0
display_image = False
warning_issued = False
errors = 0
decompression_filter = eb.PvDecompressionFilter()

##########################################

intensidades_array = np.array([40, 120, 200])
print(intensidades_array)
# Inicializo el SLM
# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4, 0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

# Este es el bucle de medición
resol_SLM = (1080, 1920)
tiempo_espera_inicial = 180
print(f'Espero {tiempo_espera_inicial} s antes de empezar')
time.sleep(tiempo_espera_inicial)
fecha = '1206'
T_dia = 21
cant_promedio = 5
save_dir = r"data\fase_tiempo"
muestras = 240
intervalo = 5  # para el sleep entre mediciones
cant_pruebas_retrieve = 4
tiempo_prueba = 0.05

os.makedirs(save_dir, exist_ok=True)
for i in intensidades_array:
    print(f"Mostrando patrón con intensidad: {i}")
    patron = slm_tools.crear_patron(resol_SLM, "horizontal", "sup", i)
    # va a guardar cada patron como archivo.npy en la carpeta patrones
    # np.save(f"patrones/patron_{idx:03d}.npy", patron)
    err, dataHandle = slm.loadImageData(patron)  # carga la data (array) a la video memory del display SLM
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    err = dataHandle.show()  # Show the returned data handle on the SLM
    assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
    time.sleep(0.5)
    result, pvbuffer, operational_result = stream.RetrieveBuffer(1000)
    stream.QueueBuffer(pvbuffer)  # Acá manda el buffer a buscar
    for h in range(muestras):
        print(f"midiendo muestra {h}/{muestras}")
        lista_promedio = []
        t_ini = time.time()

        for j in range(cant_promedio):

            # Retrieve next pvbuffer
            pruebas = 0
            while pruebas <= cant_pruebas_retrieve:
                result, pvbuffer, operational_result = stream.RetrieveBuffer(1000)
                if acq_tools.buffer_check(result, doodle, doodle_index, operational_result):
                    break
                stream.QueueBuffer(pvbuffer)
                time.sleep(tiempo_prueba)
                pruebas += 1

            #
            # We now have a valid pvbuffer. This is where you would typically process the pvbuffer.
            # -----------------------------------------------------------------------------------------
            # ...
            result, frame_rate_val = frame_rate.GetValue()
            result, bandwidth_val = bandwidth.GetValue()
            print(f"{doodle[doodle_index]} BlockID: {pvbuffer.GetBlockID():016d}", end='')

            image = None
            payload_type = pvbuffer.GetPayloadType()

            image = acq_tools.get_data(pvbuffer, decompression_filter, payload_type)  # acá consigue la imagen

            if image:

                print(f"  W: {image.GetWidth()} H: {image.GetHeight()} ", end='')
                image_data = image.GetDataPointer()
                # guardar imagen
                # image_data = image_data[:,:,0]
                lista_promedio.append(image_data)

            # print(f" {frame_rate_val:.1f} FPS  {bandwidth_val / 1000000.0:.1f} Mb/s     ", end='\r')

            # Re-queue the pvbuffer in the stream object
            stream.QueueBuffer(pvbuffer)  # Acá manda el buffer a buscar

            doodle_index = (doodle_index + 1) % 6
        hora_actual = time.strftime("%H-%M-%S", time.localtime())
        file_name = os.path.join(save_dir, f'{fecha}_{hora_actual}_I{i}_T{T_dia}')
        if cant_promedio == 1:
            cv2.imwrite(f"{file_name}.png", image_data)
        else:
            arr = np.stack(lista_promedio, axis=2)
            image_data_prom = np.mean(arr, axis=2)
            with open(f"{file_name}.pkl", 'wb') as f:
                pickle.dump(image_data_prom, f)
        t_final = time.time()
        time.sleep(intervalo - (t_final - t_ini))
# Acá se cierra el while

# Acá se apaga todo

# Tell the device to stop sending images.
print("\nSending AcquisitionStop command to the device")
stop.Execute()

# Disable streaming on the device
print("Disable streaming on the controller.")
device.StreamDisable()

# Abort all buffers from the stream and dequeue
print("Aborting buffers still in stream")
stream.AbortQueuedBuffers()
while stream.GetQueuedBufferCount() > 0:
    result, pvbuffer, lOperationalResult = stream.RetrieveBuffer()

# if image.GetPixelType() == eb.PvPixelMono8:
#    display_image = True
if image.GetPixelType() == eb.PvPixelRGB8:
    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    display_image = True

cv2.imshow("foto", image_data)
cv2.imwrite("fotin.jpg", image_data)
