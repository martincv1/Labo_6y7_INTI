# Importo librerias
import numpy as np
import eBUS as eb
import lib.PvSampleUtils as psu
import time
import cv2

import Calibracion_SLM.scripts.adquisicion_y_control.acquisition_tools as acq_tools

"""----------------------------- Defino funciones de configuración -----------------------"""
BUFFER_COUNT = 16

device, stream = acq_tools.initt()

# Termino de configurar algunas cosas
acq_tools.configure_stream(device, stream)
buffer_list = acq_tools.configure_stream_buffers(device, stream, set_buffer_count=BUFFER_COUNT)

"""----------------------------- Empiezo a streamear -------------------------------------"""
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

time.sleep(2)

doodle = "|\\-|-/"
doodle_index = 0
display_image = False
warning_issued = False
errors = 0
decompression_filter = eb.PvDecompressionFilter()

flag = True
while flag:
    # Retrieve next pvbuffer
    result, pvbuffer, operational_result = stream.RetrieveBuffer(1000)
    acq_tools.buffer_check(result, doodle, doodle_index, operational_result)

    #
    # We now have a valid pvbuffer. This is where you would typically process the pvbuffer.
    # -----------------------------------------------------------------------------------------
    # ...

    result, frame_rate_val = frame_rate.GetValue()
    result, bandwidth_val = bandwidth.GetValue()

    print(f"{doodle[doodle_index]} BlockID: {pvbuffer.GetBlockID():016d}", end='')

    image = None

    payload_type = pvbuffer.GetPayloadType()

    image = acq_tools.get_data(pvbuffer, decompression_filter, payload_type)

    if image:

        print(f"  W: {image.GetWidth()} H: {image.GetHeight()} ", end='')
        image_data = image.GetDataPointer()

        flag = False

    print(f" {frame_rate_val:.1f} FPS  {bandwidth_val / 1000000.0:.1f} Mb/s     ", end='\r')

    # Re-queue the pvbuffer in the stream object
    stream.QueueBuffer(pvbuffer)  # Acá pasa a otra imagen

    doodle_index = (doodle_index + 1) % 6

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
