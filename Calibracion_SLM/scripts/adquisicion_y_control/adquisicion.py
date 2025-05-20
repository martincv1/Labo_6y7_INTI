#Importo librerias
import numpy as np
import eBUS as eb
import lib.PvSampleUtils as psu
import HEDS
from hedslib.heds_types import *
import time

########################################## Defino funciones de configuración
BUFFER_COUNT = 1

kb = psu.PvKb()

opencv_is_available=True
try:
    # Detect if OpenCV is available
    import cv2
    opencv_version=cv2.__version__
except:
    opencv_is_available=False
    print("Warning: This sample requires python3-opencv to display a window")

def connect_to_device(connection_ID):
    # Connect to the GigE Vision or USB3 Vision device
    print("Connecting to device.")
    result, device = eb.PvDevice.CreateAndConnect(connection_ID)
    if device == None:
        print(f"Unable to connect to device: {result.GetCodeString()} ({result.GetDescription()})")
    return device

def open_stream(connection_ID):
    # Open stream to the GigE Vision or USB3 Vision device
    print("Opening stream from device.")
    result, stream = eb.PvStream.CreateAndOpen(connection_ID)
    if stream == None:
        print(f"Unable to stream from device. {result.GetCodeString()} ({result.GetDescription()})")
    return stream

def configure_stream(device, stream):
    # If this is a GigE Vision device, configure GigE Vision specific streaming parameters
    if isinstance(device, eb.PvDeviceGEV):
        # Negotiate packet size
        device.NegotiatePacketSize()
        # Configure device streaming destination
        device.SetStreamDestination(stream.GetLocalIPAddress(), stream.GetLocalPort())

def configure_stream_buffers(device, stream):
    buffer_list = []
    # Reading payload size from device
    size = device.GetPayloadSize()

    # Use BUFFER_COUNT or the maximum number of buffers, whichever is smaller
    buffer_count = stream.GetQueuedBufferMaximum()
    if buffer_count > BUFFER_COUNT:
        buffer_count = BUFFER_COUNT

    # Allocate buffers
    for i in range(buffer_count):
        # Create new pvbuffer object
        pvbuffer = eb.PvBuffer()
        # Have the new pvbuffer object allocate payload memory
        pvbuffer.Alloc(size)
        # Add to external list - used to eventually release the buffers
        buffer_list.append(pvbuffer)
    
    # Queue all buffers in the stream
    for pvbuffer in buffer_list:
        stream.QueueBuffer(pvbuffer)
    print(f"Created {buffer_count} buffers")
    return buffer_list
##########################################
# Defino una funcion para conectar camara y emepezar stream
# y chequear que todo conectó o empezó bien
def initt():
    connection_ID = psu.PvSelectDevice()
    if not connection_ID:
        raise Exception("Error al seleccionar ID")
    device = connect_to_device(connection_ID)
    if not device:
        raise Exception("Error al conectar dispositivo")
    stream = open_stream(connection_ID)
    if not stream:
        raise Exception("Error al abrir stream")
    return device, stream

# Defino una función que chequee el tipo de data que saca del buffer y adquiere una imagen
def get_data(payload_type):
    if payload_type == eb.PvPayloadTypeImage:
        image = pvbuffer.GetImage()

    elif payload_type == eb.PvPayloadTypeChunkData:
        print(f" Chunk Data payload type with {pvbuffer.GetChunkCount()} chunks", end='')

    elif payload_type == eb.PvPayloadTypeRawData:
        print(f" Raw Data with {pvbuffer.GetRawData().GetPayloadLength()} bytes", end='')

    elif payload_type == eb.PvPayloadTypeMultiPart:
        print(f" Multi Part with {pvbuffer.GetMultiPartContainer().GetPartCount()} parts", end='')

    elif payload_type == eb.PvPayloadTypePleoraCompressed:
        if eb.PvDecompressionFilter.IsCompressed(pvbuffer):
            result, pixel_type, width, height = eb.PvDecompressionFilter.GetOutputFormatFor(pvbuffer)
            if result.IsOK():
                calculated_size = eb.PvImage.GetPixelSize(pixel_type) * width * height / 8;
                out_buffer = eb.PvBuffer()
                result, decompressed_buffer = decompression_filter.Execute(pvbuffer, out_buffer)
                image = decompressed_buffer.GetImage()
                if result.IsOK():
                    decompressed_size = decompressed_buffer.GetSize()
                    compression_ratio = decompressed_size / pvbuffer.GetAcquiredSize()
                    if calculated_size != decompressed_size:
                        errors = errors + 1
                    print(f" Pleora compressed type.   Compression ratio: {'{0:.2f}'.format(compression_ratio)} Errors: {errors}", end='')
                else:
                    print(f" Could not decompress (Pleora compressed)", end='')
                    errors = errors + 1
            else:
                print(f" Could not read header (Pleora compressed)", end='')
                errors = errors + 1
        else:
            print(f" Contents do not match payload type (Pleora compressed)", end='')
            errors = errors + 1

    else:
        print(" Payload type not supported by this sample", end='')
    return image


# Defino una función que chequee si el buffer adquirió un resultado útil o no
def buffer_check(result, operational_result):
    if not result.IsOK():
        print(f"{doodle[ doodle_index ]} {result.GetCodeString()}      ", end='\r')
        raise Exception("Buffer no adquirido")
    if not operational_result.IsOK():
        print(f"{doodle[ doodle_index ]} {operational_result.GetCodeString()}       ", end='\r')
        raise Exception("Buffer mal adquirido")

# Función que crea el patron escalon
def crear_patron(resolucion, orientacion, mitad, intensidad):
    grayscale_array = np.zeros(resolucion, dtype=np.uint8)
    if orientacion == "horizontal":
        half_height = resolucion[0]//2  #altura mitad
        if mitad == "sup":
            grayscale_array[:half_height, :] = intensidad #llena las filas hasta half_height con intensidad
        elif mitad == "inf":
            grayscale_array[half_height:, :] = intensidad #llena las filas desde half_height con intensidad
    if orientacion == "vertical":
        half_width = resolucion[1]//2
        if mitad == "izq":
            grayscale_array[:, :half_width] = intensidad
        elif mitad =="der":
            grayscale_array[:, half_width:] = intensidad
    return grayscale_array


##########################################


device, stream = initt()

# Termino de configurar algunas cosas
configure_stream(device, stream)
buffer_list = configure_stream_buffers(device, stream)
print('taman', len(buffer_list))
########################################## Empiezo a streamear
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
bandwidth = stream_params[ "Bandwidth" ]
# Enable streaming and send the AcquisitionStart command
print("Enabling streaming and sending AcquisitionStart command.")
device.StreamEnable()
start.Execute()
#Por las dudas pongo un sleep
time.sleep(2)

doodle = "|\\-|-/"
doodle_index = 0
display_image = False
warning_issued = False
errors = 0
decompression_filter = eb.PvDecompressionFilter()

##########################################

intensidades_array = np.arange(256)
#Inicializo el SLM
# Init HOLOEYE SLM Display SDK and make sure to check for the correct version this script was written with:
err = HEDS.SDK.Init(4,0)
assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

# Open device detection and retrieve one SLM, and open the SLM Preview window in "Fit" mode for the selected SLM:
slm  = HEDS.SLM.Init("", True, 0.0)
assert slm.errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(slm.errorCode())

#Este es el bucle de medición
resol_SLM = (1080, 1920)
time.sleep(120)
fecha = 3004
for i in intensidades_array:
    for j in range(10):
        print(f"Mostrando patrón con intensidad: {i}")
        patron = crear_patron(resol_SLM, "horizontal", "sup", i )
        #np.save(f"patrones/patron_{idx:03d}.npy", patron)    # va a guardar cada patron como archivo.npy en la carpeta patrones
        err, dataHandle = slm.loadImageData(patron)  #carga la data (array) a la video memory del display SLM
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        err = dataHandle.show() # Show the returned data handle on the SLM
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        time.sleep(0.5)
        # Retrieve next pvbuffer
        result, pvbuffer, operational_result = stream.RetrieveBuffer(1000)
        buffer_check(result, operational_result)
        #
        # We now have a valid pvbuffer. This is where you would typically process the pvbuffer.
        # -----------------------------------------------------------------------------------------
        # ...
        result, frame_rate_val = frame_rate.GetValue()
        result, bandwidth_val = bandwidth.GetValue()
        print(f"{doodle[doodle_index]} BlockID: {pvbuffer.GetBlockID():016d}", end='')

        image = None    
        payload_type = pvbuffer.GetPayloadType()
        
        image = get_data(payload_type) # acá consigue la imagen

        if image:

            print(f"  W: {image.GetWidth()} H: {image.GetHeight()} ", end='')
            image_data = image.GetDataPointer()
            #guardar imagen
            cv2.imwrite(f'fotos/{fecha}_I{i}_{j}_T22.png', image_data)

        #print(f" {frame_rate_val:.1f} FPS  {bandwidth_val / 1000000.0:.1f} Mb/s     ", end='\r')
        
        # Re-queue the pvbuffer in the stream object
        stream.QueueBuffer(pvbuffer) # Acá manda el buffer a buscar
        
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

#if image.GetPixelType() == eb.PvPixelMono8:
#    display_image = True
if image.GetPixelType() == eb.PvPixelRGB8:
    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    display_image = True

cv2.imshow("foto", image_data)
cv2.imwrite("fotin.jpg", image_data)
