# Importo librerias
import eBUS as eb
import lib.PvSampleUtils as psu
import warnings

""" ------------------------- Defino funciones de configuración ------------------------------------"""
kb = psu.PvKb()


def connect_to_device(connection_ID):
    # Connect to the GigE Vision or USB3 Vision device
    print("Connecting to device.")
    result, device = eb.PvDevice.CreateAndConnect(connection_ID)
    if device is None:
        print(f"Unable to connect to device: {result.GetCodeString()} ({result.GetDescription()})")
    return device


def open_stream(connection_ID):
    # Open stream to the GigE Vision or USB3 Vision device
    print("Opening stream from device.")
    result, stream = eb.PvStream.CreateAndOpen(connection_ID)
    if stream is None:
        print(f"Unable to stream from device. {result.GetCodeString()} ({result.GetDescription()})")
    return stream


def configure_stream(device, stream):
    # If this is a GigE Vision device, configure GigE Vision specific streaming parameters
    if isinstance(device, eb.PvDeviceGEV):
        # Negotiate packet size
        device.NegotiatePacketSize()
        # Configure device streaming destination
        device.SetStreamDestination(stream.GetLocalIPAddress(), stream.GetLocalPort())


def configure_stream_buffers(device, stream, set_buffer_count=1):
    buffer_list = []
    # Reading payload size from device
    size = device.GetPayloadSize()

    # Use BUFFER_COUNT or the maximum number of buffers, whichever is smaller
    buffer_count = stream.GetQueuedBufferMaximum()
    if buffer_count > set_buffer_count:
        buffer_count = set_buffer_count

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
def get_data(pvbuffer, decompression_filter, payload_type):
    errors = 0
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
                calculated_size = eb.PvImage.GetPixelSize(pixel_type) * width * height / 8
                out_buffer = eb.PvBuffer()
                result, decompressed_buffer = decompression_filter.Execute(pvbuffer, out_buffer)
                image = decompressed_buffer.GetImage()
                if result.IsOK():
                    decompressed_size = decompressed_buffer.GetSize()
                    compression_ratio = decompressed_size / pvbuffer.GetAcquiredSize()
                    if calculated_size != decompressed_size:
                        errors = errors + 1
                    print(
                        f" Pleora compressed type.   Compression ratio: {'{0:.2f}'.format(compression_ratio)}"
                        f" Errors: {errors}", end='',
                    )
                else:
                    print(" Could not decompress (Pleora compressed)", end='')
                    errors = errors + 1
            else:
                print(" Could not read header (Pleora compressed)", end='')
                errors = errors + 1
        else:
            print(" Contents do not match payload type (Pleora compressed)", end='')
            errors = errors + 1

    else:
        print(" Payload type not supported by this sample", end='')
    return image


# Defino una función que chequee si el buffer adquirió un resultado útil o no
def buffer_check(result, doodle, doodle_index, operational_result):
    if not result.IsOK():
        print(f"{doodle[doodle_index]} {result.GetCodeString()}      ", end='\r')
        # raise Exception("Buffer no adquirido")
        warnings.warn('Buffer no adquirido')
        return False
    if not operational_result.IsOK():
        print(f"{doodle[doodle_index]} {operational_result.GetCodeString()}       ", end='\r')
        # raise Exception("Buffer mal adquirido")
        warnings.warn('Buffer mal adquirido')
        return False
    else:
        return True
