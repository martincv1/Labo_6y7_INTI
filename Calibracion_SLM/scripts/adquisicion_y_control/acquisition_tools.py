# Importo librerias
import eBUS as eb
import lib.PvSampleUtils as psu
import warnings
import time
import numpy as np


class jai_camera:
    def __init__(self, buffers=1, verbose=False, n_retry_retrieve=5, retry_wait_time=0.5):
        self.buffers = buffers
        self.verbose = verbose
        self.n_retry_retrieve = n_retry_retrieve
        self.retry_wait_time = retry_wait_time
        self.device: eb.PvDeviceGEV
        self.stream: eb.PvStreamGEV

        self.kb = psu.PvKb()
        self._initt()
        # Termino de configurar algunas cosas
        self._configure_stream()
        self._configure_stream_buffers()

        # Primero agarro algunas propiedades de Genicam
        # Get device parameters need to control streaming
        self.device_params = self.device.GetParameters()
        # Map the GenICam AcquisitionStart and AcquisitionStop commands
        self.start = self.device_params.Get("AcquisitionStart")
        self.stop = self.device_params.Get("AcquisitionStop")
        # Get stream parameters
        self.stream_params = self.stream.GetParameters()
        # Map a few GenICam stream stats counters
        self.frame_rate = self.stream_params.Get("AcquisitionRate")
        self.bandwidth = self.stream_params["Bandwidth"]
        # Enable streaming and send the AcquisitionStart command
        self.print("Enabling streaming and sending AcquisitionStart command.")
        self.device.StreamEnable()
        self.start.Execute()
        # Por las dudas pongo un sleep
        time.sleep(2)

        self.doodle = "|\\-|-/"
        self.doodle_index = 0
        self.errors = 0
        self.decompression_filter = eb.PvDecompressionFilter()

    def _connect_to_device(self):
        # Connect to the GigE Vision or USB3 Vision device
        self.print("Connecting to device.")
        result, self.device = eb.PvDevice.CreateAndConnect(self.connection_ID)
        if not self.device:
            raise Exception(f"Unable to connect to device: {result.GetCodeString()} ({result.GetDescription()})")

    def _open_stream(self):
        # Open stream to the GigE Vision or USB3 Vision device
        self.print("Opening stream from device.")
        result, self.stream = eb.PvStream.CreateAndOpen(self.connection_ID)
        if not self.stream:
            raise Exception(f"Unable to stream from device. {result.GetCodeString()} ({result.GetDescription()})")

    def _configure_stream(self):
        # If this is a GigE Vision device, configure GigE Vision specific streaming parameters
        if isinstance(self.device, eb.PvDeviceGEV):
            # Negotiate packet size
            self.device.NegotiatePacketSize()
            # Configure device streaming destination
            self.device.SetStreamDestination(self.stream.GetLocalIPAddress(), self.stream.GetLocalPort())

    def _configure_stream_buffers(self):
        self.buffer_list = []
        # Reading payload size from device
        size = self.device.GetPayloadSize()

        # Use BUFFER_COUNT or the maximum number of buffers, whichever is smaller
        buffer_count = self.stream.GetQueuedBufferMaximum()
        if buffer_count > self.buffers:
            buffer_count = self.buffers

        # Allocate buffers
        for i in range(buffer_count):
            # Create new pvbuffer object
            pvbuffer = eb.PvBuffer()
            # Have the new pvbuffer object allocate payload memory
            pvbuffer.Alloc(size)
            # Add to external list - used to eventually release the buffers
            self.buffer_list.append(pvbuffer)

        # Queue all buffers in the stream
        for pvbuffer in self.buffer_list:
            self.stream.QueueBuffer(pvbuffer)
        self.print(f"Created {buffer_count} buffers", override_verbose=buffer_count != self.buffers)

    def _initt(self):
        # Conectar camara, empezar stream y chequear que todo conectó o empezó bien
        self.connection_ID = psu.PvSelectDevice()
        if not self.connection_ID:
            raise Exception("Error al seleccionar ID")
        self._connect_to_device()
        self._open_stream()

    def _get_data(self, pvbuffer, payload_type):
        # Chequeo el tipo de data que saca del buffer y adquiere una imagen
        if payload_type == eb.PvPayloadTypeImage:
            image = pvbuffer.GetImage()

        elif payload_type == eb.PvPayloadTypeChunkData:
            self.print(f" Chunk Data payload type with {pvbuffer.GetChunkCount()} chunks")

        elif payload_type == eb.PvPayloadTypeRawData:
            self.print(f" Raw Data with {pvbuffer.GetRawData().GetPayloadLength()} bytes")

        elif payload_type == eb.PvPayloadTypeMultiPart:
            self.print(f" Multi Part with {pvbuffer.GetMultiPartContainer().GetPartCount()} parts")

        elif payload_type == eb.PvPayloadTypePleoraCompressed:
            if eb.PvDecompressionFilter.IsCompressed(pvbuffer):
                result, pixel_type, width, height = eb.PvDecompressionFilter.GetOutputFormatFor(pvbuffer)
                if result.IsOK():
                    calculated_size = eb.PvImage.GetPixelSize(pixel_type) * width * height / 8
                    out_buffer = eb.PvBuffer()
                    result, decompressed_buffer = self.decompression_filter.Execute(pvbuffer, out_buffer)
                    image = decompressed_buffer.GetImage()
                    if result.IsOK():
                        decompressed_size = decompressed_buffer.GetSize()
                        compression_ratio = decompressed_size / pvbuffer.GetAcquiredSize()
                        if calculated_size != decompressed_size:
                            self.errors = self.errors + 1
                        self.print(
                            f" Pleora compressed type.   Compression ratio: {'{0:.2f}'.format(compression_ratio)}"
                            f" Errors: {self.errors}",
                        )
                    else:
                        self.print(" Could not decompress (Pleora compressed)", override_verbose=True)
                        self.errors = self.errors + 1
                else:
                    self.print(" Could not read header (Pleora compressed)", override_verbose=True)
                    self.errors = self.errors + 1
            else:
                self.print(" Contents do not match payload type (Pleora compressed)", override_verbose=True)
                self.errors = self.errors + 1

        else:
            self.print(" Payload type not supported by this sample", override_verbose=True)
        return image

    def buffer_check(self, result, operational_result):
        # Chequeo si el buffer adquirió un resultado útil o no
        if not result.IsOK():
            self.print(f"{result.GetCodeString()}      ", doodle_it=True, override_verbose=True)
            warnings.warn('Buffer no adquirido')
            return False
        if not operational_result.IsOK():
            self.print(f"{operational_result.GetCodeString()}       ", doodle_it=True, override_verbose=True)
            warnings.warn('Buffer mal adquirido')
            return False
        return True

    def print(self, message, override_verbose=False, doodle_it=False, *args, **kwargs):
        if self.verbose or override_verbose:
            if doodle_it:
                print(f"{self.doodle[self.doodle_index]} {message}", end='\r', *args, **kwargs)
                self._roll_doodle()
            else:
                print(message, *args, **kwargs)

    def _roll_doodle(self):
        self.doodle_index = (self.doodle_index + 1) % 6

    def get_frame(self, do_queue=True):
        # Retrieve next pvbuffer
        tries = 0
        while tries <= self.n_retry_retrieve:
            result, pvbuffer, operational_result = self.stream.RetrieveBuffer(1000)
            print("Buffer retrieved")
            if self.buffer_check(result, operational_result):
                break
            self.stream.QueueBuffer(pvbuffer)
            time.sleep(self.retry_wait_time)
            tries += 1

        # We now have a valid pvbuffer. This is where you would typically process the pvbuffer.
        result, self.frame_rate_val = self.frame_rate.GetValue()
        result, self.bandwidth_val = self.bandwidth.GetValue()
        self.print(f"BlockID: {pvbuffer.GetBlockID():016d}", doodle_it=True)

        image = None
        payload_type = pvbuffer.GetPayloadType()

        image = self._get_data(pvbuffer, payload_type)  # acá consigue la imagen

        if image:
            image_data = image.GetDataPointer()

        if do_queue:
            # Re-queue the pvbuffer in the stream object
            self.stream.QueueBuffer(pvbuffer)  # Acá manda el buffer a buscar

        return image_data

    def get_average_frame(self, n_average, do_queue=True):
        image_sum = None

        for i in range(n_average):
            do_this_queue = (i != n_average - 1) or do_queue
            image_data = self.get_frame(do_queue=do_this_queue)
            if image_sum is None:
                image_sum = np.array(image_data, dtype=np.float64)
            else:
                image_sum += image_data

        average_image = (image_sum / n_average).astype(image_data.dtype)
        return average_image

    def close(self):
        # Tell the device to stop sending images.
        self.print("\nSending AcquisitionStop command to the device")
        self.stop.Execute()

        # Disable streaming on the device
        self.print("Disable streaming on the controller.")
        self.device.StreamDisable()

        # Abort all buffers from the stream and dequeue
        self.print("Aborting buffers still in stream")
        self.stream.AbortQueuedBuffers()
        while self.stream.GetQueuedBufferCount() > 0:
            result, pvbuffer, lOperationalResult = self.stream.RetrieveBuffer()

    def reset_queue(self):
        self.stream.AbortQueuedBuffers()
        pvbuffer = None
        while self.stream.GetQueuedBufferCount() > 0:
            result, pvbuffer, lOperationalResult = self.stream.RetrieveBuffer()

        if pvbuffer:
            self.stream.QueueBuffer(pvbuffer)
