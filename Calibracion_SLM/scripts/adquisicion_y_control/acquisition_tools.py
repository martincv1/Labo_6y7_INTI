# Importo librerias
import eBUS as eb
import lib.PvSampleUtils as psu
import warnings
import time
import numpy as np

# las del SLM
import HEDS
from hedslib.heds_types import *


class holoeye_SLM:
    def __init__(self, version=(4, 0), preview=True):
        # Inicializo el SDK
        err = HEDS.SDK.Init(*version)
        assert err == HEDS.HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        # Inicializo el SLM
        self.slm = HEDS.SLM.Init("", preview, 0.0)
        assert self.slm.errorCode() == HEDS.HEDSERR_NoError, HEDS.SDK.ErrorString(
            self.slm.errorCode()
        )

    def crear_patron(self, resolucion, orientacion, mitad, intensidad):
        grayscale_array = np.zeros(resolucion, dtype=np.uint8)

        if orientacion == "horizontal":
            half_height = resolucion[0] // 2
            if mitad == "sup":
                grayscale_array[:half_height, :] = intensidad
            elif mitad == "inf":
                grayscale_array[half_height:, :] = intensidad

        elif orientacion == "vertical":
            half_width = resolucion[1] // 2
            if mitad == "izq":
                grayscale_array[:, :half_width] = intensidad
            elif mitad == "der":
                grayscale_array[:, half_width:] = intensidad

        return grayscale_array

    def mostrar_patron(self, patron):
        err, dataHandle = self.slm.loadImageData(patron)
        assert err == HEDS.HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        err = dataHandle.show()
        assert err == HEDS.HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    def close(self):
        """Liberar recursos del SLM (si el SDK lo requiere)."""
        self.slm = None
        HEDS.SDK.Exit()


class jai_camera:
    def __init__(
        self, buffers=1, verbose=False, n_retry_retrieve=5, retry_wait_time=0.5
    ):
        self.buffers = buffers
        self.verbose = verbose
        self.n_retry_retrieve = n_retry_retrieve
        self.retry_wait_time = retry_wait_time
        self.wait_after_start = 0.5
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
        self.start: eb.PvGenCommand = self.device_params.Get("AcquisitionStart")
        self.stop: eb.PvGenCommand = self.device_params.Get("AcquisitionStop")
        # Get stream parameters
        self.stream_params: eb.PvGenParameterArray = self.stream.GetParameters()
        # Map a few GenICam stream stats counters
        self.frame_rate: eb.PvGenParameter = self.stream_params["AcquisitionRate"]
        self.bandwidth: eb.PvGenParameter = self.stream_params["Bandwidth"]
        # Enable streaming and send the AcquisitionStart command
        self.print("Enabling streaming and sending AcquisitionStart command.")
        result = self.device.StreamEnable()
        self._result_ok_or_error(result, "Unable to enable streaming.")
        result = self.start.Execute()
        self._result_ok_or_error(result, "Unable to start acquisition.")

        # Por las dudas pongo un sleep
        time.sleep(self.wait_after_start)

        self.doodle = "|\\-|-/"
        self.doodle_index = 0
        self.errors = 0
        self.decompression_filter = eb.PvDecompressionFilter()

    def _result_ok_or_error(self, result: eb.PvResult, message: str, check=None):
        if not check:
            if not result.IsOK():
                raise Exception(
                    f"{message}: {result.GetCodeString()} ({result.GetDescription()})"
                )
        elif result.GetCode() != check:
            raise Exception(
                f"{message}: {result.GetCodeString()} ({result.GetDescription()})"
            )

    def _connect_to_device(self):
        # Connect to the GigE Vision or USB3 Vision device
        self.print("Connecting to device.")
        result, self.device = eb.PvDevice.CreateAndConnect(self.connection_ID)
        self._result_ok_or_error(result, "Unable to connect to device")

    def _open_stream(self):
        # Open stream to the GigE Vision or USB3 Vision device
        self.print("Opening stream from device.")
        result, self.stream = eb.PvStream.CreateAndOpen(self.connection_ID)
        self._result_ok_or_error(result, "Unable to open stream from device")

    def _configure_stream(self):
        # If this is a GigE Vision device, configure GigE Vision specific streaming parameters
        if isinstance(self.device, eb.PvDeviceGEV):
            # Negotiate packet size
            result = self.device.NegotiatePacketSize()
            self._result_ok_or_error(result, "Unable to negotiate packet size")
            # Configure device streaming destination
            result = self.device.SetStreamDestination(
                self.stream.GetLocalIPAddress(), self.stream.GetLocalPort()
            )
            self._result_ok_or_error(result, "Unable to set stream destination")

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
            result = pvbuffer.Alloc(size)
            self._result_ok_or_error(result, "Unable to allocate payload memory")
            # Add to external list - used to eventually release the buffers
            self.buffer_list.append(pvbuffer)

        # Queue all buffers in the stream
        for pvbuffer in self.buffer_list:
            result = self.stream.QueueBuffer(pvbuffer)
            self._result_ok_or_error(
                result, "Unable to queue buffer", check=eb.PV_PENDING
            )
        self.print(
            f"Created {buffer_count} buffers",
            override_verbose=buffer_count != self.buffers,
        )

    def _initt(self):
        # Conectar camara, empezar stream y chequear que todo conectó o empezó bien
        self.connection_ID = psu.PvSelectDevice()
        if not self.connection_ID:
            raise Exception("Error al seleccionar ID")
        self._connect_to_device()
        self._open_stream()

    def _get_data(self, pvbuffer: eb.PvBuffer, payload_type):
        # Chequeo el tipo de data que saca del buffer y adquiere una imagen
        if payload_type == eb.PvPayloadTypeImage:
            image = pvbuffer.GetImage()

        elif payload_type == eb.PvPayloadTypeChunkData:
            self.print(
                f" Chunk Data payload type with {pvbuffer.GetChunkCount()} chunks"
            )

        elif payload_type == eb.PvPayloadTypeRawData:
            self.print(
                f" Raw Data with {pvbuffer.GetRawData().GetPayloadLength()} bytes"
            )

        elif payload_type == eb.PvPayloadTypeMultiPart:
            self.print(
                f" Multi Part with {pvbuffer.GetMultiPartContainer().GetPartCount()} parts"
            )

        elif payload_type == eb.PvPayloadTypePleoraCompressed:
            if eb.PvDecompressionFilter.IsCompressed(pvbuffer):
                result, pixel_type, width, height = (
                    eb.PvDecompressionFilter.GetOutputFormatFor(pvbuffer)
                )
                if result.IsOK():
                    calculated_size = (
                        eb.PvImage.GetPixelSize(pixel_type) * width * height / 8
                    )
                    out_buffer = eb.PvBuffer()
                    result, decompressed_buffer = self.decompression_filter.Execute(
                        pvbuffer, out_buffer
                    )
                    image = decompressed_buffer.GetImage()
                    if result.IsOK():
                        decompressed_size = decompressed_buffer.GetSize()
                        compression_ratio = (
                            decompressed_size / pvbuffer.GetAcquiredSize()
                        )
                        if calculated_size != decompressed_size:
                            self.errors = self.errors + 1
                        self.print(
                            f" Pleora compressed type.   Compression ratio: {'{0:.2f}'.format(compression_ratio)}"
                            f" Errors: {self.errors}",
                        )
                    else:
                        self.print(
                            " Could not decompress (Pleora compressed)",
                            override_verbose=True,
                        )
                        self.errors = self.errors + 1
                else:
                    self.print(
                        " Could not read header (Pleora compressed)",
                        override_verbose=True,
                    )
                    self.errors = self.errors + 1
            else:
                self.print(
                    " Contents do not match payload type (Pleora compressed)",
                    override_verbose=True,
                )
                self.errors = self.errors + 1

        else:
            self.print(
                " Payload type not supported by this sample", override_verbose=True
            )
        return image

    def buffer_check(self, result: eb.PvResult, operational_result: eb.PvResult):
        # Chequeo si el buffer adquirió un resultado útil o no
        if not result.IsOK():
            self.print(
                f"{result.GetCodeString()}      ", doodle_it=True, override_verbose=True
            )
            warnings.warn("Buffer no adquirido")
            return False
        if not operational_result.IsOK():
            self.print(
                f"{operational_result.GetCodeString()}       ",
                doodle_it=True,
                override_verbose=True,
            )
            warnings.warn("Buffer mal adquirido")
            return False
        return True

    def print(self, message, override_verbose=False, doodle_it=False, *args, **kwargs):
        if self.verbose or override_verbose:
            if doodle_it:
                print(
                    f"{self.doodle[self.doodle_index]} {message}",
                    end="\r",
                    *args,
                    **kwargs,
                )
                self._roll_doodle()
            else:
                print(message, *args, **kwargs)

    def _roll_doodle(self):
        self.doodle_index = (self.doodle_index + 1) % 6

    def get_frame(self, do_queue=True):
        # Retrieve next pvbuffer
        tries = 0
        result: eb.PvResult
        pvbuffer: eb.PvBuffer
        operational_result: eb.PvResult
        while tries <= self.n_retry_retrieve:
            result, pvbuffer, operational_result = self.stream.RetrieveBuffer(1000)
            self.print("Buffer retrieved")
            if self.buffer_check(result, operational_result):
                break
            result = self.stream.QueueBuffer(pvbuffer)
            self._result_ok_or_error(
                result, "Unable to queue buffer", check=eb.PV_PENDING
            )
            time.sleep(self.retry_wait_time)
            tries += 1

        # We now have a valid pvbuffer. This is where you would typically process the pvbuffer.
        result, self.frame_rate_val = self.frame_rate.GetValue()
        self._result_ok_or_error(result, "Unable to read frame rate value.")
        result, self.bandwidth_val = self.bandwidth.GetValue()
        self._result_ok_or_error(result, "Unable to read bandwidth value.")
        self.print(f"BlockID: {pvbuffer.GetBlockID():016d}", doodle_it=True)

        image = None
        payload_type = pvbuffer.GetPayloadType()

        image = self._get_data(pvbuffer, payload_type)  # acá consigue la imagen

        if image:
            image_data = image.GetDataPointer()

        if do_queue:
            # Re-queue the pvbuffer in the stream object
            result = self.stream.QueueBuffer(pvbuffer)  # Acá manda el buffer a buscar
            self._result_ok_or_error(
                result, "Unable to queue buffer", check=eb.PV_PENDING
            )

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

    def get_multiple_frame(self, cantidad, do_queue=True):
        imgs = []
        for i in range(cantidad):
            do_this_queue = (i != cantidad - 1) or do_queue
            image_data = self.get_frame(do_queue=do_this_queue)
            imgs.append(image_data)

        return imgs

    def close(self):
        # Tell the device to stop sending images.
        self.print("\nSending AcquisitionStop command to the device")
        result = self.stop.Execute()
        self._result_ok_or_error(result, "AcquisitionStop command failed")

        # Disable streaming on the device
        self.print("Disable streaming on the controller.")
        result = self.device.StreamDisable()
        self._result_ok_or_error(result, "StreamDisable command failed")

        # Abort all buffers from the stream and dequeue
        self.print("Aborting buffers still in stream")
        result = self.stream.AbortQueuedBuffers()
        self._result_ok_or_error(result, "AbortQueuedBuffers command failed")

        pvbuffer = None
        while self.stream.GetQueuedBufferCount() > 0:
            result, pvbuffer, lOperationalResult = self.stream.RetrieveBuffer()

    def reset_queue(self):
        result = self.stream.AbortQueuedBuffers()
        self._result_ok_or_error(result, "Unable to abort queued buffers.")
        pvbuffer = None
        while self.stream.GetQueuedBufferCount() > 0:
            result, pvbuffer, lOperationalResult = self.stream.RetrieveBuffer()

        if pvbuffer:
            result = self.stream.QueueBuffer(pvbuffer)
            self._result_ok_or_error(
                result, "Unable to queue buffer", check=eb.PV_PENDING
            )
