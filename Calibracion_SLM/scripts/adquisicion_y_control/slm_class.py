import numpy as np
import HEDS
from hedslib.heds_types import *

class holoeye_SLM:
    def __init__(self, version=(4,0), preview=True):
        # Inicializo el SDK
        err = HEDS.SDK.Init(*version)
        assert err == HEDS.HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        # Inicializo el SLM
        self.slm = HEDS.SLM.Init("", preview, 0.0)
        assert self.slm.errorCode() == HEDS.HEDSERR_NoError, HEDS.SDK.ErrorString(self.slm.errorCode())

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

