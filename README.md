# Labo 6/7 — Calibración de un SLM para topografía óptica

Repositorio de la materia **Laboratorio 6 y 7** de la Facultad de Ciencias Exactas y Naturales
(FCEN, UBA), desarrollado en el **INTI** (Instituto Nacional de Tecnología Industrial).

El objetivo del proyecto es aplicar métodos de análisis (y eventualmente
inteligencia artificial) para mejorar la medición de topografía en planos ópticos y patrones de
rugosidad. Esta primera etapa, la que vive en este repositorio, se centra en **calibrar un
modulador espacial de luz (SLM)**: entender con qué precisión y con qué incertidumbre se puede
usar como referencia controlada antes de aplicar el mismo método a superficies reales.

## Equipo

Proyecto desarrollado en conjunto por **Martín Castelli** y **Juan Marino**, bajo la supervisión
de **Pablo Etchepareborda** y **Agustina Viaggio** en el INTI.

## SLM como dataset programable

Un SLM reflectivo (HOLOEYE Pluto VIS020) modula la fase de la luz que se refleja en él: el nivel
de gris que se le manda a cada píxel cambia el índice de refracción del cristal líquido, y por
lo tanto el camino óptico que recorre la luz. Ese es el mismo efecto que produce un
desnivel de altura en una superficie reflectante real, por eso el SLM sirve como aparato programable para un método que después se puede aplicar a defectos de topografía en
muestras reales.

El experimento programa el SLM con dos mitades a distinto nivel de gris, se capturan las
franjas de interferencia resultantes, y se extrae la **diferencia de fase entre ambas mitades**
a partir de la imagen. Esa fase, vía la longitud de onda del láser, se traduce a una diferencia
de altura equivalente (`Δh = (Δφ / 2π) · (λ / 2)`). Repitiendo
esto para los 256 niveles de gris se obtiene la curva de calibración del SLM (gris → fase), que
es el input necesario para corregir su respuesta.

## Scripts más relevantes

El método de extracción de fase usa una transformada de Hilbert: se aísla en Fourier la
frecuencia portadora de las franjas, se filtra, se toma la señal y se ajusta la fase
resultante. Los scripts centrales de ese pipeline, en `Calibracion_SLM/scripts/analisis/`:

- **`funciones.py`** — librería compartida. `fase_hilbert` implementa el método (FFT → filtro
  gaussiano en la frecuencia portadora → transformada de Hilbert → ajuste lineal de fase).
  `simular_imagen` genera interferogramas sintéticos con fase conocida, usados como referencia
  para validar el método.
- **`incertidumbres.py`** — valida `fase_hilbert` con una simulación Monte Carlo: genera pares de
  imágenes con una diferencia de fase impuesta y conocida, mide esa diferencia con el método, y
  reporta el error (RMSD) entre lo impuesto y lo medido.
- **`fase_tension.py`** — aplica el método a los datos experimentales: por cada nivel de gris,
  valida la calidad espectral de cada muestra, promedia repeticiones como fasores y agrega
  varias corridas de repetibilidad para construir la curva final de fase vs. nivel de gris, con
  su incertidumbre.
- **`gamma_curver.py`** — clase `GammaCurver`: toma la curva de fase medida y la tabla de gamma
  actual del SLM, y produce una curva de corrección suavizada y monótona para cargar al SLM.

Los scripts de `Calibracion_SLM/scripts/adquisicion_y_control/` controlan el hardware (SLM vía
HOLOEYE SLM Display SDK, cámara JAI vía eBUS SDK) para capturar las imágenes. Requieren esos SDKs
del fabricante, que no están incluidos en el repositorio.

## Datos

`Calibracion_SLM/data/` está vacío a propósito — las imágenes y mediciones (`.png`, `.pkl`,
`.csv`) están excluidas del repositorio por tamaño (ver `.gitignore`).

## Cómo correr los scripts de análisis

```bash
pip install -r Calibracion_SLM/requirements.txt
```

Los scripts de `scripts/analisis/` son código de experimentación: cada uno tiene sus parámetros
(rutas, coordenadas de recorte, rangos de intensidad) definidos como constantes al principio del
archivo, pensados para editarse por corrida en vez de recibir argumentos por línea de comandos.
`funciones.py`, `incertidumbres.py` y `gamma_curver.py` no dependen de datos externos y se pueden
correr directamente para ver el método en acción sobre imágenes simuladas.
