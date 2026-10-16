# Air Piano

Air Piano es un instrumento en tiempo real controlado mediante webcam, utilizando gestos de flexión de dedos y cierre de puño. MediaPipe Hand Landmarker proporciona 21 puntos de referencia (*landmarks*) para cada mano detectada; OpenCV dibuja el esqueleto completo de la mano, las notas asignadas, el estado de los gestos y una octava de piano centrada con fines exclusivamente visuales.

## Arquitectura

```text
main.py
  ├── vision_worker.py            Hilo de captura de cámara e inferencia con MediaPipe
  ├── hand_tracking/
  │     ├── detector.py           Adaptador de MediaPipe Hand Landmarker
  │     └── gesture_state_manager.py
  │                                  Suavizado de landmarks, flexión, detección
  │                                  de puños, debounce y estado de gestos
  ├── piano/
  │     ├── note_mapper.py        Mapeo fijo de gestos a notas MIDI
  │     └── keyboard.py           Teclado visual de una octava
  ├── ui/renderer.py              Superposición de esqueleto y retroalimentación visual
  └── audio/midi_engine.py        Hilo de MIDI polifónico / sintetizador en cola
```

El reconocimiento de gestos utiliza ángulos articulares calculados a partir de landmarks suavizados, no la posición de la mano en pantalla ni colisiones con las teclas mostradas. El motor de audio y la tubería de procesamiento de cámara funcionan en hilos independientes; el diseño basado en el último fotograma evita la acumulación de retraso cuando la inferencia tarda más que el intervalo entre capturas de cámara.

## Asignación fija de notas

Con `keyboard.start_octave: 3`, el mapeo cubre una octava desde C3 hasta B3:

| Gesto | Nota | MIDI |
|--------|------|------|
| Pulgar / índice / medio / anular / meñique izquierdo | C / C# / D / D# / E | 48–52 |
| Pulgar / índice / medio / anular / meñique derecho | F / F# / G / G# / A | 53–57 |
| Puño cerrado izquierdo | A# | 58 |
| Puño cerrado derecho | B | 59 |

Las notas se activan cuando un dedo pasa del estado extendido al estado flexionado y permanecen sonando hasta que el dedo vuelve a extenderse.

Un puño se detecta cuando al menos cuatro de los cinco dedos están flexionados. Cuando se detecta un puño, cualquier nota mantenida por los dedos de esa mano se libera para evitar la activación accidental de acordes. Al abrir la mano, la nota asociada al puño se libera.

Cada gesto tiene un intervalo mínimo de reactivación de 100 ms.

La flexión se clasifica a partir del ángulo tridimensional en la articulación IP del pulgar o en la articulación PIP del resto de dedos. Se utilizan umbrales distintos para la flexión y la extensión a fin de proporcionar histéresis; además, se emplean fotogramas estables consecutivos y una ventana temporal móvil para filtrar pequeñas fluctuaciones en los landmarks.

Estos ángulos geométricos articulares son invariantes respecto a la posición de la mano en la imagen capturada por la cámara.

## Instalación

Se recomienda utilizar Python 3.11 o superior. Las versiones 3.11 y 3.12 ofrecen una disponibilidad fiable de paquetes precompilados para visión por computador. El modelo MediaPipe Hand Landmarker ya está incluido en `assets/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para la reproducción mediante SoundFont con FluidSynth en Debian/Ubuntu, instala la biblioteca nativa correspondiente:

```bash
sudo apt install libfluidsynth3
```

Configura una ruta válida a un archivo `.sf2` con licencia adecuada dentro de `config.yaml` para utilizar un SoundFont.

Si `audio.soundfont` está configurado como `null`, Air Piano utilizará una salida MIDI del sistema cuando esté disponible; de lo contrario, recurrirá a su sintetizador polifónico integrado.

## Ejecución y pruebas

```bash
python main.py
python -m unittest discover -s tests -v
```

Pulsa `Q` o `Esc` en la ventana de la cámara para salir.

Utiliza:

```bash
python main.py --config ruta/al/config.yaml
```

para seleccionar un archivo de configuración diferente.

## Configuración

Edita `config.yaml`:

- `keyboard.start_octave`: octava MIDI que se muestra y reproduce; las teclas son únicamente visuales. `width_ratio`, `top` y `height` controlan la geometría de visualización centrada.
- `gesture.flexion_angle_degrees` y `extension_angle_degrees`: umbrales de histéresis. El ángulo de flexión debe ser menor que el de extensión.
- `gesture.stable_frames`: número de observaciones consecutivas necesarias para aceptar un estado.
- `gesture.debounce_ms`: intervalo mínimo entre activaciones; se rechazan valores inferiores a 100 ms.
- `gesture.visual_hold_ms`: duración del resaltado visual de una tecla tras su activación.
- `filters.landmark_window_size`: tamaño de la ventana de media móvil aplicada a todos los landmarks.
- `audio.soundfont`, `video.*`: configuración local de audio y webcam.

## Rendimiento y limitaciones

Los FPS solicitados a la cámara son una sugerencia para el controlador del dispositivo y no una garantía. Si la inferencia no puede mantener el ritmo, reduce la resolución de la cámara.

Los eventos de audio se colocan en una cola para el hilo de audio, de modo que la síntesis sonora no bloquee el seguimiento de las manos.

La latencia de extremo a extremo depende de factores como:

- Tiempo de exposición de la cámara.
- Frecuencia de captura de fotogramas.
- Tiempo de inferencia.
- Planificación de hilos del sistema operativo.
- Backend de audio utilizado.

La aplicación no puede garantizar una latencia inferior a 20 ms.