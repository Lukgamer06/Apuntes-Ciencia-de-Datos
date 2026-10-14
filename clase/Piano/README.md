# Air Piano

Air Piano is a real-time webcam instrument controlled by finger-flexion and fist
gestures. MediaPipe Hand Landmarker supplies 21 landmarks for each detected
hand; OpenCV draws the full hand skeleton, assigned note labels, gesture status,
and a centered, visual-only piano octave.

## Architecture

```text
main.py
  ├── vision_worker.py            Camera capture + MediaPipe inference thread
  ├── hand_tracking/
  │     ├── detector.py            MediaPipe Hand Landmarker adapter
  │     └── gesture_state_manager.py
  │                                  Landmark smoothing, flexion, fist edges,
  │                                  debounce and gesture state
  ├── piano/
  │     ├── note_mapper.py         Fixed gesture-to-MIDI mapping
  │     └── keyboard.py            Visual-only one-octave keyboard
  ├── ui/renderer.py               Skeleton and feedback overlay
  └── audio/midi_engine.py          Queued polyphonic MIDI / synth thread
```

Gesture recognition uses joint angles from smoothed landmarks, not the hand's
screen location or collisions with displayed keys. The audio engine and camera
pipeline remain on independent worker threads; the latest-frame design avoids
building a backlog when inference takes longer than a camera frame.

## Fixed note assignment

At `keyboard.start_octave: 3`, the mapping is one octave from C3 through B3:

| Gesture | Note | MIDI |
|---|---:|---:|
| Left thumb / index / middle / ring / pinky | C / C# / D / D# / E | 48–52 |
| Right thumb / index / middle / ring / pinky | F / F# / G / G# / A | 53–57 |
| Left closed fist | A# | 58 |
| Right closed fist | B | 59 |

Notes activate on an extended-to-flexed transition and remain held until the
finger extends again. A fist is detected when at least four of its five fingers
are flexed. When a fist takes over, any held finger notes on that hand are
released to prevent accidental chord activation. Opening the hand releases the
fist note. Each gesture has a 100 ms minimum retrigger interval.

Flexion is classified from the 3D angle at the thumb IP joint or each other
finger's PIP joint. Separate flexion and extension angles provide hysteresis;
consecutive stable frames and a temporal moving window reject brief landmark
jitter. These geometric joint angles are invariant to where the hand is in the
camera image.

## Installation

Python 3.11 or newer is recommended; Python 3.11/3.12 is a reliable choice for
computer-vision wheel availability. The MediaPipe Hand Landmarker model is
included in `assets/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For FluidSynth SoundFont playback on Debian/Ubuntu, install its native library:

```bash
sudo apt install libfluidsynth3
```

Set a locally available, licensed `.sf2` path in `config.yaml` to use a
SoundFont. With `audio.soundfont: null`, Air Piano uses a system MIDI output
when available, otherwise its built-in polyphonic synthesizer.

## Run and tests

```bash
python main.py
python -m unittest discover -s tests -v
```

Press `Q` or `Esc` in the camera window to exit. Use `python main.py --config
path/to/config.yaml` to select a different configuration.

## Configuration

Edit `config.yaml`:

- `keyboard.start_octave`: MIDI octave rendered and played; the keys are visual
  only. `width_ratio`, `top` and `height` control its centered display geometry.
- `gesture.flexion_angle_degrees` and `extension_angle_degrees`: hysteresis
  thresholds. The flexion angle must be lower than the extension angle.
- `gesture.stable_frames`: consecutive observations required to accept a state.
- `gesture.debounce_ms`: minimum activation gap; values below 100 ms are rejected.
- `gesture.visual_hold_ms`: duration of the post-trigger key highlight.
- `filters.landmark_window_size`: moving-average window for all landmarks.
- `audio.soundfont`, `video.*`: local audio and webcam settings.

## Performance and limitations

The camera's requested FPS is a driver hint, not a guarantee. Reduce camera
resolution if inference cannot keep up. Audio events are queued to the audio
thread so synthesis does not block tracking. End-to-end latency depends on
camera exposure, frame rate, inference, scheduling, and the audio backend;
sub-20 ms latency cannot be guaranteed by the application.
