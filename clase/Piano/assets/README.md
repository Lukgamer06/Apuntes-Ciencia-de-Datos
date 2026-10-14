# Models and SoundFonts

`hand_landmarker.task` is the MediaPipe Hand Landmarker model used by the video
pipeline. It is distributed from the official MediaPipe model storage:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

For audio, place a SoundFont you are licensed to use here and set
`audio.soundfont` in `config.yaml` to its path (for example,
`assets/piano.sf2`). SoundFonts are not bundled. With no SoundFont configured,
the application uses the operating system's default MIDI output device.
