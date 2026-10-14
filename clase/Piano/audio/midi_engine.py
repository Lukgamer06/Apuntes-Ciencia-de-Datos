"""Dedicated audio thread for non-blocking polyphonic MIDI events."""

from __future__ import annotations

import logging
import queue
import threading
from collections import Counter
from pathlib import Path
from typing import Protocol

from config import AudioConfig
from piano.note_mapper import NoteEvent

LOGGER = logging.getLogger(__name__)


class _FluidSynth(Protocol):
    def sfload(self, filename: str, reset_presets: bool = ...) -> int: ...
    def program_select(self, channel: int, soundfont_id: int, bank: int, preset: int) -> None: ...
    def start(self, driver: str | None = ...) -> None: ...
    def noteon(self, channel: int, key: int, velocity: int) -> None: ...
    def noteoff(self, channel: int, key: int) -> None: ...
    def delete(self) -> None: ...


class _MidiOutput(Protocol):
    def note_on(self, note: int, velocity: int, channel: int = ...) -> None: ...
    def note_off(self, note: int, velocity: int = ..., channel: int = ...) -> None: ...
    def close(self) -> None: ...


class _PygameSynth:
    """Small local polyphonic fallback that does not need MIDI or a SoundFont."""

    def __init__(self, sample_rate: int) -> None:
        import pygame

        self._pygame = pygame
        self._owns_mixer = pygame.mixer.get_init() is None
        if self._owns_mixer:
            pygame.mixer.pre_init(
                frequency=sample_rate, size=-16, channels=1, buffer=256
            )
            pygame.mixer.init()
        mixer_settings = pygame.mixer.get_init()
        if mixer_settings is None:
            raise RuntimeError("pygame.mixer did not initialize an audio device.")
        self._sample_rate = mixer_settings[0]
        pygame.mixer.set_num_channels(max(pygame.mixer.get_num_channels(), 32))
        self._sounds: dict[int, object] = {}
        self._channels: dict[int, object] = {}

    def note_on(self, note: int, velocity: int) -> None:
        import numpy as np

        channel = self._pygame.mixer.find_channel()
        if channel is None:
            raise RuntimeError("No free pygame mixer channel for another piano voice.")
        sound = self._sounds.get(note)
        if sound is None:
            frequency = 440.0 * 2.0 ** ((note - 69) / 12.0)
            period_samples = max(2, round(self._sample_rate / frequency))
            phase = 2.0 * np.pi * np.arange(period_samples) / period_samples
            waveform = (
                np.sin(phase)
                + 0.35 * np.sin(2.0 * phase)
                + 0.18 * np.sin(3.0 * phase)
                + 0.08 * np.sin(4.0 * phase)
            )
            waveform = waveform / 1.61 * 0.55
            samples = np.asarray(waveform * 32767, dtype=np.int16)
            sound = self._pygame.mixer.Sound(buffer=samples.tobytes())
            self._sounds[note] = sound
        channel.set_volume(max(0.0, min(velocity / 127.0, 1.0)) * 0.45)
        channel.play(sound, loops=-1)
        self._channels[note] = channel

    def note_off(self, note: int) -> None:
        channel = self._channels.pop(note, None)
        if channel is not None:
            channel.fadeout(35)

    def close(self) -> None:
        for channel in self._channels.values():
            channel.stop()
        self._channels.clear()
        if self._owns_mixer:
            self._pygame.mixer.quit()


class MidiEngine:
    """Use FluidSynth, host MIDI, or a local polyphonic pygame synthesizer."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._queue: queue.SimpleQueue[NoteEvent | None] = queue.SimpleQueue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="midi-audio", daemon=True)
        self._startup_error: BaseException | None = None
        self._runtime_error: Exception | None = None

    def start(self, timeout: float = 10.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Audio engine initialization timed out.")
        if self._startup_error is not None:
            raise RuntimeError(f"Cannot initialize MIDI audio: {self._startup_error}") from self._startup_error

    def send(self, events: tuple[NoteEvent, ...]) -> None:
        if self._runtime_error is not None:
            raise RuntimeError(f"MIDI audio worker failed: {self._runtime_error}") from self._runtime_error
        for event in events:
            self._queue.put(event)

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                LOGGER.error("Audio worker did not stop within timeout")

    def _run(self) -> None:
        backend: str | None = None
        synth: _FluidSynth | None = None
        soundfont_id: int | None = None
        midi_output: _MidiOutput | None = None
        pygame_synth: _PygameSynth | None = None
        midi_initialized = False
        active: Counter[int] = Counter()
        try:
            if self._config.soundfont:
                soundfont = Path(self._config.soundfont).expanduser().resolve()
                if not soundfont.is_file():
                    raise FileNotFoundError(f"SoundFont not found: {soundfont}")
                import fluidsynth

                synth = fluidsynth.Synth(samplerate=self._config.sample_rate)
                soundfont_id = synth.sfload(str(soundfont), reset_presets=True)
                if soundfont_id < 0:
                    raise RuntimeError(f"FluidSynth could not load SoundFont: {soundfont}")
                synth.program_select(0, soundfont_id, self._config.bank, self._config.preset)
                synth.start(driver=self._config.audio_driver)
                backend = "fluidsynth"
            else:
                try:
                    import pygame.midi

                    pygame.midi.init()
                    midi_initialized = True
                    output_id = pygame.midi.get_default_output_id()
                    if output_id >= 0:
                        midi_output = pygame.midi.Output(output_id)
                        backend = "pygame.midi"
                    else:
                        LOGGER.info("No MIDI output found; using the local pygame synthesizer")
                except Exception:
                    LOGGER.warning(
                        "MIDI output is unavailable; trying the local pygame synthesizer",
                        exc_info=True,
                    )
                if backend is None:
                    if midi_initialized:
                        import pygame.midi

                        pygame.midi.quit()
                        midi_initialized = False
                    pygame_synth = _PygameSynth(self._config.sample_rate)
                    backend = "pygame.mixer"
            LOGGER.info("MIDI audio initialized (%s)", backend)
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            LOGGER.exception("Audio backend initialization failed")
            if synth is not None:
                try:
                    synth.delete()
                except Exception:
                    LOGGER.exception("Failed to dispose partially initialized FluidSynth")
            if midi_initialized:
                try:
                    if midi_output is not None:
                        midi_output.close()
                    import pygame.midi

                    pygame.midi.quit()
                except Exception:
                    LOGGER.exception("Failed to dispose partially initialized MIDI output")
            return
        self._ready.set()

        try:
            while True:
                event = self._queue.get()
                if event is None:
                    break
                if event.is_on:
                    active[event.note] += 1
                    if active[event.note] == 1:
                        if backend == "fluidsynth":
                            if synth is None:
                                raise RuntimeError("FluidSynth backend has no active synthesizer.")
                            synth.noteon(0, event.note, event.velocity)
                        else:
                            if backend == "pygame.midi":
                                if midi_output is None:
                                    raise RuntimeError("MIDI output backend has no active device.")
                                midi_output.note_on(event.note, event.velocity, 0)
                            else:
                                if pygame_synth is None:
                                    raise RuntimeError("Pygame synthesizer is not initialized.")
                                pygame_synth.note_on(event.note, event.velocity)
                elif active[event.note] > 0:
                    active[event.note] -= 1
                    if active[event.note] == 0:
                        del active[event.note]
                        if backend == "fluidsynth":
                            if synth is None:
                                raise RuntimeError("FluidSynth backend has no active synthesizer.")
                            synth.noteoff(0, event.note)
                        else:
                            if backend == "pygame.midi":
                                if midi_output is None:
                                    raise RuntimeError("MIDI output backend has no active device.")
                                midi_output.note_off(event.note, 0, 0)
                            else:
                                if pygame_synth is None:
                                    raise RuntimeError("Pygame synthesizer is not initialized.")
                                pygame_synth.note_off(event.note)
        except Exception as exc:
            self._runtime_error = exc
            LOGGER.exception("MIDI event processing failed")
        finally:
            for note in active:
                try:
                    if backend == "fluidsynth":
                        if synth is None:
                            raise RuntimeError("FluidSynth backend has no active synthesizer.")
                        synth.noteoff(0, note)
                    else:
                        if backend == "pygame.midi":
                            if midi_output is None:
                                raise RuntimeError("MIDI output backend has no active device.")
                            midi_output.note_off(note, 0, 0)
                        elif pygame_synth is not None:
                            pygame_synth.note_off(note)
                except Exception:
                    LOGGER.exception("Failed to release MIDI note %d", note)
            if backend == "fluidsynth" and synth is not None:
                try:
                    synth.delete()
                except Exception:
                    LOGGER.exception("Failed to close FluidSynth")
            elif backend == "pygame.midi":
                try:
                    if midi_output is None:
                        raise RuntimeError("MIDI output backend has no active device.")
                    midi_output.close()
                    import pygame.midi

                    pygame.midi.quit()
                except Exception:
                    LOGGER.exception("Failed to close pygame.midi")
            elif pygame_synth is not None:
                try:
                    pygame_synth.close()
                except Exception:
                    LOGGER.exception("Failed to close pygame synthesizer")
