import numpy as np
import sounddevice as sd


def warning_beep(freq=880, duration=0.4, volume=0.3):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = volume * np.sin(2 * np.pi * freq * t)
    sd.play(tone, samplerate=sample_rate)
    sd.wait()  # wait until sound is done
