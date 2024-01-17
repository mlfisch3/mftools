import winsound
import numpy as np

def beep(freq=440, duration=100):
    winsound.Beep(freq, duration)

def beep_random(freq_range=(320,660), size=10, duration=100):
    freqs = np.random.randint(*freq_range,size=size)
    _ = [winsound.Beep(freq, duration) for freq in freqs]
    