import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import IPython.display as ipds
from python_speech_features import mfcc, logfbank


freq_sample, signal_audio = wavfile.read("data/welcome.wav")

print(f"Shape of signal:: {signal_audio.shape}")
print(f"Signal datatype:: {signal_audio.dtype}")
print(f"Signal duration:: {round(signal_audio.shape[0] / float(freq_sample), 2)} seconds")


ipds.Audio("data/welcome.wav")

signal_audio = signal_audio / np.power(2, 15)

time_axis = 1000 * np.arange(0, len(signal_audio), 1) / float(freq_sample)

plt.plot(time_axis, signal_audio, color="blue")
plt.xlabel("Time (miliseconds)")
plt.ylabel("Amplitude")
plt.title("Time domain feature")

plt.show()