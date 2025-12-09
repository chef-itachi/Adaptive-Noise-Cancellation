#Denoise a generated audio using Weiner filter
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import windows

audio = AudioSegment.from_file("/content/adsp_audio_lab.aac", format="aac")
fs = audio.frame_rate
samples = np.array(audio.get_array_of_samples()).astype(np.float64)

samples /= np.max(np.abs(samples))
if audio.channels == 2:
    samples = samples.reshape((-1, 2))
    samples = samples.mean(axis=1)

u = samples

FRAME_SIZE = 1024
OVERLAP = 512
window = windows.hann(FRAME_SIZE, sym=False)
N_NOISE_FRAMES = int(0.5 * fs) // (FRAME_SIZE - OVERLAP)

noise_segment = u[:N_NOISE_FRAMES * (FRAME_SIZE - OVERLAP) + OVERLAP]
noise_frames = []
for start in range(0, len(noise_segment) - FRAME_SIZE, FRAME_SIZE - OVERLAP):
    noise_frames.append(noise_segment[start:start + FRAME_SIZE] * window)
noise_frames = np.array(noise_frames)

noise_fft_magnitudes = np.abs(fft(noise_frames, axis=1))
noise_power_spectrum = np.mean(noise_fft_magnitudes**2, axis=0)

filtered_output = np.zeros_like(u)
u_idx = 0
y_idx = 0
for start in range(0, len(u) - FRAME_SIZE, FRAME_SIZE - OVERLAP):
    frame = u[start:start + FRAME_SIZE] * window

    U_omega = fft(frame)
    U_mag_sq = np.abs(U_omega)**2

    Pu = U_mag_sq
    Pv = noise_power_spectrum

    gain = (Pu - Pv) / Pu
    gain[gain < 0] = 0
    gain[Pv == 0] = 1.0

    Y_omega = np.sqrt(gain) * U_omega
    y_frame = np.real(ifft(Y_omega))


    overlap_add_start = start + OVERLAP
    overlap_add_end = overlap_add_start + OVERLAP

    filtered_output[start:start + FRAME_SIZE] += y_frame

time_vector = np.linspace(0, len(u) / fs, len(u))


plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(time_vector, u, color='blue')
plt.title("Original Noisy Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(time_vector, filtered_output, color='green')
plt.title("Denoised Audio Signal (Wiener Filter Output)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

error = u - filtered_output
plt.subplot(3, 1, 3)
plt.plot(time_vector, error, color='red')
plt.title("Removed Noise Estimate (Error Signal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
