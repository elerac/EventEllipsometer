from pathlib import Path
import numpy as np

N = 16384
p = 0.1
N_ramp = int(N * p)
N_hold = (N - 2 * N_ramp) // 2
dir = Path(__file__).parent


print("Generating waveforms in", dir)
print("--------------------------------------")
print("Usage: Import the generated txt files in the WaveForms software")
print("It is recommended to import the default settings (First, Count, Offset, Amplitude, etc.) and adjust the frequency and voltage as needed.")

# Ramp up waveform
x = np.empty(N)
x[:N_ramp] = np.linspace(0, 1, N_ramp, endpoint=True)  # Ramp up
x[N_ramp : N_ramp + N_hold] = 1  # Hold
x[N_ramp + N_hold :] = 0  # Back to zero
np.savetxt(dir / "MyRampUp.txt", x, fmt="%f")

# Ramp down waveform
x = np.empty(N)
x[:N_ramp] = np.linspace(1, 0, N_ramp, endpoint=True)  # Ramp down
x[N_ramp : N_ramp + N_hold] = 0  # Hold
x[N_ramp + N_hold :] = 1  # Back to one
np.savetxt(dir / "MyRampDown.txt", x, fmt="%f")

# Pulse (0: the ramp up/down period, 1: the hold/reset period)
x = np.full(N, 1)
x[1:N_ramp] = 0
np.savetxt(dir / "MyPulse.txt", x, fmt="%f")
