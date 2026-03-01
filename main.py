import numpy as np
import matplotlib.pyplot as plt

# Parameter
n_curves = 1000
t = np.linspace(0, 10, 500)  # Zeitachse jeder Kurve

T_min = 0.1   # gesund
T_max = 5.0   # verschlissen

# T wächst exponentiell über die 1000 Kurven
T_values = T_min * (T_max / T_min) ** (np.linspace(0, 1, n_curves))

# PT1 Kurven generieren
K = 1.0  # Verstärkung
dataset = np.zeros((n_curves, len(t)))

for i, T in enumerate(T_values):
    dataset[i] = K * (1 - np.exp(-t / T))

# Plotten: erste, mittlere, letzte Kurve
plt.figure(figsize=(10, 5))
for idx, label in [(0, "gesund (T=0.1)"), (499, "mittel"), (999, "verschlissen (T=5.0)")]:
    plt.plot(t, dataset[idx], label=label)

plt.title("PT1 Kurven - Alterungsverlauf")
plt.xlabel("Zeit (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

print(f"Dataset Shape: {dataset.shape}")  # (1000, 500)
print(f"T_min: {T_values[0]:.3f}, T_max: {T_values[-1]:.3f}")