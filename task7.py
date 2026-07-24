import matplotlib.pyplot as plt
import numpy as np

hbar = 1.054571818e-34
m = 9.1093837e-31
L = 1e-9
eV = 1.602176634e-19

n_values = np.arange(1, 6)
energies_joules = (np.pi**2 * n_values**2 * hbar**2) / (2 * m * L**2)
energies_ev = energies_joules / eV

fig1, ax1 = plt.subplots(figsize=(8, 5))
n_smooth = np.linspace(0, 5, 200)
E_smooth = (np.pi**2 * n_smooth**2 * hbar**2) / (2 * m * L**2) / eV
ax1.plot(n_smooth, E_smooth, '-', color='navy', linewidth=2)
ax1.plot(n_values, energies_ev, 'o', color='navy', markersize=8)
ax1.set_xlabel('Quantum Number (n)', fontsize=12)
ax1.set_ylabel('Energy (eV)', fontsize=12)
ax1.set_title(f'Particle in a Box Energy /eV\nm = {m:.4e} kg', fontsize=14)
ax1.set_xticks(n_values)
ax1.set_xlim(0, 6)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task7_energy_levels.png', dpi=150)
plt.show()

x = np.linspace(0, L, 1000)
x_nm = x * 1e9

fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

for n in n_values:
    psi = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
    ax2_1.plot(x_nm, psi * 1e-4, label=f'n = {n}', linewidth=1.8)

ax2_1.set_ylabel('Wavefunction  ψₙ(x)  [×10⁴ m⁻¹/²]', fontsize=11)
ax2_1.set_title('Particle in a 1D Box — Wavefunctions ψₙ(x) & Probability Densities |ψₙ(x)|²', fontsize=13)
ax2_1.legend(loc='upper right')
ax2_1.grid(True, alpha=0.3)

for n in n_values:
    psi = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
    prob_density = psi**2
    ax2_2.plot(x_nm, prob_density * 1e-9, label=f'n = {n}', linewidth=1.8)

ax2_2.set_xlabel('Displacement  x  (nm)', fontsize=11)
ax2_2.set_ylabel('Probability Density  |ψₙ(x)|²  (nm⁻¹)', fontsize=11)
ax2_2.set_xlim(0, L * 1e9)
ax2_2.set_ylim(bottom=0)
ax2_2.legend(loc='upper right')
ax2_2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task7_wavefunctions_and_densities.png', dpi=150)
plt.show()
