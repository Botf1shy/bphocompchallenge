import numpy as np
import matplotlib.pyplot as plt

h  = 6.626e-34   # Planck's constant (J·s)
c  = 3.0e8       # Speed of light (m/s)
kB = 1.381e-23   # Boltzmann constant (J/K)
R  = 8.314       # Gas constant (J/mol/K)


def planck(wavelength, T):
    """Spectral radiance B(λ, T) in W/m²/sr/m"""
    exponent = (h * c) / (wavelength * kB * T)
    return (2 * h * c**2 / wavelength**5) / (np.exp(exponent) - 1)


wavelengths = np.linspace(100e-9, 3000e-9, 2000)

temperatures = [3000, 4000, 5000, 6000]   # Kelvin

fig, ax = plt.subplots(figsize=(9, 5))

for T in temperatures:
    B = planck(wavelengths, T)
    ax.plot(wavelengths * 1e9, B, label=f'T = {T} K')

ax.axvspan(400, 700, alpha=0.15, color='gold', label='Visible (400–700 nm)')

ax.set_xlabel('Wavelength λ (nm)')
ax.set_ylabel('Spectral Radiance B(λ,T)  [W m⁻² sr⁻¹ m⁻¹]')
ax.set_title("Planck Black Body Radiation Spectrum")
ax.legend()
ax.set_xlim(100, 3000)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('task3_blackbody.png', dpi=150)
plt.show()

def einstein_Cv(T, theta_E):
    """
    Molar heat capacity from Einstein model (J/mol/K)
    theta_E : Einstein temperature in Kelvin
    """
    x = theta_E / T
    return 3 * R * x**2 * np.exp(x) / (np.exp(x) - 1)**2

materials = {
    'Gold (Au)':   165,
    'Copper (Cu)': 240,
    'Iron (Fe)':   470,
}

T_range = np.linspace(10, 1000, 2000)  

fig, ax = plt.subplots(figsize=(9, 5))

for name, theta_E in materials.items():
    Cv = einstein_Cv(T_range, theta_E)
    ax.plot(T_range, Cv, label=f'{name}  (θ_E = {theta_E} K)')

ax.axhline(3 * R, linestyle='--', color='grey',
           label=f'Dulong–Petit limit  3R = {3*R:.1f} J mol⁻¹ K⁻¹')

ax.set_xlabel('Temperature T (K)')
ax.set_ylabel('Molar Heat Capacity Cᵥ  [J mol⁻¹ K⁻¹]')
ax.set_title("Einstein Model of Molar Heat Capacity")
ax.legend()
ax.set_xlim(0, 1000)
ax.set_ylim(0, 30)
plt.tight_layout()
plt.savefig('task3_einstein.png', dpi=150)
plt.show()
