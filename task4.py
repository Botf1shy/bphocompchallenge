import numpy as np
import matplotlib.pyplot as plt

h  = 6.626e-34    # Planck's constant (J·s)
c  = 3.0e8        # Speed of light (m/s)
e  = 1.602e-19    # Electron charge (C)



def stopping_voltage(freq, work_function_eV):
    """
    Stopping voltage V₀ for the photoelectric effect.
    freq             : photon frequency in Hz
    work_function_eV : metal work function in eV
    Returns V₀ in volts (clipped to 0 — no negative stopping voltage)
    """
    phi_J = work_function_eV * e         
    V0 = (h * freq - phi_J) / e          
    return np.maximum(V0, 0)              


metals = {
    'Caesium (Cs)':   2.1,
    'Sodium (Na)':    2.28,
    'Zinc (Zn)':      4.3,
    'Copper (Cu)':    4.7,
    'Platinum (Pt)':  5.6,
}

freq = np.linspace(0, 2.5e15, 2000)

fig, ax = plt.subplots(figsize=(9, 5))

for name, phi in metals.items():
    V0 = stopping_voltage(freq, phi)
    f_threshold = phi * e / h             
    ax.plot(freq * 1e-14, V0, label=f'{name}  (φ = {phi} eV)')
    ax.plot(f_threshold * 1e-14, 0, 'o', markersize=5)

ax.set_xlabel('Frequency  f  (× 10¹⁴ Hz)')
ax.set_ylabel('Stopping Voltage  V₀  (V)')
ax.set_title('Photoelectric Effect — Stopping Voltage vs Frequency')
ax.legend()
ax.set_xlim(0, 25)
ax.set_ylim(0, 8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task4_photoelectric_freq.png', dpi=150)
plt.show()



wavelength = np.linspace(50e-9, 800e-9, 2000)   
freq_from_wl = c / wavelength                     

fig, ax = plt.subplots(figsize=(9, 5))

for name, phi in metals.items():
    V0 = stopping_voltage(freq_from_wl, phi)
    lambda_threshold = h * c / (phi * e)          
    ax.plot(wavelength * 1e9, V0, label=f'{name}  (λ₀ = {lambda_threshold*1e9:.0f} nm)')

ax.axvspan(400, 700, alpha=0.12, color='gold', label='Visible (400–700 nm)')

ax.set_xlabel('Wavelength  λ  (nm)')
ax.set_ylabel('Stopping Voltage  V₀  (V)')
ax.set_title('Photoelectric Effect — Stopping Voltage vs Wavelength')
ax.legend()
ax.set_xlim(50, 800)
ax.set_ylim(0, 8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task4_photoelectric_wavelength.png', dpi=150)
plt.show()
