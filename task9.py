import numpy as np
import matplotlib.pyplot as plt

# ── Physical constants ────────────────────────────────────────────────────────
h    = 6.626e-34     # Planck's constant (J·s)
c    = 3.0e8         # Speed of light (m/s)
m_e  = 9.109e-31     # Electron mass (kg)
eV   = 1.602e-19     # Joule per eV

m_e_c2_eV = m_e * c**2 / eV   # electron rest energy ≈ 0.511 MeV
lambda_C  = h / (m_e * c)     # Compton wavelength ≈ 2.426 pm

# ── Compton scattering equations ──────────────────────────────────────────────
# alpha = E_photon / (m_e c^2)   (dimensionless photon energy)
#
# Wavelength shift:    Dlambda/lambda = alpha * (1 - cos(theta))
# Scattered energy:    E' = E / (1 + alpha*(1 - cos(theta)))
# Electron KE:         K_e = E - E'
# Electron speed:      gamma = 1 + K_e/(m_e c^2),  v/c = sqrt(1 - 1/gamma^2)
# Electron recoil angle: cot(phi) = (1 + alpha) * tan(theta/2)

# Photon scattering angle: 0 to 180 degrees
theta = np.linspace(0.001, np.pi, 1000)   # avoid exactly 0 for cot/tan
theta_deg = np.degrees(theta)

# Incident photon energies to compare (in keV)
photon_energies_keV = [50, 100, 200, 500, 1000]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

for E_keV in photon_energies_keV:
    E_eV  = E_keV * 1e3
    alpha = E_eV / m_e_c2_eV              # dimensionless

    # ── Fractional wavelength shift ──
    frac_shift = alpha * (1 - np.cos(theta))
    ax1.plot(theta_deg, frac_shift, label=f'{E_keV} keV')

    # ── Electron recoil speed ──
    K_e_eV = E_eV * alpha * (1 - np.cos(theta)) / (1 + alpha * (1 - np.cos(theta)))
    gamma  = 1 + K_e_eV / m_e_c2_eV
    beta   = np.sqrt(1 - 1 / gamma**2)    # v/c
    ax2.plot(theta_deg, beta, label=f'{E_keV} keV')

    # ── Electron recoil angle ──
    phi = np.arctan2(1, (1 + alpha) * np.tan(theta / 2))
    phi_deg = np.degrees(phi)
    ax3.plot(theta_deg, phi_deg, label=f'{E_keV} keV')


# ── Axis labels and formatting ───────────────────────────────────────────────
ax1.set_ylabel('Fractional shift  d(lambda)/lambda')
ax1.set_title('Compton Scattering')
ax1.legend(title='Photon energy')
ax1.grid(True, alpha=0.3)

ax2.set_ylabel('Electron recoil speed  v/c')
ax2.legend(title='Photon energy')
ax2.grid(True, alpha=0.3)

ax3.set_xlabel('Photon scattering angle  theta  (degrees)')
ax3.set_ylabel('Electron recoil angle  phi  (degrees)')
ax3.legend(title='Photon energy')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 180)

plt.tight_layout()
plt.savefig('task9_compton.png', dpi=150)
plt.show()
