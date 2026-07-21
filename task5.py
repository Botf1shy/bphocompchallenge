import numpy as np
import matplotlib.pyplot as plt

h  = 6.626e-34    # Planck's constant (J·s)
c  = 3.0e8        # Speed of light (m/s)
e  = 1.602e-19    # Electron charge (C)
E0 = 13.6         # Hydrogen ground state energy (eV)

def photon_energy(n_i, n_f):
    """Photon energy in eV for transition n_i → n_f"""
    return E0 * (1/n_f**2 - 1/n_i**2)

def photon_wavelength(n_i, n_f):
    """Photon wavelength in nm for transition n_i → n_f"""
    E_eV = photon_energy(n_i, n_f)
    E_J  = E_eV * e
    return h * c / E_J * 1e9    


series = {
    'Lyman (n_f=1)':    {'n_f': 1, 'color': 'violet',    'region': 'UV'},
    'Balmer (n_f=2)':   {'n_f': 2, 'color': 'dodgerblue','region': 'Visible'},
    'Paschen (n_f=3)':  {'n_f': 3, 'color': 'green',     'region': 'IR'},
    'Brackett (n_f=4)': {'n_f': 4, 'color': 'orange',    'region': 'IR'},
    'Pfund (n_f=5)':    {'n_f': 5, 'color': 'red',       'region': 'IR'},
}

n_max = 20 

fig, ax = plt.subplots(figsize=(10, 6))

for name, info in series.items():
    n_f = info['n_f']
    n_i_values = np.arange(n_f + 1, n_max + 1)

    energies    = [photon_energy(ni, n_f) for ni in n_i_values]
    wavelengths = [photon_wavelength(ni, n_f) for ni in n_i_values]

    ax.scatter(wavelengths, energies, label=f'{name}  ({info["region"]})',
               color=info['color'], s=30, zorder=3)

    for wl, en in zip(wavelengths, energies):
        ax.vlines(wl, 0, en, color=info['color'], alpha=0.3, linewidth=0.8)

ax.axvspan(400, 700, alpha=0.1, color='gold', label='Visible (400–700 nm)')

ax.set_xlabel('Wavelength  λ  (nm)')
ax.set_ylabel('Photon Energy  (eV)')
ax.set_title('Hydrogen Emission Spectrum — Bohr Model')
ax.legend(loc='upper right')
ax.set_xlim(0, 2500)
ax.set_ylim(0, 14)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('task5_hydrogen_energy_wavelength.png', dpi=150)
plt.show()




print(f'{"Series":<12} {"Transition":<12} {"Energy (eV)":<14} {"Wavelength (nm)":<10}')
print('-' * 50)
for name, info in series.items():
    n_f = info['n_f']
    for n_i in range(n_f + 1, n_f + 5):
        E = photon_energy(n_i, n_f)
        wl = photon_wavelength(n_i, n_f)
        print(f'{name:<12} {n_i} -> {n_f:<8} {E:<14.4f} {wl:<10.1f}')
    print()
