import base64
import io
import math
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

@dataclass
class TaskField:
    name: str
    label: str
    type: str = 'text'
    value: str = ''
    min: str | None = None
    max: str | None = None
    step: str | None = None
    placeholder: str | None = None

@dataclass
class TaskConfig:
    id: int
    name: str
    description: str
    fields: list[TaskField]
    notes: str | None = None

TASKS: dict[int, TaskConfig] = {
    1: TaskConfig(
        id=1,
        name='Random Walk',
        description='Simulate 2D random walks with a selectable step size, number of steps, and number of paths.',
        fields=[
            TaskField(name='step_size', label='Step Size', type='number', value='1', min='1', step='1'),
            TaskField(name='num_steps', label='Number of Steps', type='number', value='100', min='1', step='1'),
            TaskField(name='num_randomwalks', label='Number of Random Walks', type='number', value='5', min='1', max='20', step='1'),
        ],
    ),
    2: TaskConfig(
        id=2,
        name='Brownian Motion',
        description='Run a simplified Brownian motion simulation and display the large particle trajectory and particle positions.',
        fields=[
            TaskField(name='num_particles', label='Number of Small Particles', type='number', value='60', min='10', max='200', step='10'),
            TaskField(name='temperature_c', label='Temperature (°C)', type='number', value='100', min='0', max='500', step='10'),
            TaskField(name='time_ps', label='Simulation Time (ps)', type='number', value='50', min='1', max='200', step='1'),
        ],
        notes='Using a simplified model optimized for interactive use.',
    ),
    3: TaskConfig(
        id=3,
        name='Black Body and Einstein Heat Capacity',
        description='Plot Planck black body spectra at different temperatures and Einstein-model heat capacity curves.',
        fields=[
            TaskField(name='max_temperature', label='Maximum Temperature (K)', type='number', value='6000', min='100', max='10000', step='100'),
            TaskField(name='temperature_step', label='Temperature Step (K)', type='number', value='1000', min='100', max='2000', step='100'),
        ],
    ),
    4: TaskConfig(
        id=4,
        name='Photoelectric Effect',
        description='Show how stopping voltage depends on photon frequency and wavelength for a selection of metals.',
        fields=[
            TaskField(name='metals', label='Metals', type='text', value='Caesium (Cs), Sodium (Na), Zinc (Zn), Copper (Cu)', placeholder='Comma-separated metal names'),
            TaskField(name='max_frequency', label='Maximum Frequency (×10¹⁴ Hz)', type='number', value='25', min='1', max='50', step='1'),
        ],
    ),
    5: TaskConfig(
        id=5,
        name='Hydrogen Emission Spectrum',
        description='Plot Bohr-series transition energies and wavelengths for hydrogen emission lines.',
        fields=[
            TaskField(name='max_n', label='Maximum Principal Quantum Number', type='number', value='12', min='3', max='30', step='1'),
            TaskField(name='min_series', label='Lowest Series (n_f)', type='number', value='1', min='1', max='5', step='1'),
        ],
    ),
    6: TaskConfig(
        id=6,
        name='Simple Harmonic Oscillator',
        description='Plot the position and energy of a mass-spring oscillator as a function of time.',
        fields=[
            TaskField(name='mass', label='Mass (kg)', type='number', value='0.5', min='0.01', max='10', step='0.01'),
            TaskField(name='spring_constant', label='Spring Constant (N/m)', type='number', value='20', min='0.1', max='200', step='0.1'),
            TaskField(name='amplitude', label='Amplitude (m)', type='number', value='0.5', min='0.01', max='2', step='0.01'),
        ],
    ),
    7: TaskConfig(
        id=7,
        name='Particle in a Box',
        description='Plot quantum energy levels, wavefunctions, and probability densities for a particle in a 1D box.',
        fields=[
            TaskField(name='box_length_nm', label='Box Length (nm)', type='number', value='1', min='0.1', max='10', step='0.1'),
            TaskField(name='max_n', label='Maximum Quantum Number', type='number', value='5', min='1', max='12', step='1'),
        ],
    ),
    8: TaskConfig(
        id=8,
        name='Double-Slit Interference',
        description='Visualize the interference pattern from a double-slit experiment.',
        fields=[
            TaskField(name='wavelength_nm', label='Wavelength (nm)', type='number', value='550', min='100', max='800', step='10'),
            TaskField(name='slit_separation_um', label='Slit Separation (μm)', type='number', value='0.5', min='0.01', max='5', step='0.01'),
            TaskField(name='screen_distance_m', label='Screen Distance (m)', type='number', value='1', min='0.1', max='5', step='0.1'),
        ],
    ),
    9: TaskConfig(
        id=9,
        name='Compton Scattering',
        description='Plot Compton wavelength shift, recoil speed, and recoil angle for different photon energies.',
        fields=[
            TaskField(name='photon_energies', label='Photon Energies (keV)', type='text', value='50, 100, 200, 500, 1000', placeholder='Comma-separated values'),
            TaskField(name='num_points', label='Angular Resolution', type='number', value='200', min='50', max='1000', step='50'),
        ],
    ),
    10: TaskConfig(
        id=10,
        name='Radioactive Decay',
        description='Plot the exponential decay of a radioactive sample and its activity over time.',
        fields=[
            TaskField(name='initial_amount', label='Initial Amount (atoms)', type='number', value='1e6', min='1', max='1e9', step='1000'),
            TaskField(name='half_life', label='Half-Life (s)', type='number', value='30', min='1', max='1000', step='1'),
            TaskField(name='duration', label='Duration (s)', type='number', value='150', min='10', max='1000', step='10'),
        ],
    ),
}


def fig_to_data_uri(fig: matplotlib.figure.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buffer.seek(0)
    data = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{data}'


def task1_plot(step_size: int, num_steps: int, num_randomwalks: int) -> list[str]:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Random Walks (step={step_size}, steps={num_steps})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, linestyle='--', alpha=0.5)

    for _ in range(num_randomwalks):
        x, y = 0.0, 0.0
        xs, ys = [x], [y]
        for _ in range(num_steps):
            angle = np.random.uniform(0, 2 * np.pi)
            x += step_size * np.cos(angle)
            y += step_size * np.sin(angle)
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, marker='o', markersize=3, alpha=0.7)

    return [fig_to_data_uri(fig)]


def task2_plot(num_particles: int, temperature_c: float, time_ps: float) -> list[str]:
    num_particles = max(10, min(num_particles, 200))
    num_small = int(num_particles)
    m = 28.96e-3 / 6.02e23
    M = 10.0 * m
    r = 0.16
    R = 10.0 * r
    a = 7.0 * R
    C = 1.0
    k_B = 1.38e-23
    T = temperature_c + 273.15
    v = math.sqrt(3.0 * k_B * T / m) / 1000.0
    V = math.sqrt(3.0 * k_B * T / M) / 1000.0
    Kn = 15.0
    dt = max(0.001, 0.01 * Kn * r / v)
    steps = min(int(time_ps / dt), 300)

    x = np.empty(num_small)
    y = np.empty(num_small)
    rng = np.random.default_rng(1234)
    for n in range(num_small):
        while True:
            cx = r + rng.random() * (a - 2.0 * r)
            cy = r + rng.random() * (a - 2.0 * r)
            if np.hypot(cx - 0.5 * a, cy - 0.5 * a) >= r + R:
                x[n] = cx
                y[n] = cy
                break

    theta = rng.uniform(0, 2 * math.pi, size=num_small)
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    X = 0.5 * a
    Y = 0.5 * a
    theta = rng.uniform(0, 2 * math.pi)
    Vx = V * math.cos(theta)
    Vy = V * math.sin(theta)
    trail_x = [X]
    trail_y = [Y]

    def collide(x1, y1, x2, y2, ux1, uy1, ux2, uy2):
        displacement = np.array([x2 - x1, y2 - y1], dtype=float)
        distance = np.linalg.norm(displacement)
        if distance == 0:
            direction = np.array([1.0, 0.0])
        else:
            direction = displacement / distance
        vx1, vy1 = ux1, uy1
        vx2, vy2 = ux2, uy2
        if distance <= r + R and np.dot(np.array([ux2 - ux1, uy2 - uy1]), direction) < 0:
            centre_of_mass = (M * np.array([ux1, uy1]) + m * np.array([ux2, uy2])) / (M + m)
            v1 = centre_of_mass - (ux1 - centre_of_mass[0]) * direction * C
            v2 = centre_of_mass - (ux2 - centre_of_mass[0]) * direction * C
            vx1, vy1 = v1
            vx2, vy2 = v2
            overlap = (r + R - distance) / 2.0
            x1 -= overlap * direction[0]
            y1 -= overlap * direction[1]
            x2 += overlap * direction[0]
            y2 += overlap * direction[1]
        return vx1, vy1, vx2, vy2, x1, y1, x2, y2

    for _ in range(steps):
        X += Vx * dt
        Y += Vy * dt
        if X <= R:
            X = R
            Vx = abs(Vx)
        elif X >= a - R:
            X = a - R
            Vx = -abs(Vx)
        if Y <= R:
            Y = R
            Vy = abs(Vy)
        elif Y >= a - R:
            Y = a - R
            Vy = -abs(Vy)

        x += vx * dt
        y += vy * dt
        below_x = x < r
        above_x = x > a - r
        x[below_x] = r
        x[above_x] = a - r
        vx[below_x] = abs(vx[below_x])
        vx[above_x] = -abs(vx[above_x])
        below_y = y < r
        above_y = y > a - r
        y[below_y] = r
        y[above_y] = a - r
        vy[below_y] = abs(vy[below_y])
        vy[above_y] = -abs(vy[above_y])

        for n in range(num_small):
            Vx, Vy, vx[n], vy[n], X, Y, x[n], y[n] = collide(
                X, Y, x[n], y[n], Vx, Vy, vx[n], vy[n]
            )
        trail_x.append(X)
        trail_y.append(Y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(trail_x, trail_y, '-', color='red', label='Large particle trajectory')
    ax.scatter(x, y, s=8, color='blue', alpha=0.7, label='Small particles')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, a)
    ax.set_ylim(0, a)
    ax.set_title('Simplified Brownian Motion Final State')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)
    return [fig_to_data_uri(fig)]


def task3_plot(max_temperature: int, temperature_step: int) -> list[str]:
    temperatures = list(range(3000, max_temperature + 1, temperature_step))
    wavelengths = np.linspace(100e-9, 3000e-9, 2000)
    h = 6.626e-34
    c = 3.0e8
    kB = 1.381e-23
    R = 8.314

    def planck(wavelength, T):
        exponent = (h * c) / (wavelength * kB * T)
        return (2 * h * c**2 / wavelength**5) / (np.exp(exponent) - 1)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for T in temperatures:
        B = planck(wavelengths, T)
        ax1.plot(wavelengths * 1e9, B, label=f'{T} K')
    ax1.axvspan(400, 700, alpha=0.15, color='gold', label='Visible')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Spectral Radiance')
    ax1.set_title('Planck Black Body Spectrum')
    ax1.legend(fontsize='small')
    ax1.set_xlim(100, 3000)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    theta_einstein = {'Gold': 165, 'Copper': 240, 'Iron': 470}
    T_range = np.linspace(10, 1000, 2000)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name, theta in theta_einstein.items():
        Cv = 3 * R * (theta / T_range)**2 * np.exp(theta / T_range) / (np.exp(theta / T_range) - 1)**2
        ax2.plot(T_range, Cv, label=f'{name} (θ={theta}K)')
    ax2.axhline(3 * R, linestyle='--', color='grey', label='3R limit')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Molar Heat Capacity C_v (J/mol K)')
    ax2.set_title('Einstein Model Heat Capacity')
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 30)
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)

    return [fig_to_data_uri(fig1), fig_to_data_uri(fig2)]


def task4_plot(metals: str, max_frequency: float) -> list[str]:
    h = 6.626e-34
    c = 3.0e8
    e = 1.602e-19
    metal_map = {
        'Caesium (Cs)': 2.1,
        'Sodium (Na)': 2.28,
        'Zinc (Zn)': 4.3,
        'Copper (Cu)': 4.7,
        'Platinum (Pt)': 5.6,
        'Aluminium (Al)': 4.28,
        'Iron (Fe)': 4.5,
    }
    requested = [name.strip() for name in metals.split(',') if name.strip()]
    selected = [name for name in requested if name in metal_map]
    if not selected:
        selected = ['Caesium (Cs)', 'Sodium (Na)', 'Zinc (Zn)', 'Copper (Cu)']
    freq = np.linspace(0.01, max_frequency * 1e14, 2000)

    def stopping_voltage(freq_arr, phi_eV):
        phi_J = phi_eV * e
        V0 = (h * freq_arr - phi_J) / e
        return np.clip(V0, 0, None)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for name in selected:
        phi = metal_map[name]
        V0 = stopping_voltage(freq, phi)
        ax1.plot(freq * 1e-14, V0, label=f'{name} (φ={phi} eV)')
    ax1.set_xlabel('Frequency (×10¹⁴ Hz)')
    ax1.set_ylabel('Stopping Voltage (V)')
    ax1.set_title('Stopping Voltage vs Frequency')
    ax1.set_xlim(0, max_frequency)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize='small')

    wavelengths = np.linspace(50e-9, 800e-9, 2000)
    freq_from_wl = c / wavelengths
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name in selected:
        phi = metal_map[name]
        V0 = stopping_voltage(freq_from_wl, phi)
        lambda_threshold = h * c / (phi * e)
        ax2.plot(wavelengths * 1e9, V0, label=f'{name} (λ₀={lambda_threshold*1e9:.0f} nm)')
    ax2.axvspan(400, 700, alpha=0.12, color='gold', label='Visible')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Stopping Voltage (V)')
    ax2.set_title('Stopping Voltage vs Wavelength')
    ax2.set_xlim(50, 800)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize='small')

    return [fig_to_data_uri(fig1), fig_to_data_uri(fig2)]


def task5_plot(max_n: int, min_series: int) -> list[str]:
    h = 6.626e-34
    c = 3.0e8
    e = 1.602e-19
    E0 = 13.6
    max_n = max(3, min(max_n, 50))
    min_series = max(1, min(min_series, 5))

    def photon_energy(n_i, n_f):
        return E0 * (1 / n_f**2 - 1 / n_i**2)

    def photon_wavelength(n_i, n_f):
        E_eV = photon_energy(n_i, n_f)
        return (h * c / (E_eV * e)) * 1e9

    series_names = {
        1: ('Lyman', 'UV', 'violet'),
        2: ('Balmer', 'Visible', 'dodgerblue'),
        3: ('Paschen', 'IR', 'green'),
        4: ('Brackett', 'IR', 'orange'),
        5: ('Pfund', 'IR', 'red'),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for n_f in range(min_series, min(min_series + 4, 6)):
        name, region, color = series_names.get(n_f, (f'n_f={n_f}', 'Unknown', 'black'))
        n_i = np.arange(n_f + 1, max_n + 1)
        wavelengths = [photon_wavelength(int(n), n_f) for n in n_i]
        energies = [photon_energy(int(n), n_f) for n in n_i]
        ax.scatter(wavelengths, energies, label=f'{name} ({region})', color=color, s=40)
        for wl, en in zip(wavelengths, energies):
            ax.vlines(wl, 0, en, color=color, alpha=0.3, linewidth=0.8)

    ax.axvspan(400, 700, alpha=0.1, color='gold', label='Visible range')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Photon Energy (eV)')
    ax.set_title('Hydrogen Emission Spectrum')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 14)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize='small', loc='upper right')
    return [fig_to_data_uri(fig)]


def task6_plot(mass: float, spring_constant: float, amplitude: float) -> list[str]:
    omega = math.sqrt(spring_constant / mass)
    period = 2 * math.pi / omega
    t = np.linspace(0, 2 * period, 400)
    x = amplitude * np.cos(omega * t)
    energy = 0.5 * spring_constant * x**2 + 0.5 * mass * (omega * amplitude * np.sin(omega * t))**2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, x, label='Position x(t)')
    ax.plot(t, energy, label='Total energy E(t)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m) / Energy (J)')
    ax.set_title('Simple Harmonic Oscillator')
    ax.legend()
    ax.grid(True, alpha=0.4)
    return [fig_to_data_uri(fig)]


def task7_plot(box_length_nm: float, max_n: int) -> list[str]:
    hbar = 1.054571818e-34
    m = 9.1093837e-31
    eV = 1.602176634e-19
    L = box_length_nm * 1e-9
    n_values = np.arange(1, min(max_n, 15) + 1)
    energies_joules = (np.pi**2 * n_values**2 * hbar**2) / (2 * m * L**2)
    energies_ev = energies_joules / eV

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(n_values, energies_ev, 'o-', color='navy')
    ax1.set_xlabel('Quantum Number n')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title('Particle in a Box Energy Levels')
    ax1.grid(True, alpha=0.3)

    x = np.linspace(0, L, 1000)
    x_nm = x * 1e9
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for n in n_values[:5]:
        psi = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
        ax2.plot(x_nm, psi * 1e4, label=f'n={n}')
        ax3.plot(x_nm, psi**2 * 1e9, label=f'n={n}')
    ax2.set_ylabel('Wavefunction ψ(x) (×10⁴)')
    ax2.set_title('Wavefunctions for a 1D Particle in a Box')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    ax3.set_xlabel('Position x (nm)')
    ax3.set_ylabel('Probability Density |ψ|²')
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)

    return [fig_to_data_uri(fig1), fig_to_data_uri(fig2)]


def task8_plot(wavelength_nm: float, slit_separation_um: float, screen_distance_m: float) -> list[str]:
    wavelength = wavelength_nm * 1e-9
    d = slit_separation_um * 1e-6
    D = screen_distance_m
    x = np.linspace(-0.02, 0.02, 1200)
    theta = np.arctan2(x, D)
    slit_width = max(1e-7, d * 0.2)
    beta = np.pi * slit_width * np.sin(theta) / wavelength
    gamma = np.pi * d * np.sin(theta) / wavelength
    envelope = np.sinc(beta / np.pi) ** 2
    intensity = envelope * np.cos(gamma) ** 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x * 1000, intensity, color='purple')
    ax.set_xlabel('Screen position (mm)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.set_title('Double-Slit Interference Pattern')
    ax.grid(True, alpha=0.3)
    return [fig_to_data_uri(fig)]


def task9_plot(photon_energies: str, num_points: int) -> list[str]:
    h = 6.626e-34
    c = 3.0e8
    m_e = 9.109e-31
    eV = 1.602e-19
    m_e_c2_eV = m_e * c**2 / eV
    energies = []
    for token in photon_energies.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            energies.append(float(token))
        except ValueError:
            continue
    if not energies:
        energies = [50, 100, 200, 500, 1000]
    theta = np.linspace(0.001, np.pi, max(50, min(num_points, 1000)))
    theta_deg = np.degrees(theta)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    for E_keV in energies:
        E_eV = E_keV * 1e3
        alpha = E_eV / m_e_c2_eV
        frac_shift = alpha * (1 - np.cos(theta))
        K_e_eV = E_eV * alpha * (1 - np.cos(theta)) / (1 + alpha * (1 - np.cos(theta)))
        gamma = 1 + K_e_eV / m_e_c2_eV
        beta = np.sqrt(np.maximum(0, 1 - 1 / gamma**2))
        phi = np.arctan2(1, (1 + alpha) * np.tan(theta / 2))
        phi_deg = np.degrees(phi)
        ax1.plot(theta_deg, frac_shift, label=f'{E_keV} keV')
        ax2.plot(theta_deg, beta, label=f'{E_keV} keV')
        ax3.plot(theta_deg, phi_deg, label=f'{E_keV} keV')

    ax1.set_ylabel('Fractional shift Δλ/λ')
    ax1.set_title('Compton Scattering: Wavelength Shift')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize='small')

    ax2.set_ylabel('Electron speed v/c')
    ax2.set_title('Electron Recoil Speed')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize='small')

    ax3.set_xlabel('Scattering angle θ (degrees)')
    ax3.set_ylabel('Recoil angle φ (degrees)')
    ax3.set_title('Electron Recoil Angle')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 180)
    ax3.legend(fontsize='small')

    return [fig_to_data_uri(fig)]


def task10_plot(initial_amount: float, half_life: float, duration: float) -> list[str]:
    t = np.linspace(0, duration, 500)
    half_life = max(1e-6, half_life)
    N0 = max(1.0, initial_amount)
    decay_constant = math.log(2) / half_life
    N = N0 * np.exp(-decay_constant * t)
    activity = decay_constant * N

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, N, label='Remaining atoms')
    ax.plot(t, activity, label='Activity', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Atoms / Activity (s⁻¹)')
    ax.set_title('Radioactive Decay')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return [fig_to_data_uri(fig)]


def generate_task_images(task_id: int, form_data: dict[str, Any]) -> list[str]:
    if task_id == 1:
        return task1_plot(
            int(form_data.get('step_size', 1)),
            int(form_data.get('num_steps', 100)),
            int(form_data.get('num_randomwalks', 5)),
        )
    if task_id == 2:
        return task2_plot(
            int(form_data.get('num_particles', 60)),
            float(form_data.get('temperature_c', 100.0)),
            float(form_data.get('time_ps', 50.0)),
        )
    if task_id == 3:
        return task3_plot(
            int(form_data.get('max_temperature', 6000)),
            int(form_data.get('temperature_step', 1000)),
        )
    if task_id == 4:
        return task4_plot(
            form_data.get('metals', ''),
            float(form_data.get('max_frequency', 25.0)),
        )
    if task_id == 5:
        return task5_plot(
            int(form_data.get('max_n', 12)),
            int(form_data.get('min_series', 1)),
        )
    if task_id == 6:
        return task6_plot(
            float(form_data.get('mass', 0.5)),
            float(form_data.get('spring_constant', 20.0)),
            float(form_data.get('amplitude', 0.5)),
        )
    if task_id == 7:
        return task7_plot(
            float(form_data.get('box_length_nm', 1.0)),
            int(form_data.get('max_n', 5)),
        )
    if task_id == 8:
        return task8_plot(
            float(form_data.get('wavelength_nm', 550.0)),
            float(form_data.get('slit_separation_um', 0.5)),
            float(form_data.get('screen_distance_m', 1.0)),
        )
    if task_id == 9:
        return task9_plot(
            form_data.get('photon_energies', '50, 100, 200, 500, 1000'),
            int(form_data.get('num_points', 200)),
        )
    if task_id == 10:
        return task10_plot(
            float(form_data.get('initial_amount', 1e6)),
            float(form_data.get('half_life', 30.0)),
            float(form_data.get('duration', 150.0)),
        )
    raise ValueError(f'Unknown task {task_id}')


@app.route('/')
def index():
    return render_template('index.html', tasks=TASKS.values())


@app.route('/task/<int:task_id>', methods=['GET', 'POST'])
def task_page(task_id: int):
    task = TASKS.get(task_id)
    if task is None:
        return f'Task {task_id} does not exist.', 404

    results = None
    if request.method == 'POST':
        results = generate_task_images(task_id, request.form)

    return render_template('task.html', task=task, results=results, form=request.form)


if __name__ == '__main__':
    app.run(debug=True)
