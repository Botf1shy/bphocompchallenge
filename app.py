import base64
import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from flask import Flask, render_template, request

from task2 import bounce

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
        description='Create a model of a random walk of N steps of size s',
        fields=[
            TaskField(name='step_size', label='Step Size', type='number', value='1', min='1', step='1'),
            TaskField(name='num_steps', label='Number of Steps', type='number', value='100', min='1', step='1'),
            TaskField(name='num_randomwalks', label='Number of Random Walks', type='number', value='5', min='1', max='20', step='1'),
        ],
    ),
    2: TaskConfig(
        id=2,
        name='Brownian Motion',
        description='Consider N small particles of mass m and radius r moving randomly, some of which collide with a large particle of mass M and radius R. Animate the subsequent motion of the system',
        fields=[
            TaskField(name='num_particles', label='Number of Small Particles', type='number', value='60', min='10', max='200', step='10'),
            TaskField(name='temperature_c', label='Temperature (°C)', type='number', value='100', min='0', max='500', step='10'),
            TaskField(name='time_ps', label='Simulation Time (ps)', type='number', value='50', min='1', max='200', step='1'),
        ],
        
    ),
    3: TaskConfig(
        id=3,
        name='Black Body and Einstein Heat Capacity',
        description='Plot the Planck black body radiation spectrum for several temperatures, and also plot Einstein\'s model of molar heat capacity of solids vs temperature',
        fields=[],
    ),
    4: TaskConfig(
        id=4,
        name='Photoelectric Effect',
        description='Plot photoelectron stopping voltage vs frequency of incident photons for various metals',
        fields=[],
    ),
    5: TaskConfig(
        id=5,
        name='Hydrogen Emission Spectrum',
        description='A graph of photon energy vs wavelength for photon emissions from hydrogen atoms due to transitions between electron energy levels',
        fields=[],
    ),
    6: TaskConfig(
        id=6,
        name='Electron diffraction',
        description='Create a computer model of the electron wave rings on a phosphor screen with accelerating voltage V as a variable',
        fields=[],
    ),
    7: TaskConfig(
        id=7,
        name='Particle in a Box',
        description='Plot energy vs quantum number, and probability densities vs displacement in the box',
        fields=[],
    ),
    8: TaskConfig(
        id=8,
        name='Quantum cryptography',
        description='Create a visual calculator of the classical and quantum mismatch probabilities for the detection of polarized entangled photons',
        fields=[
            TaskField(name='wavelength_nm', label='Wavelength (nm)', type='number', value='550', min='100', max='800', step='10'),
            TaskField(name='slit_separation_um', label='Slit Separation (μm)', type='number', value='0.5', min='0.01', max='5', step='0.01'),
            TaskField(name='screen_distance_m', label='Screen Distance (m)', type='number', value='1', min='0.1', max='5', step='0.1'),
        ],
    ),
    9: TaskConfig(
        id=9,
        name='Compton Scattering',
        description='Plot fractional wavelength shift, electron recoil speed and electron recoil angle vs photon scattering angle',
        fields=[],
    ),
    10: TaskConfig(
        id=10,
        name='Hydrogenic orbitals',
        description='Plot 2D slices and 3D visualizatioins of the probability density for an electron in a hydrogenic atom',
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
    """
    Render an animated GIF
    
    uses the same physics
    style
    line
    brownian
    []
    """
    rng = np.random.default_rng()

    # Physical parameters (identical to task2.py)
    N = max(10, min(int(num_particles), 200))
    m = 28.96e-3 / 6.02e23
    M = 10.0 * m
    r = 0.16
    R = 10.0 * r
    a = 7.0 * R
    C = 1.0
    k_B = 1.38e-23
    T_K = float(temperature_c) + 273.15
    v = math.sqrt(3.0 * k_B * T_K/ m) / 1000.0 #nm/ps
    V = math.sqrt(3.0 * k_B * T_K / M) / 1000.0
    Kn = 15.0
    t_max = max(1.0,  float(time_ps))
    dt = 0.01 * Kn * r / v

    total_steps = max(1, int(t_max / dt))
    max_frames = 80
    stride = max(1, total_steps // max_frames)

    #Initial position and velocity of the large particle
    X = 0.5 * a
    Y = 0.5 * a
    theta0 = rng.uniform(0.0, 2.0 * math.pi)
    Vx = V * math.cos(theta0)
    Vy = V * math.sin(theta0)

    #Initital positions of small particles (rejection sample so none overlap
    #the large particle).
    x = np.empty(N)
    y = np.empty(N)
    for n in range(N):
        while True:
            cx = r + rng.random() * (a - 2.0 * r)
            cy = r + rng.random() * (a - 2.0 * r)
            if math.hypot(cx - X, cy - Y) >= r + R:
                x[n] = cx
                y[n] = cy
                break

    #Initial velocities of small particles
    theta = rng.uniform(0.0, 2.0 * math.pi, size=N)
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)

    # Run the simulation and snapshot every 'stride' steps.
    trail_x = [X]
    trail_y = [Y]
    frames: list[tuple[float, float, np.ndarray, np.ndarray, float, list[float], list[float]]] = [
        (X, Y, x.copy(), y.copy(), 0.0, trail_x.copy(), trail_y.copy())
    ]

    t = 0.0
    time_since_randomisation = 0.0

    for step in range(1, total_steps + 1):
        t += dt
        time_since_randomisation += dt
    
        # Ballistic motion (no wall collisions - matches task2.py)
        X += Vx * dt
        Y += Vy * dt
        x += vx * dt
        y += vy * dt
        
        trail_x.append(X)
        trail_y.append(Y)

        # Collisions between small particles and the large particle
        for n in range(N):
            (
                Vx,
                Vy,
                vx[n],
                vy[n],
                X,
                x[n],
                Y,
                y[n],
            ) = bounce(
                X, Y, x[n] , y[n],
                Vx, vx[n], Vy, vy[n],
                C, M, m, R, r,
            )

        # Randomise small-particle velocity directions every Kn * r / v ps
        if time_since_randomisation > Kn * r / v:
            time_since_randomisation = 0.0
            theta = rng.uniform(0.0, 2.0 * math.pi, size=N)
            vx = v * np.cos(theta)
            vy = v * np.sin(theta)

        if step % stride == 0 or step == total_steps:
            frames.append((X, Y, x.copy(), y.copy(), t, list(trail_x), list(trail_y)))

    # Build the animation using the same visual style as task2.py
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.2 * a, 1.2 * a)
    ax.set_ylim(-0.2 * a, 1.2 * a)
    ax.set_axis_off()

    # Black bounding box
    ax.plot([0, a, a, 0, 0], [0, 0, a, a, 0], 'k-', linewidth=3)

    # Circle points for the large particle outline
    angle = np.linspace(0, 2 * np.pi, 500)
    xc = R * np.cos(angle)
    yc = R * np.sin(angle)

    X0, Y0, x0, y0, t0, tx0, ty0 = frames[0]
    large_line, = ax.plot(X0 + xc, Y0 + yc, 'r-', linewidth=2)
    small_points, = ax.plot(x0, y0, 'b*', markersize=4)
    trail_line, = ax.plot(tx0, ty0, 'r-', linewidth=1, alpha=0.7)
    ax.plot([X0], [Y0], 'g*', markersize=8) # start marker (static)
    latest_point, = ax.plot([X0], [Y0], 'r*', markersize=8) # latest position marker
    title = ax.set_title(f"Brownian motion simulation: t = {t0:.3f} ps", fontsize=18)

    def update(i: int):
        Xi, Yi, xi, yi, ti, txi, tyi = frames[i]
        large_line.set_data(Xi + xc, Yi + yc)
        small_points.set_data(xi, yi)
        trail_line.set_data(txi, tyi)
        latest_point.set_data([Xi], [Yi])
        title.set_text(f"Brownian motion simulation: t = {ti:.3f} ps")
        return large_line, small_points, trail_line, latest_point, title

    anim = manimation.FuncAnimation(
        fig, update, frames=len(frames), interval=80, blit=False
    )

    # Save the animation to a temporary GIF file and return its data URI
    tmp = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    tmp.close()
    try:
        anim.save(tmp.name, writer=manimation.PillowWriter(fps=15))
        with open(tmp.name, 'rb') as f:
            data = base64.b64encode(f.read()).decode('ascii')
    finally:
        plt.close(fig)
        try:
            os.remove(tmp.name)
        except OSError:
            pass

    return [f'data:image/gif;base64,{data}']

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

    def photon_energy(n_i, n_f):
        return E0 * (1 / n_f**2 - 1 / n_i**2)

    def photon_wavelength(n_i, n_f):
        E_eV = photon_energy(n_i, n_f)
        E_J = E_eV * e
        return h * c / E_J * 1e9

    series = {
        'Lyman (n_f=1)':    {'n_f': 1, 'color': 'violet',     'region': 'UV'},
        'Balmer (n_f=2)':   {'n_f': 2, 'color': 'dodgerblue', 'region': 'Visible'},
        'Paschen (n_f=3)':  {'n_f': 3, 'color': 'green',      'region': 'IR'},
        'Brackett (n_f=4)': {'n_f': 4, 'color': 'orange',     'region': 'IR'},
        'Pfund (n_f=5)':    {'n_f': 5, 'color': 'red',        'region': 'IR'},
    }

    n_max = 20

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, info in series.items():
        n_f = info['n_f']
        n_i_values = np.arange(n_f + 1, n_max + 1)

        energies = [photon_energy(ni, n_f) for ni in n_i_values]
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
    ax.set_xlim(0, 8000)
    ax.set_ylim(0, 14)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    return [fig_to_data_uri(fig)]

def task6_plot(r_mm: float, d1_nm: float, d2_nm: float, v_min_kv: float, v_max_kv: float) -> list[str]:
    h = 6.626e-34
    e_ = 1.602e-19
    m_ = 9.109e-31

    d_values = [d1_nm * 1e-9, d2_nm * 1e-9]
    d_labels = [f'd\u2081={d1_nm} nm', f'd\u2082={d2_nm} nm']
    colors = ['#4C72B0', '#DD8452']
    n_orders = 2

    voltages_kv = np.linspace(v_min_kv, v_max_kv, 41)
    theta_grid = np.linspace(0, 2 * np.pi, 200)

    def ring_radius(V_volts, d, n):
        lam = h / math.sqrt(2 * m_ * e_ * V_volts)
        s = n * lam / (2 * d)
        if s > 1:
            return None
        phi = 2 * math.asin(s)
        return r_mm * math.sin(phi)

    # trace order is fixed across frames: (d_idx, n) pairs
    trace_specs = [(d, color, label, n)
                   for d, color, label in zip(d_values, colors, d_labels)
                   for n in range(1, n_orders + 1)]

    frames = []
    for V_kv in voltages_kv:
        V = V_kv * 1000
        traces = []
        for d, color, label, n in trace_specs:
            x = ring_radius(V, d, n)
            if x is None:
                xs, ys = [np.nan], [np.nan]
            else:
                xs, ys = x * np.cos(theta_grid), x * np.sin(theta_grid)
            traces.append(go.Scatter(
                x=xs, y=ys, mode='lines',
                line=dict(color=color, width=2 if n == 1 else 1,
                          dash='solid' if n == 1 else 'dot'),
                name=f'{label}, n={n}',
            ))
        frames.append(go.Frame(data=traces, name=f'{V_kv:.2f}'))

    fig1 = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title='Electron Diffraction Rings on Phosphor Screen',
            xaxis=dict(range=[-r_mm, r_mm], scaleanchor='y', title='mm'),
            yaxis=dict(range=[-r_mm, r_mm], title='mm'),
            shapes=[dict(type='circle', x0=-r_mm, y0=-r_mm, x1=r_mm, y1=r_mm,
                         line=dict(color='green'))],
            sliders=[dict(
                active=0,
                currentvalue={'prefix': 'Accelerating Voltage: ', 'suffix': ' kV'},
                steps=[dict(
                    method='animate',
                    args=[[f'{V_kv:.2f}'],
                          {'mode': 'immediate',
                           'frame': {'duration': 0, 'redraw': True},
                           'transition': {'duration': 0}}],
                    label=f'{V_kv:.1f}',
                ) for V_kv in voltages_kv],
            )],
        ),
    )

    # Calibration check: sin(phi/2) vs 1/sqrt(V), n=1 line for each spacing
    V_dense = np.linspace(v_min_kv, v_max_kv, 200) * 1000
    fig2 = go.Figure()
    for d, color, label in zip(d_values, colors, d_labels):
        lam = h / np.sqrt(2 * m_ * e_ * V_dense)
        sin_half = lam / (2 * d)
        inv_sqrtV = 1 / np.sqrt(V_dense)
        fig2.add_trace(go.Scatter(x=inv_sqrtV, y=sin_half, mode='lines',
                                   name=label, line=dict(color=color)))
    fig2.update_layout(
        title='Calibration check: sin(\u03c6/2) vs 1/\u221aV (should be a straight line)',
        xaxis_title='1/\u221aV  (V in volts)',
        yaxis_title='sin(\u03c6/2)',
    )

    html1 = pio.to_html(fig1, include_plotlyjs='cdn', full_html=False)
    html2 = pio.to_html(fig2, include_plotlyjs=False, full_html=False)
    return [html1, html2]

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

    ax1.set_xlabel('Photon scattering angle θ/deg')
    ax1.set_ylabel('Fractional shift Δλ/λ')
    ax1.set_title('Compton scattering of X-ray photon off an electron')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize='small')

    ax2.set_xlabel('Photon scattering angle θ/deg')
    ax2.set_ylabel('Electron recoil speed v/c')
    ax2.set_title('Compton scattering of X-ray photon off an electron')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize='small')

    ax3.set_xlabel('Photon scattering angle θ/deg')
    ax3.set_ylabel('Electron recoil angle φ/deg')
    ax3.set_title('Compton scattering of X-ray photon off an electron')
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
            float(form_data.get('r_mm', 65.0)),
            float(form_data.get('d1_nm', 0.123)),
            float(form_data.get('d2_nm', 0.213)),
            float(form_data.get('v_min_kv', 1.0)),
            float(form_data.get('v_max_kv', 5.0)),
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
