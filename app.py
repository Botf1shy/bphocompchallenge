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
from scipy.constants import physical_constants, electron_mass, atomic_mass
from scipy.special import eval_genlaguerre, sph_harm_y

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
        fields=[],
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
            TaskField(name='z', label='Proton Number Z', type='number', value='1', min='1', max='20', step='1'),
            TaskField(name='mass_number', label='Mass Number A', type='number', value='1', min='1', max='40', step='1'),
            TaskField(name='n', label='Principal Quantum Number n', type='number', value='3', min='1', max='6', step='1'),
            TaskField(name='l', label='Angular Quantum Number l', type='number', value='2', min='0', max='5', step='1'),
            TaskField(name='m', label='Magnetic Quantum Number m', type='number', value='0', min='-5', max='5', step='1'),
            TaskField(name='extent', label='Plot Extent (Å)', type='number', value='12', min='2', max='40', step='1'),
            TaskField(name='threshold', label='3D Density Threshold', type='number', value='0.15', min='0.01', max='0.9', step='0.01'),
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

    def hex_to_rgba(hex_color: str, opacity: float) -> str:
        """Convert a hex colour such as #4C72B0 into Plotly rgba format."""
        hex_color = hex_color.lstrip('#')

        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)

        return f'rgba({red}, {green}, {blue}, {opacity})'


    def ring_style(base_color: str, n: int, max_order: int) -> dict:
        """Make higher-order rings thinner and more transparent."""
        if max_order <= 1:
            relative_order = 0
        else:
            relative_order = (n - 1) / (max_order - 1)

        # First rings are strong; higher orders gradually fade.
        opacity = max(0.12, 0.9 - 0.75 * relative_order)
        width = max(0.5, 2.6 - 1.8 * relative_order)

        return {
            'color': hex_to_rgba(base_color, opacity),
            'width': width,
        }

    h = 6.626e-34
    e_ = 1.602e-19
    m_ = 9.109e-31

    d_values = [d1_nm * 1e-9, d2_nm * 1e-9]
    d_labels = [f'd\u2081={d1_nm} nm', f'd\u2082={d2_nm} nm']
    colors = ['#4C72B0', '#DD8452']
    V_max = v_max_kv * 1000
    lambda_min = h / math.sqrt(2 * m_ * e_ * V_max)

    # Maximum order allowed by Bragg's law:
    # n * lambda / (2d) <= 1
    max_orders = [
        math.floor(2 * d / lambda_min)
        for d in d_values
    ]

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
    trace_specs = [
        (d, color, label, n, max_order)
        for d, color, label, max_order in zip(
            d_values,
            colors,
            d_labels,
            max_orders,
        )
        for n in range(1, max_order + 1)
    ]

    frames = []

    for V_kv in voltages_kv:
        V = V_kv * 1000
        traces = []

        for d, color, label, n, max_order in trace_specs:
            radius = ring_radius(V, d, n)

            if radius is None:
                xs, ys = [np.nan], [np.nan]
            else:
                xs = radius * np.cos(theta_grid)
                ys = radius * np.sin(theta_grid)

            style = ring_style(color, n, max_order)

            traces.append(go.Scatter(
                x=xs,
                y=ys,
                mode='lines',

                line=dict(
                    color=style['color'],
                    width=style['width'],
                ),

                # Keeps all orders for one spacing together.
                legendgroup=label,
                legendgrouptitle_text=label if n == 1 else None,

                # Do not fill the legend with every order.
                showlegend=n == 1,
                name=label,

                # Order is still shown when hovering.
                customdata=np.full(len(xs), n),
                hovertemplate=(
                    f'{label}<br>'
                    'Order: n=%{customdata}<br>'
                    'x: %{x:.2f} mm<br>'
                    'y: %{y:.2f} mm'
                    '<extra></extra>'
                ),
            ))

        frames.append(
            go.Frame(
                data=traces,
                name=f'{V_kv:.2f}',
            )
        )

    fig1 = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title='Electron Diffraction Rings on Phosphor Screen',

            template='plotly_white',

            xaxis=dict(
                range=[-r_mm, r_mm],
                scaleanchor='y',
                scaleratio=1,
                title='Horizontal position / mm',
                showgrid=True,
                gridcolor='rgba(0, 0, 0, 0.08)',
                zeroline=True,
                zerolinecolor='rgba(0, 0, 0, 0.25)',
            ),

            yaxis=dict(
                range=[-r_mm, r_mm],
                title='Vertical position / mm',
                showgrid=True,
                gridcolor='rgba(0, 0, 0, 0.08)',
                zeroline=True,
                zerolinecolor='rgba(0, 0, 0, 0.25)',
            ),

            shapes=[
                dict(
                    type='circle',
                    x0=-r_mm,
                    y0=-r_mm,
                    x1=r_mm,
                    y1=r_mm,
                    line=dict(
                        color='rgba(0, 100, 0, 0.55)',
                        width=2,
                    ),
                )
            ],

            legend=dict(
                title='Lattice spacings',
                groupclick='togglegroup',
                bgcolor='rgba(255, 255, 255, 0.8)',
            ),

            hovermode='closest',

            sliders=[dict(
                active=0,
                currentvalue={
                    'prefix': 'Accelerating voltage: ',
                    'suffix': ' kV',
                },
                steps=[
                    dict(
                        method='animate',
                        args=[
                            [f'{V_kv:.2f}'],
                            {
                                'mode': 'immediate',
                                'frame': {
                                    'duration': 0,
                                    'redraw': True,
                                },
                                'transition': {
                                    'duration': 0,
                                },
                            },
                        ],
                        label=f'{V_kv:.1f}',
                    )
                    for V_kv in voltages_kv
                ],
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


def task10_plot(
    z: int,
    mass_number: int,
    n: int,
    l: int,
    m: int,
    extent_angstrom: float,
    threshold: float,
) -> list[str]:
    """Plot radial, 2D-slice and 3D probability density for a hydrogenic orbital."""

    if n < 1:
        raise ValueError('n must be at least 1.')
    if l < 0 or l >= n:
        raise ValueError('l must satisfy 0 ≤ l ≤ n - 1.')
    if abs(m) > l:
        raise ValueError('m must satisfy -l ≤ m ≤ l.')
    if z < 1:
        raise ValueError('Z must be at least 1.')
    if mass_number < z:
        raise ValueError('A must be greater than or equal to Z.')

    bohr_radius_angstrom = physical_constants['Bohr radius'][0] / 1e-10
    nuclear_mass = mass_number * atomic_mass
    reduced_mass = electron_mass * nuclear_mass / (electron_mass + nuclear_mass)
    a = electron_mass * bohr_radius_angstrom / (reduced_mass * z)

    def radial_wavefunction(r: np.ndarray) -> np.ndarray:
        x = 2.0 * r / (a * n)
        order = n - l - 1
        laguerre = eval_genlaguerre(order, 2 * l + 1, x)
        normalisation = np.sqrt(
            math.factorial(order) / (2.0 * n * math.factorial(n + l))
        ) * (2.0 / (a * n)) ** 1.5
        return normalisation * np.exp(-x / 2.0) * x**l * laguerre

    def real_spherical_harmonic(
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """
        Return a real spherical harmonic.

        theta:
            Polar angle measured from the positive z-axis.

        phi:
            Azimuthal angle in the x-y plane.

        l and m are taken automatically from task10_plot().
        """

        if m == 0:
            return np.real(
                sph_harm_y(l, 0, theta, phi)
            )

        if m > 0:
            harmonic = sph_harm_y(
                l,
                m,
                theta,
                phi,
            )

            return (
                np.sqrt(2.0)
                * ((-1) ** m)
                * np.real(harmonic)
            )

        positive_m = abs(m)

        harmonic = sph_harm_y(
            l,
            positive_m,
            theta,
            phi,
        )

        return (
            np.sqrt(2.0)
            * ((-1) ** positive_m)
            * np.imag(harmonic)
        )

    orbital_letters = 'SPDFGH'
    orbital_name = f'{n}{orbital_letters[l] if l < len(orbital_letters) else f"(l={l})"}'

    # Radial probability density: r² |R(r)|²
    r = np.linspace(0.0, max(extent_angstrom, 1.0), 1400)
    R = radial_wavefunction(r)
    radial_probability = r**2 * np.abs(R)**2

    fig_radial = go.Figure()
    fig_radial.add_trace(go.Scatter(
        x=r,
        y=radial_probability,
        mode='lines',
        name=r'$r^2|R_{nl}(r)|^2$',
        line=dict(width=3),
        hovertemplate='r = %{x:.3f} Å<br>radial density = %{y:.4g}<extra></extra>',
    ))
    fig_radial.update_layout(
        title=f'Radial probability density: Z={z}, A={mass_number}, {orbital_name}, m={m}',
        xaxis_title='Radius r / Å',
        yaxis_title='Radial probability density',
        template='plotly_white',
    )

    # 2D x-z slice through y = 0
    points_2d = 260
    axis_2d = np.linspace(-extent_angstrom, extent_angstrom, points_2d)
    X, Z_grid = np.meshgrid(axis_2d, axis_2d)
    Y = np.zeros_like(X)
    radius = np.sqrt(X**2 + Y**2 + Z_grid**2)
    theta = np.zeros_like(radius)
    nonzero = radius > 0
    theta[nonzero] = np.arccos(np.clip(Z_grid[nonzero] / radius[nonzero], -1.0, 1.0))
    phi = np.arctan2(Y, X)
    psi = radial_wavefunction(radius) * real_spherical_harmonic(theta, phi)
    density_2d = np.abs(psi)**2
    if density_2d.max() > 0:
        density_2d /= density_2d.max()

    fig_slice = go.Figure(go.Heatmap(
        x=axis_2d,
        y=axis_2d,
        z=density_2d,
        colorscale='Viridis',
        colorbar=dict(title='Relative |ψ|²'),
        hovertemplate='x = %{x:.2f} Å<br>z = %{y:.2f} Å<br>|ψ|² = %{z:.4f}<extra></extra>',
    ))
    fig_slice.update_layout(
        title=f'Probability-density slice in the x-z plane: {orbital_name}, m={m}',
        xaxis_title='x / Å',
        yaxis_title='z / Å',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        template='plotly_white',
    )

    # 3D transparent volume, following the coloured-glass idea in the task.
    points_3d = 42
    axis_3d = np.linspace(-extent_angstrom, extent_angstrom, points_3d)
    X3, Y3, Z3 = np.meshgrid(axis_3d, axis_3d, axis_3d, indexing='ij')
    radius3 = np.sqrt(X3**2 + Y3**2 + Z3**2)
    theta3 = np.zeros_like(radius3)
    nonzero3 = radius3 > 0
    theta3[nonzero3] = np.arccos(np.clip(Z3[nonzero3] / radius3[nonzero3], -1.0, 1.0))
    phi3 = np.arctan2(Y3, X3)
    psi3 = radial_wavefunction(radius3) * real_spherical_harmonic(theta3, phi3)
    density3 = np.abs(psi3)**2
    max_density = float(density3.max())
    if max_density > 0:
        density3 /= max_density

    threshold = float(np.clip(threshold, 0.001, 0.95))
    fig_3d = go.Figure(go.Volume(
        x=X3.ravel(),
        y=Y3.ravel(),
        z=Z3.ravel(),
        value=density3.ravel(),
        isomin=threshold,
        isomax=1.0,
        opacity=0.12,
        surface_count=14,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title='Relative |ψ|²'),
        hovertemplate='x=%{x:.2f} Å<br>y=%{y:.2f} Å<br>z=%{z:.2f} Å<br>|ψ|²=%{value:.3f}<extra></extra>',
    ))
    fig_3d.update_layout(
        title=f'3D hydrogenic orbital density: Z={z}, A={mass_number}, {orbital_name}, m={m}',
        scene=dict(
            xaxis_title='x / Å',
            yaxis_title='y / Å',
            zaxis_title='z / Å',
            aspectmode='cube',
        ),
        template='plotly_white',
        margin=dict(l=0, r=0, b=0, t=55),
    )

    html_radial = pio.to_html(fig_radial, include_plotlyjs='cdn', full_html=False)
    html_slice = pio.to_html(fig_slice, include_plotlyjs=False, full_html=False)
    html_3d = pio.to_html(fig_3d, include_plotlyjs=False, full_html=False)
    return [html_radial, html_slice, html_3d]


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
    if task_id == 9:
        return task9_plot(
            form_data.get('photon_energies', '50, 100, 200, 500, 1000'),
            int(form_data.get('num_points', 200)),
        )
    if task_id == 10:
        return task10_plot(
            int(form_data.get('z', 1)),
            int(form_data.get('mass_number', 1)),
            int(form_data.get('n', 3)),
            int(form_data.get('l', 2)),
            int(form_data.get('m', 0)),
            float(form_data.get('extent', 12.0)),
            float(form_data.get('threshold', 0.15)),
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

    # Task 8 is fully interactive in the browser,
    # so it does not need a form submission.
    if task_id == 8:
        return render_template('task8.html', task=task)

    results = None

    if request.method == 'POST':
        results = generate_task_images(task_id, request.form)

    return render_template(
        'task.html',
        task=task,
        results=results,
        form=request.form,
    )

if __name__ == '__main__':
    app.run(debug=True)
