"""
Brownian motion simulation converted from the supplied MATLAB code.

The model:
- Simulates one large circular particle and N smaller particles in 2D.
- Ignores collisions between small particles.
- Updates only collisions between a small particle and the large particle.
- Randomises the directions of the small-particle velocities after a
  Knudsen-number-based time interval.
- Uses matplotlib for animation and saves the final frame as a PNG.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt


def ball_displacement(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[np.ndarray, float]:
    """
    Return the unit vector from particle 1 to particle 2 and their separation.
    """
    displacement = np.array([x2 - x1, y2 - y1], dtype=float)
    distance = np.linalg.norm(displacement)

    if distance == 0:
        # Extremely unlikely, but avoids division by zero.
        direction = np.array([1.0, 0.0])
    else:
        direction = displacement / distance

    return direction, distance


def bounce(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ux1: float,
    ux2: float,
    uy1: float,
    uy2: float,
    coefficient_of_restitution: float,
    mass1: float,
    mass2: float,
    radius1: float,
    radius2: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Update the positions and velocities of two colliding circular particles.

    This follows the supplied MATLAB implementation directly.
    """
    u1 = np.array([ux1, uy1], dtype=float)
    u2 = np.array([ux2, uy2], dtype=float)

    vx1, vy1 = ux1, uy1
    vx2, vy2 = ux2, uy2

    direction, distance = ball_displacement(x1, y1, x2, y2)

    if distance <= radius1 + radius2:
        # Move overlapping particles apart so that their centres are separated
        # by exactly radius1 + radius2.
        delta = (radius1 + radius2 - distance) / 2.0

        r1 = np.array([x1, y1]) - delta * direction
        r2 = np.array([x2, y2]) + delta * direction

        x1, y1 = r1
        x2, y2 = r2

        # A collision is processed only if the particles are approaching.
        if np.dot(u2 - u1, direction) < 0:
            centre_of_mass_velocity = (
                mass1 * u1 + mass2 * u2
            ) / (mass1 + mass2)

            v1 = (
                centre_of_mass_velocity
                - coefficient_of_restitution
                * (u1 - centre_of_mass_velocity)
            )

            v2 = (
                centre_of_mass_velocity
                - coefficient_of_restitution
                * (u2 - centre_of_mass_velocity)
            )

            vx1, vy1 = v1
            vx2, vy2 = v2

    return vx1, vy1, vx2, vy2, x1, x2, y1, y2


def brownian_motion() -> None:
    rng = np.random.default_rng()

    # Number of small particles
    N = 1000

    # Temperature in Celsius
    temperature_c = 100.0

    # Mass of a small particle, e.g. an air molecule, in kg
    m = 28.96e-3 / 6.02e23

    # Mass of the large particle
    M = 10.0 * m

    # Radii in nm
    r = 0.16
    R = 10.0 * r

    # Size of the square modelling region
    a = 7.0 * R

    # Coefficient of restitution
    C = 1.0

    # Boltzmann constant in J/K
    k_B = 1.38e-23

    # Average thermal speeds in m/s
    v = math.sqrt(
        3.0 * k_B * (temperature_c + 273.0) / m
    )

    V = math.sqrt(
        3.0 * k_B * (temperature_c + 273.0) / M
    )

    # Knudsen number
    Kn = 15.0

    # Maximum simulation time in ps
    t_max = 200.0

    # Convert m/s to nm/ps
    v /= 1000.0
    V /= 1000.0

    # Time step in ps
    dt = 0.01 * Kn * r / v

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2 * a, 1.2 * a)
    ax.set_ylim(-0.2 * a, 1.2 * a)
    ax.set_axis_off()

    ax.set_title(
        "Brownian motion simulation: t = 0 ps",
        fontsize=18,
    )

    # Draw the square box
    ax.plot(
        [0, a, a, 0, 0],
        [0, 0, a, a, 0],
        "k-",
        linewidth=3,
    )

    stop_stop = False

    while not stop_stop:
        # Initial position of the large particle
        X = 0.5 * a
        Y = 0.5 * a

        # Initial velocity of the large particle
        theta = rng.uniform(0.0, 2.0 * math.pi)

        Vx = V * math.cos(theta)
        Vy = V * math.sin(theta)

        # Initial positions of the small particles
        x = np.empty(N)
        y = np.empty(N)

        for n in range(N):
            while True:
                candidate_x = (
                    r + rng.random() * (a - 2.0 * r)
                )

                candidate_y = (
                    r + rng.random() * (a - 2.0 * r)
                )

                _, distance = ball_displacement(
                    candidate_x,
                    candidate_y,
                    X,
                    Y,
                )

                if distance >= r + R:
                    x[n] = candidate_x
                    y[n] = candidate_y
                    break

        # Initial velocities of small particles
        theta = rng.uniform(
            0.0,
            2.0 * math.pi,
            size=N,
        )

        vx = v * np.cos(theta)
        vy = v * np.sin(theta)

        # Coordinates of the large circular particle
        angle = np.linspace(
            0.0,
            2.0 * math.pi,
            500,
        )

        xc = R * np.cos(angle)
        yc = R * np.sin(angle)

        # Plot the large particle
        large_line, = ax.plot(
            X + xc,
            Y + yc,
            "r-",
            linewidth=2,
        )

        # Plot the small particles
        small_points, = ax.plot(
            x,
            y,
            "b*",
            markersize=4,
        )

        # Store the path of the large particle
        trail_x = [X]
        trail_y = [Y]

        trail_line, = ax.plot(
            trail_x,
            trail_y,
            "r-",
            linewidth=1,
            alpha=0.7,
        )

        # Mark the initial position
        start_point, = ax.plot(
            [X],
            [Y],
            "g*",
            markersize=8,
        )

        # Mark the latest position
        latest_point, = ax.plot(
            [X],
            [Y],
            "r*",
            markersize=8,
        )

        # Step through time
        t = 0.0
        time_since_randomisation = 0.0
        stop = False

        plt.show(block=False)

        while not stop:
            # Update time
            t += dt
            time_since_randomisation += dt

            # Update the position of the large particle
            X += Vx * dt
            Y += Vy * dt

            # Update positions of all small particles
            x += vx * dt
            y += vy * dt

            # Add current large-particle position to the trail
            trail_x.append(X)
            trail_y.append(Y)

            # Check collisions between each small particle
            # and the large particle
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
                    X,
                    Y,
                    x[n],
                    y[n],
                    Vx,
                    vx[n],
                    Vy,
                    vy[n],
                    C,
                    M,
                    m,
                    R,
                    r,
                )

            # Randomise the directions of small-particle velocities
            if time_since_randomisation > Kn * r / v:
                time_since_randomisation = 0.0

                theta = rng.uniform(
                    0.0,
                    2.0 * math.pi,
                    size=N,
                )

                vx = v * np.cos(theta)
                vy = v * np.sin(theta)

            # Update the plotted large particle
            large_line.set_data(
                X + xc,
                Y + yc,
            )

            # Update the plotted small particles
            small_points.set_data(
                x,
                y,
            )

            # Update the trail
            trail_line.set_data(
                trail_x,
                trail_y,
            )

            # Update the latest-position marker
            latest_point.set_data(
                [X],
                [Y],
            )

            # Update the title
            ax.set_title(
                f"Brownian motion simulation: "
                f"t = {t:.3f} ps",
                fontsize=18,
            )

            # Redraw the animation
            fig.canvas.draw_idle()
            plt.pause(0.001)

            # Stop when maximum time is reached
            if t > t_max:
                stop = True

                fig.savefig(
                    "brownian_motion.png",
                    dpi=300,
                    bbox_inches="tight",
                )

        large_line.remove()
        small_points.remove()
        trail_line.remove()
        start_point.remove()
        latest_point.remove()

        stop_stop = True

    plt.show()


if __name__ == "__main__":
    brownian_motion()