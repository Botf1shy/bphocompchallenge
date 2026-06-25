import matplotlib.pyplot as plt
import numpy as np
import random

def main():
    step_size = int(input("s:"))
    num_steps = int(input("N:"))
    num_randomwalks = int(input("Number of random walks:"))
    x, y = 0, 0
    fig, ax = plt.subplots()
    ax.set(xlabel='X', ylabel='Y', title=f'Random Walk: Step Size = {step_size}, Number of Steps = {num_steps}')
    ax.grid()

    for i in range(num_randomwalks):
        color = (random.random(), random.random(), random.random())
        xs, ys = [x], [y]
        for _ in range(num_steps):
            x, y = random_walk(x, y, step_size)
            xs.append(x); ys.append(y)
            ax.plot(xs[-2:], ys[-2:],'-o', color=color, markersize=0)

    plt.show()

def random_walk(x, y, step_size):
    angle = random.uniform(0, 2 * np.pi)
    x += step_size * np.cos(angle)
    y += step_size * np.sin(angle)
    return x, y

if __name__ == "__main__":
    main()