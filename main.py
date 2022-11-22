from numba import njit, prange, jit
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
from math import pi as PI
G_CONST = .000006

plt.style.use('dark_background')


@njit(parallel=True)
def gravitate(g_info):
    masses = g_info[0]
    pos_xs = g_info[1]
    pos_ys = g_info[2]
    vel_xs = g_info[3]
    vel_ys = g_info[4]
    for particle in prange(len(masses)):
        for comp in prange(len(masses)):
            px = pos_xs[particle]
            py = pos_ys[particle]
            pm = masses[particle]
            cx = pos_xs[comp]
            cy = pos_ys[comp]
            cm = masses[comp]
            if comp != particle:
                try:
                    d = np.sqrt(((px - cx) * (px - cx)) + ((py - cy) * (py - cy)))
                    dcube = d ** 2
                    dx = (G_CONST * (cx - px) * (pm + cm)) / dcube
                    dy = (G_CONST * (cy - py) * (pm + cm)) / dcube
                    vel_xs[particle] += (dx / pm)
                    vel_ys[particle] += (dy / pm)
                except:
                    print(comp, particle, px, py, cx, cy, pm, cm)
            else:
                pass
        pos_xs[particle] += vel_xs[particle]
        pos_ys[particle] += vel_ys[particle]
    g_info[0] = masses
    g_info[1] = pos_xs
    g_info[2] = pos_ys
    g_info[3] = vel_xs
    g_info[4] = vel_ys
    return g_info


# masses, px, py, vx, vy
xi = np.random.random((5, 1))
start = time()
gravitate(xi)
print(time() - start)

size = 1000
xi = np.ones((5, size))
xi[0] = np.ones(size)

rand_radius = np.random.random(size) * 5
rand_theta =  np.random.random(size) * 2 * PI

xi[1] = np.cos(rand_theta) * rand_radius
xi[2] = np.sin(rand_theta) * rand_radius

xi[3] = ((np.random.random(size) - .5) * .1) + ( xi[2]) / 66
xi[4] = ((np.random.random(size) - .5) * .1) + (-xi[1]) / 66

figure, ax = plt.subplots(figsize=(9, 9))
plt.ion()
plot1, = ax.plot(xi[1], xi[2], "o", markersize=2, alpha=.3, color='white')

#os.makedirs("alot5")

for value in range(1500):
    start = time()
    xi = gravitate(xi)
    print(time() - start)

    plot1.set_xdata(xi[1])
    plot1.set_ydata(xi[2])

    figure.canvas.draw()
    figure.canvas.flush_events()
    plt.show()
    #plt.savefig("alot5//ALOT" + str(value))
    print(time() - start)