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
    pos_xs += vel_xs * .02
    pos_ys += vel_ys * .02
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
xi[0] = 2 * np.random.random(size) + 1

rand_radius = np.random.random(size) * 5
rand_theta =  np.random.random(size) * 2 * PI

xi[1] = np.cos(rand_theta) * rand_radius
xi[2] = np.sin(rand_theta) * rand_radius

xi[3] = ( xi[2]) / 100
xi[4] = (-xi[1]) / 100

figure, ax = plt.subplots(figsize=(29, 29))
plt.ion()
plot1, = ax.plot(xi[1], xi[2], "o", markersize=.7, alpha=.2, color='white')
ax.set_aspect('equal')

foldername = str(input("File save folder: "))
filesavename = str(input("File save name: "))
if not os.path.exists(foldername):
    os.makedirs(foldername)
else:
    print("Your selected folder already exists; rendered frames will be placed in that folder.")
render = (input("Render frames? ('True' or 'False'): "))
if render == 'True':
    render = True
elif render == 'False':
    render = False
print(render)
simsteps = int(input("Steps (will render into pngs if you selected 'True'): "))


value = 0
value1 = 0
while True:
    start = time()
    if value < simsteps:
        xi = gravitate(xi)
        plot1.set_xdata(xi[1])
        plot1.set_ydata(xi[2])
    print(time() - start)
    figure.canvas.draw()
    figure.canvas.flush_events()
    plt.show()
    if value < simsteps:
        if render==True:
            plt.savefig(f"{foldername}//{filesavename}" + str(value1))
            value1 += 1
        value += 1
    print(time() - start)
    if value >= simsteps:
        print("Enter 'r' into the prompt to choose a new starting state.")
        simsteps = (input("Steps to render:"))
        if simsteps == 'r':
            size = int(input("Number of bodies: "))
            rand_radius = float(input("Radius for galactic generation: "))
            vel_fac = float(input("Volcity multiplier"))
            rand_vels = float(input("Randomized velocity multiplier: "))
            simsteps = int(input("Steps to render:"))
            render = (input("Output frames as images? ('True' or 'False': "))
            if render == 'True':
                render = True
            if render == 'False':
                render = False
            if render:
                foldername = str(input(f"File save folder (current is {foldername}): "))
                filesavename = str(input(f"File save name (current is {filesavename}): "))
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                else:
                    print("Your selected folder already exists; rendered frames will be placed in that folder.")
            xi = np.ones((5, size))
            xi[0] = 2 * np.random.random(size) + 1

            rand_radius = np.random.random(size) * rand_radius
            rand_theta = np.random.random(size) * 2 * PI

            xi[1] = np.cos(rand_theta) * rand_radius
            xi[2] = np.sin(rand_theta) * rand_radius

            #xi[3] = (((np.random.random(size) - .5) * rand_vels) + (vel_fac*xi[2]) / 100)
            #xi[4] = (((np.random.random(size) - .5) * rand_vels) + (vel_fac*(-xi[1])) / 100)
            xi[3] = (xi[2] / np.sqrt((xi[1]**2)+(xi[2]**2))) * np.sqrt(G_CONST*np.sqrt((xi[1]**2)+(xi[2]**2))*size)
            xi[4] =(-xi[1] / np.sqrt((xi[1]**2)+(xi[2]**2))) * np.sqrt(G_CONST*np.sqrt((xi[1]**2)+(xi[2]**2))*size)
        simsteps = int(simsteps)
        value = 0

