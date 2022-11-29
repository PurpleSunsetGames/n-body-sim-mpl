import moderngl.buffer
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
from math import pi as PI
G_CONST = .000006

plt.style.use('dark_background')

import moderngl
import struct

ctx = moderngl.create_context(standalone=True)
program = ctx.program(
    vertex_shader=str(open("programs/vertex_shader.glsl").read()),
    varyings=["out_poss", "out_vels", "out_masses"]
)

@njit(parallel=True)
def gravitate(g_info, vel_mult):
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
                    d = np.sqrt(((px - cx)**2) + ((py - cy)**2))
                    dcube = d ** 2
                    dx = (G_CONST * (cx - px) * (pm + cm)) / dcube
                    dy = (G_CONST * (cy - py) * (pm + cm)) / dcube
                    vel_xs[particle] += (dx / pm)
                    vel_ys[particle] += (dy / pm)
                except:
                    print(comp, particle, px, py, cx, cy, pm, cm)
            else:
                pass
    pos_xs += vel_xs * vel_mult
    pos_ys += vel_ys * vel_mult
    g_info[0] = masses
    g_info[1] = pos_xs
    g_info[2] = pos_ys
    g_info[3] = vel_xs
    g_info[4] = vel_ys
    return g_info


# masses, px, py, vx, vy
xi = np.random.random((5, 1))
start = time()
gravitate(xi, 1)
print(time() - start)

size = 1000
xi = np.ones((5, size))
#masses
xi[0] = 2 * np.random.random(size) + 1

rand_radius = np.random.random(size) * 25
rand_theta =  np.random.random(size) * 2 * PI
#positions
xi[1] = np.cos(rand_theta) * rand_radius
xi[2] = np.sin(rand_theta) * rand_radius
#velocities
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
vel_fac = 1
while True:
    start = time()
    if value < simsteps:
        #xi = gravitate(xi, vel_fac)
        comp_num = 0
        NUM_VERTICES = len(xi[1])
        xp = xi[1]
        yp = xi[2]
        xv = xi[3]
        yv = xi[4]
        m = xi[0]
        while comp_num < NUM_VERTICES:
            vertices = np.dstack([xp, yp, xv, yv, m,
                                  np.zeros(NUM_VERTICES) + xp[comp_num],
                                  np.zeros(NUM_VERTICES) + yp[comp_num],
                                  np.zeros(NUM_VERTICES) + xv[comp_num],
                                  np.zeros(NUM_VERTICES) + yv[comp_num],
                                  np.zeros(NUM_VERTICES) + m[comp_num]])

            vbo = ctx.buffer(vertices.astype('f4').tobytes())

            vao = ctx.vertex_array(program, vbo, "in_poss", "in_vels", "in_masses", "comp_pos", "comp_vel", "comp_mass")

            vao.transform(vbo, vertices=NUM_VERTICES)

            # unpacking 'number of points times number of attributes per point' floats
            data = np.array(struct.unpack(f'{NUM_VERTICES * 10}f', vbo.read())).reshape((NUM_VERTICES * 2, 5))[:NUM_VERTICES]
            xi[3] += np.swapaxes(data, 0, 1)[2]
            xi[4] += np.swapaxes(data, 0, 1)[3]
            comp_num += 1
        xi[1] = xi[1] + xi[3] / xi[0]
        xi[2] = xi[2] + xi[4] / xi[0]
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
            vel_fac = float(input("Velocity multiplier: "))
            window_range = float(input("View range: "))
            figure, ax = plt.subplots(figsize=(window_range, window_range))
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
            xi[3] = (xi[2] / np.sqrt((xi[1]**2)+(xi[2]**2))) * np.sqrt(G_CONST*np.sqrt((xi[1]**2)+(xi[2]**2))*size*2)
            xi[4] =(-xi[1] / np.sqrt((xi[1]**2)+(xi[2]**2))) * np.sqrt(G_CONST*np.sqrt((xi[1]**2)+(xi[2]**2))*size*2)
        simsteps = int(simsteps)
        value = 0
    ax.set_aspect('equal')

