#!/usr/bin/python3

import graphics.engine
import numpy as np


def I_section(coor, x_axis, z_axis, h, b, nh=4, nb=2):
    dx = x_axis / np.linalg.norm(x_axis)
    dy = np.cross(z_axis, dx)
    dz = np.cross(dx, dy)

    dx /= np.linalg.norm(dx)
    dy /= np.linalg.norm(dy)
    dz /= np.linalg.norm(dz)
    # print(dx)
    # print(dy)
    # print(dz)

    coors = []
    # web 0, 1, 2, 3, 4
    for i in range(nh + 1):
        coors.append(coor + (i * h / nh - h/2) * dz)

    # lower left flange 5, 6
    # print(nb)
    for i in range(1, nb + 1):
        coors.append(coor - h / 2 * dz + i * b/nb * dy)

    # lower right flange 7, 8
    for i in range(1, nb + 1):
        coors.append(coor - h / 2 * dz - i * b/nb * dy)

    # upper left flange 9, 10
    for i in range(1, nb + 1):
        coors.append(coor + h / 2 * dz + i * b/nb * dy)

    # upper right flange 11, 12
    for i in range(1, nb + 1):
        coors.append(coor + h / 2 * dz - i * b/nb * dy)

    return np.array(coors, dtype=float)


def I_beam(start, end, zaxis, h, b, nh=4, nb=2, nl=10):
    len = np.linalg.norm(end - start)
    x = (end - start) / len
    z = zaxis / np.linalg.norm(zaxis)
    y = np.cross(z, x)
    z = np.cross(x, y)

    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)

    dx = len / nl * x

    coors = []
    for i in range(nl+1):
        coors.append(I_section(coor=start + i * dx, x_axis=x, z_axis=z, h=h, b=b, nh=nh, nb=nb))

    coors = np.array(coors, dtype=float).reshape(-1,3)
    # print(coors)

    num_section = int(coors.shape[0] / (nl + 1))
    # print(num_section)

    quads = []
    for i in range(nl):
        # web
        for j in range(nh):
            quads.append([i * num_section + j, (i + 1) * num_section + j,
                          (i + 1) * num_section + j + 1, i * num_section + j + 1])

        # lower left flange
        quads.append([i * num_section, (i + 1) * num_section,
                      (i + 1) * num_section + nh + 1, i * num_section + nh + 1])

        for j in range(nh + 1, nh + nb):
            quads.append([i * num_section + j, (i + 1) * num_section + j,
                          (i + 1) * num_section + j + 1, i * num_section + j + 1])

        # lower right flange
        quads.append([i * num_section, (i + 1) * num_section,
                      (i + 1) * num_section + nh + nb + 1, i * num_section + nh + nb + 1])

        for j in range(nh + nb + 1, nh + 2 * nb):
            quads.append([i * num_section + j, (i + 1) * num_section + j,
                          (i + 1) * num_section + j + 1, i * num_section + j + 1])

        # upper left flange
        quads.append([i * num_section + nh, (i + 1) * num_section + nh,
                      (i + 1) * num_section + nh + 2 * nb + 1, i * num_section + nh + 2 * nb + 1])

        for j in range(nh + 2 * nb + 1, nh + 3 * nb):
            quads.append([i * num_section + j, (i + 1) * num_section + j,
                          (i + 1) * num_section + j + 1, i * num_section + j + 1])

        # upper right flange
        quads.append([i * num_section + nh, (i + 1) * num_section + nh,
                      (i + 1) * num_section + nh + 3 * nb + 1, i * num_section + nh + 3 * nb + 1])

        for j in range(nh + 3 * nb + 1, nh + 4 * nb):
            quads.append([i * num_section + j, (i + 1) * num_section + j,
                          (i + 1) * num_section + j + 1, i * num_section + j + 1])

    quads = np.array(quads, dtype='int32').reshape(-1, 4)

    return coors, quads


def animation():
    model.clear()
    model.render()
    model.screen.after(100, animation)


if __name__ == '__main__':
    start = np.array([0,0,0], dtype=float)
    end = np.array([3000,0,0], dtype=float)
    xaxis = np.array([1,0,0], dtype=float)
    zaxis = np.array([0,1,0], dtype=float)
    points, elements = I_beam(start, end, zaxis, h=500., b=300., nh=4, nb=2, nl=25)

    deform = []
    for i in range(1):
        deform.append(np.zeros(points.shape, dtype=float))
        for j in range(points.shape[0]):
          deform[i][j,1] = (300. / 2.) * np.sin(points[j,0] / 1500. * np.pi / 2. * (i + 1.))

    deform = np.array(deform, dtype=float)

    model = graphics.engine.Engine3D(points, elements, displacement=deform, title='Cube', distance=6, num_steps=11, projection='ortho')

    animation()
    model.screen.window.mainloop()

