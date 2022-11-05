#!/usr/bin/python3

import numpy as np
import graphics.engine

points = np.array([[0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]], dtype=float)

lines = [[0, 1],
         [1, 2],
         [2, 0],
         [0, 3],
         [1, 3],
         [2, 3]]

mass = [[0], [1], [2], [3]]


a = 0.58541020
b = 0.13819660
integration_points = np.array([[a, b, b, b],
                               [b, a, b, b],
                               [b, b, a, b],
                               [b, b, b, a]], dtype=float)

ip = integration_points @ points
points = np.vstack((points, ip))

ip = [[4], [5], [6], [7]]

elements = lines + mass + ip

displacements = np.hstack((np.ones((points.shape[0], 1), dtype=float) * 0.1,
                           np.zeros((points.shape[0], 1), dtype=float),
                           np.zeros((points.shape[0], 1), dtype=float))
                         ).reshape(1, points.shape[0], points.shape[1])

test = graphics.engine.Engine3D(points, elements, displacements, title='TET4', distance=0)

def animate():
    test.clear()
    test.render()
    test.screen.after(100, animate)

animate()
test.screen.window.mainloop()

