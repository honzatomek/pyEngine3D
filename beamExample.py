#!/usr/bin/python3

### Beam Structure ###

import graphics.engine
import numpy as np


def cube_beam(width, depth, height, m, n, o):
  dx = width / m
  dy = depth / n
  dz = height / o

  # nodes and coors
  coors = []
  nodes = []
  for k in range(o+1):
    for j in range(n+1):
      for i in range(m+1):
        nodes.append(i + (m+1)*j + (m+1)*(n+1)*k)
        coors.append([i*dx, j*dy, k*dz])

  # connectivity beam
  lme = []
  for k in range(o):
    for j in range(n):
      for i in range(m):
        n1 = i + (m+1)*j + (m+1)*(n+1)*k
        n2 = i + (m+1)*j + (m+1)*(n+1)*k + 1
        n3 = i + (m+1)*(j+1) + (m+1)*(n+1)*k + 1
        n4 = i + (m+1)*(j+1) + (m+1)*(n+1)*k
        n5 = i + (m+1)*j + (m+1)*(n+1)*(k+1)
        n6 = i + (m+1)*j + (m+1)*(n+1)*(k+1) + 1
        n7 = i + (m+1)*(j+1) + (m+1)*(n+1)*(k+1) + 1
        n8 = i + (m+1)*(j+1) + (m+1)*(n+1)*(k+1)
        lme.extend([[n1,n2],[n2,n3],[n3,n4],[n4,n1]]) # bottom face
        lme.extend([[n5,n6],[n6,n7],[n7,n8],[n8,n5]]) # top face
        lme.extend([[n1,n5],[n2,n6],[n3,n7],[n4,n8]]) # vertical edges
        lme.extend([[n1,n6],[n2,n7],[n3,n8],[n4,n5],[n1,n7],[n1,n3],[n5,n7]]) # diagonals

  return np.array(coors, dtype=float), np.array(lme, dtype='int32')

x = 100.0
y = 20.0
z = 20.0
nx = 10
ny = 4
nz = 4
points, lines = cube_beam(x, y, z, nx, ny, nz)

deform = []
for i in range(10):
  deform.append(np.zeros(points.shape, dtype=float))
  for j in range(points.shape[0]):
    deform[i][j,1] = (y / 2) * np.sin(points[j,0] / x * np.pi / 2 * (i + 1))

deform = np.array(deform, dtype=float)

test = graphics.engine.Engine3D(points, lines, displacement=deform, title='Cube', distance=6, num_steps=11, projection='ortho')

def animation():
    test.clear()
    test.rotate('y', 0.01)
    test.rotate('x', 0.01)
    test.render()
    test.screen.after(1, animation)


def animation2():
    test.clear()
    # test.rotate('y', 0.01)
    # test.rotate('x', 0.01)
    test.render()
    test.screen.after(100, animation2)

animation2()
test.screen.window.mainloop()

############
