#!/usr/bin/python3

### Beam Structure ###

import graphics.engine
import numpy as np

def cube_hexa(width, depth, height, m, n, o):
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

  # connectivity hexa
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
        lme.append([n1, n2, n3, n4, n5, n6, n7, n8])

  return coors, lme


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
        lme.extend([[n1,n6],[n2,n7],[n3,n8],[n4,n5],[n1,n7],[n4,n6]]) # diagonals

  return coors, lme

points, lines = cube_beam(0.60, 2.00, 0.60, 3, 10, 3)
points = np.array(points, dtype=float)
points[:,0] -= 0.3
points[:,1] -= 1.
points[:,2] -= 0.3
print(np.min(points.flatten()))
print(np.max(points.flatten()))
points = points.tolist()
test = graphics.engine.Engine3D(points, lines=lines, title='Cube')

def animation():
    test.clear()
    test.rotate('y', 0.01)
    test.rotate('x', 0.01)
    test.render()
    test.screen.after(1, animation)

animation()
test.screen.window.mainloop()

############
