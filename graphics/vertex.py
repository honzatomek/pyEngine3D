import math
import numpy as np

class Vertex:
    def __init__(self, point, displacement=None):
        #store x, y, z coordinates
        (x,y,z) = point
        self.x = x
        self.y = y
        self.z = z

        if displacement is None:
            self.dx = 0.
            self.dy = 0.
            self.dz = 0.
        else:
            self.dx = displacement[0]
            self.dy = displacement[1]
            self.dz = displacement[2]

    def d(self, Tr, Tt, ds=0.):
        """
        returns point distance from screen
        In:
            Tr - 3x3 rotations matrix
            Tt - 3x1 translation vector
            ds - displacement scale (float)
        """
        # calculate rotated 3D coordinates
        X = Tr[2,:] @ np.array([self.x + ds * self.dx, self.y + ds * self.dy, self.z + ds * self.dz]).reshape((3, 1)) + Tt[2,0]

        return X[0]

    def square(self, ds=0.):
        """
        pythagorean displacement length
        In:
            ds - displacement scale (float)
        """
        return (self.dx ** 2 + self.dy ** 2 + self.dz ** 2) ** 0.5

    def flatten(self, scale, distance, Tr, Tt, ds=0.):
        """
        returns point projection to screen
        In:
            Tr - 3x3 rotations matrix
            Tt - 3x1 translation vector
            ds - displacement scale (float)
        """
        # calculate rotated 3D coordinates
        X = Tr @ np.array([self.x + ds * self.dx, self.y + ds * self.dy, self.z + ds * self.dz]).reshape((3, 1)) + Tt

        # calculate 2D coordinates from 3D point
        if distance is None or distance == 0:
            projectedX = int(X[0,0] * scale)
            projectedY = int(X[1,0] * scale)
        else:
            projectedX = int(((X[0,0] * distance) / (X[2,0] + distance)) * scale)
            projectedY = int(((X[1,0] * distance) / (X[2,0] + distance)) * scale)
        return (projectedX, projectedY)

    def move(self, axis, value):
        if axis == 'x':
            self.x += value
        elif axis == 'y':
            self.y += value
        elif axis == 'z':
            self.z += value
        else:
            raise ValueError('Invalid movement axis')

