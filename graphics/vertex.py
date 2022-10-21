import math
import numpy as np

class Vertex:
    def __init__(self, point, displacement=None):
        """
        In:
            point        - (x, y, z)
            displacement - ((dx1, dx2, ... , dxn), (dy1, dy2, ... , dyn), (dz1, dz2, ... , dzn))
        """
        #store x, y, z coordinates
        self.X = np.array(point, dtype=float).reshape((3,1))

        if displacement is None:
            self.dX = np.zeros((3,1), dtype=float)
        else:
            self.dX = displacement.T

        if len(self.dX.shape) == 1:
            self.dX = self.dX.reshape((3,1))

    def __getitem__(self, dof, index=0):
        return np.array([self.X[dof,0], self.dX[dof,index]], dtype=float)

    @property
    def x(self):
        return self.X[0,0]

    @property
    def y(self):
        return self.X[1,0]

    @property
    def z(self):
        return self.X[2,0]

    def dx(self, index=0):
        return self.dX[0,index]

    def dy(self, index=0):
        return self.dX[1,index]

    def dz(self, index=0):
        return self.dX[2,index]

    def dist(self, Tr, Tt, ds=1., index=0):
        """
        returns point distance from screen
        In:
            Tr    - 3x3 rotations matrix
            Tt    - 3x1 translation matrix
            ds    - displacement scale (float)
            index - displacement load case index
        """
        Xl = Tr[2,:] @ (self.X + ds * self.dX[:,[index]] + Tt)

        return Xl[0]

    def rms(self, ds=1., index=0):
        """
        pythagorean displacement length (root mean square)
        In:
            ds    - displacement scale (float)
            index - displacement load case index
        """
        return (self.dX[:,[index]].T @ self.dX[:,[index]]) ** 0.5

    def flatten(self, scale, distance, Tr, Tt, ds=1., index=0):
        """
        returns point projection to screen

        Xloc  = T @ (Xg + ds * dXg)

        Ortho:
            Xproj = Xloc * scale
            projectedX, projectedY = Xproj[0], Xproj[1]

        Perspective:
            Xproj = (Xloc * distance) / (Xloc[2] + distance) * scale
            projectedX, projectedY = Xproj[0], Xproj[1]

        In:
            scale    - display scale
            distance - projection distance (if None or 0 = ortho, else perspective)
            Tr       - 3x3 rotations matrix
            Tt       - 3x1 translation matrix
            ds       - displacement scale (float)
        """
        # calculate rotated 3D coordinates
        X = Tr @ (self.X + ds * self.dX[:,[index]] + Tt)

        # calculate 2D coordinates from 3D point
        if distance is None or distance == 0:
            pX = (X[:2,:] * scale).astype('int32')
        else:
            pX = ((X[:2,:] * distance) / (X[2,0] + distance) * scale).astype('int32')
        return pX

    def move(self, axis, value):
        if axis == 'x':
            self.X[0,0] += value
        elif axis == 'y':
            self.X[1,0] += value
        elif axis == 'z':
            self.X[2,0] += value
        else:
            raise ValueError('Invalid movement axis')

