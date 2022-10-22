import graphics.screen
import graphics.face
import graphics.vertex
import copy
import sys

import numpy as np


PALETTE = np.array([[  0.,   0., 255.],
                    [  0., 255.,   0.],
                    [255., 255.,   0.],
                    [255.,   0.,   0.],
                    [255.,   0., 255.]], dtype=float)

# PALETTE = np.array([[   0,   0, 255],
#                     [ 255, 255, 255],
#                     [ 255,   0,   0]], dtype='int32')


def _from_rgb(rgb):
    """
    translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


class Engine3D:
    def __resetDrag(self, event):
        self.__prev = []
        self.__restore_deform(event)

    def __drag(self, event):
        if self.__prev:
            # self.rotate('y', (event.x - self.__prev[0]) / 20)
            # self.rotate('x', (event.y - self.__prev[1]) / 20)
            self.rotate_view('y', (self.__prev[0] - event.x) / 20)
            self.rotate_view('x', (self.__prev[1] - event.y) / 20)
            self.clear()
            deform = self.deform
            self.deform = False
            self.render()
            self.deform = deform
        self.__prev = [event.x, event.y]
        self.__save_deform(event)

    def __select(self, event):
        zeros = self.screen.zeros
        event = (event.x - zeros[0], event.y - zeros[1])

        possibilities = []
        for a in range(-6, 5):
            for b in range(-6, 5):
                possibilities.append((event[0] + a, event[1] + b))

        found = [e for e in possibilities if e in self.flattened]
        if found != []:
            self.__moveaxis = None
            # self.__selected = self.flattened.index(found[0])
            self.__selected = np.where(self.flattened==found[0])[0]

            i = self.points[self.__selected]
            self.__axis = [[copy.deepcopy(i) for a in range(2)] for b in range(3)]

            self.__axis[0][0].x -= 40 / self.scale
            self.__axis[0][1].x += 40 / self.scale
            self.__axis[1][0].y -= 40 / self.scale
            self.__axis[1][1].y += 40 / self.scale
            self.__axis[2][0].z -= 40 / self.scale
            self.__axis[2][1].z += 40 / self.scale

            self.__axis = [[point.flatten(self.scale, self.distance, self.Tr, self.Tt) for point in i] for i in self.__axis]
            self.__axis = [[[i[0] + zeros[0], i[1] + zeros[1]] for i in j] for j in self.__axis]
            self.__axis = [self.screen.createLine(self.__axis[0], 'red'), self.screen.createLine(self.__axis[1], 'green'), self.screen.createLine(self.__axis[2], 'blue')]

    def __selectx(self, event):
        self.__moveaxis = 'x'

    def __selecty(self, event):
        self.__moveaxis = 'y'

    def __selectz(self, event):
        self.__moveaxis = 'z'

    def __moveup(self, event):
        if self.__selected != None and self.__moveaxis != None:
            self.points[self.__selected].move(self.__moveaxis, 0.1)
            self.clear()
            deform = self.deform
            self.deform = False
            self.render()
            self.deform = deform

    def __movedown(self, event):
        if self.__selected != None and self.__moveaxis != None:
            self.points[self.__selected].move(self.__moveaxis, -0.1)
            self.clear()
            deform = self.deform
            self.deform = False
            self.render()
            self.deform = deform

    def __zoomin(self, event):
        # self.scale += 2.5
        self.scale += self.scale_step
        self.clear()
        deform = self.deform
        self.deform = False
        self.render()
        self.deform = deform

    def __zoomout(self, event):
        # self.scale -= 2.5
        self.scale -= self.scale_step
        self.clear()
        deform = self.deform
        self.deform = False
        self.render()
        self.deform = deform

    def __deselect(self, event):
        if self.__selected != None:
            self.__selected = None
            self.__axis = [self.screen.delete(line) for line in self.__axis]
            self.__moveaxis = None

    def __cameraleft(self, event):
        self.screen.zeros[0] -= 5
        self.clear()
        deform = self.deform
        self.deform = False
        self.render()
        self.deform = deform

    def __cameraright(self, event):
        self.screen.zeros[0] += 5
        self.clear()
        deform = self.deform
        self.deform = False
        self.render()
        self.deform = deform

    def __cameraup(self, event):
        self.screen.zeros[1] -= 5
        self.clear()
        deform = self.deform
        self.deform = False
        self.render()
        self.deform = deform

    def __cameradown(self, event):
        self.screen.zeros[1] += 5
        self.clear()
        deform = self.deform
        self.deform = False
        self.render()
        self.deform = deform

    def __deform(self, event):
        self.deform = not self.deform

    def get_model_extents(self, points):
        Xmin = points.min(axis=0)
        Xmax = points.max(axis=0)
        return np.vstack((Xmin, Xmax, 0.5 * (Xmin + Xmax)))

    def next_tstep(self):
        self.tstep += 1
        if self.tstep > self.num_steps:
            self.tstep = 0
        elif self.tstep < 0:
            self.tstep = self.num_steps

    def __next_tstep(self, event=None):
        if event is not None:
            self.deform = False
        self.tstep += 1
        if self.tstep > self.num_steps:
            self.tstep = 0
        elif self.tstep < 0:
            self.tstep = self.num_steps
        self.clear()
        self.render()

    def __prev_tstep(self, event=None):
        if event is not None:
            self.deform = False
        self.tstep -= 1
        if self.tstep > self.num_steps:
            self.tstep = 0
        elif self.tstep < 0:
            self.tstep = self.num_steps
        self.clear()
        self.render()

    def displacement_scale(self):
        scale = np.sin((self.tstep / (self.num_steps + 1)) * 2 * np.pi)
        return scale

    def palette_range(self):
        dmin = np.mean([self.points[vertex].rms(1.) for vertex in self.elements[0].v])
        dmax = dmin
        for i in range(self.elements.shape[0]):
            valmax = np.mean([self.points[vertex].rms(1.) for vertex in self.elements[i].v])
            valmin = np.mean([self.points[vertex].rms(0.) for vertex in self.elements[i].v])
            dmax = max(dmax, valmin, valmax)
            dmin = min(dmax, valmin, valmax)
            # dmin = val if val < dmin else dmin
            # dmax = val if val > dmax else dmax
        # for i in range(self.points.shape[0]):
        #     val = self.points[i].rms(1.)
        #     dmin = val if val < dmin else dmin
        #     dmax = val if val > dmax else dmax
        return np.array([dmin, dmax], dtype=float)

    def color(self, fraction):
        n = len(self.palette)
        if fraction >= 1.:
            i = n - 2
        else:
            i = int((n - 1) * fraction)
        col1 = self.palette[i]
        col2 = self.palette[i+1]
        f = (fraction - i / (n - 1)) * (n - 1)

        col = (int(col1[0] + (col2[0] - col1[0]) * f), int(col1[1] + (col2[1] - col1[1]) * f), int(col1[2] + (col2[2] - col1[2]) * f))
        # print(f'{col} - #{col[0]:02x}{col[1]:02x}{col[2]:02x}')

        # return f'#{col[0]:02x}{col[1]:02x}{col[2]:02x}'
        return col

    def writePoints(self, points, displacement):
        self.points = np.array([graphics.vertex.Vertex(points[i,:], displacement[:,i,:]) for i in range(points.shape[0])])

    def writeElements(self, elements):
        self.elements = np.array([graphics.face.Element(elements[i,:]) for i in range(elements.shape[0])])

    def writeLines(self, lines):
        # self.elements = []
        for line in lines:
            if len(line) != 4:
                line.append('gray')
            self.elements.append(graphics.face.Line(line))

    def writeTriangles(self, triangles):
        # self.elements = []
        for triangle in triangles:
            if len(triangle) != 4:
                triangle.append('gray')
            self.elements.append(graphics.face.Face3(triangle))

    def writeQuads(self, quads):
        # self.quads = []
        for quad in quads:
            if len(quad) != 5:
                quad.append('gray')
            self.elements.append(graphics.face.Face4(quad))

    def __save_deform(self, event=None):
        if self.__pressed == 0:
            if event.keysym != 'p':
                self.__deform = self.deform
                self.deform = False
            self.__pressed = 1

    def __restore_deform(self, event=None):
        if self.__pressed == 1:
            if event.keysym != 'p':
                self.deform = self.__deform
            self.__pressed = 0

    def initial_scale(self, width, height):
        return 0.6 * min(width, height) / max(self.extents[1,0]-self.extents[0,0], self.extents[1,1]-self.extents[0,1], self.extents[1,2]-self.extents[0,2])

    def __rotate_xp(self, event):
        self.rotate_model('x', 0.08726646259971647)
        self.clear()
        self.render()

    def __rotate_xn(self, event):
        self.rotate_model('x', -0.08726646259971647)
        self.clear()
        self.render()

    def __rotate_yp(self, event):
        self.rotate_model('y', 0.08726646259971647)
        self.clear()
        self.render()

    def __rotate_yn(self, event):
        self.rotate_model('y', -0.08726646259971647)
        self.clear()
        self.render()

    def __rotate_zp(self, event):
        self.rotate_model('z', 0.08726646259971647)
        self.clear()
        self.render()

    def __rotate_zn(self, event):
        self.rotate_model('z', -0.08726646259971647)
        self.clear()
        self.render()

    def bind_keys(self):
        self.__pressed = 0

        # initialize display
        self.screen.window.bind('<B1-Motion>', self.__drag)
        self.screen.window.bind('<ButtonRelease-1>', self.__resetDrag)

        self.screen.window.bind('<Up>', self.__zoomin)
        self.screen.window.bind('+', self.__zoomin)
        self.screen.window.bind('<Down>', self.__zoomout)
        self.screen.window.bind('-', self.__zoomout)
        self.screen.window.bind('w', self.__cameraup)
        self.screen.window.bind('s', self.__cameradown)
        self.screen.window.bind('a', self.__cameraleft)
        self.screen.window.bind('d', self.__cameraright)
        self.screen.window.bind('p', self.__deform)
        self.screen.window.bind('q', sys.exit)
        self.screen.window.bind('n', self.__next_tstep)
        self.screen.window.bind('N', self.__prev_tstep)
        self.screen.window.bind('1', self.__rotate_xp)
        self.screen.window.bind('!', self.__rotate_xn)
        self.screen.window.bind('2', self.__rotate_yp)
        self.screen.window.bind('@', self.__rotate_yn)
        self.screen.window.bind('3', self.__rotate_zp)
        self.screen.window.bind('#', self.__rotate_zn)
        self.screen.window.bind('<KeyPress>', self.__save_deform)
        self.screen.window.bind('<KeyRelease>', self.__restore_deform)

        # this is for editing the model
        self.screen.window.bind('<ButtonPress-3>', self.__select)
        self.screen.window.bind('<ButtonRelease-3>', self.__deselect)
        self.screen.window.bind('x', self.__selectx)
        self.screen.window.bind('y', self.__selecty)
        self.screen.window.bind('z', self.__selectz)
        self.screen.window.bind('<Left>', self.__movedown)
        self.screen.window.bind('<Right>', self.__moveup)

    def __init__(self, points, elements, displacement=None, width=1000, height=700, distance=6, scale=100, title='3D', background='white', num_steps=11, projection='perspective'):
        # object parameters
        self.distance = distance
        self.extents = self.get_model_extents(points)
        self.scale = self.initial_scale(width, height)
        self.scale_step = 0.025 * self.scale
        self.palette = PALETTE
        self.projection = projection

        # transformation matrix
        # self.Tr = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float) # x right, y down
        # self.rotate('x', np.pi/2)
        self.Tr = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float) # x right, y up
        # self.Tr = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float) # x right, z up
        # move model center to 0
        self.Tt =  -1 * self.extents[2].flatten().reshape((3,1))

        # initialize display
        self.status = '({0:' + str(len(str(num_steps))) + 'n}/{1:' + str(len(str(num_steps))) + 'n})'
        self.screen = graphics.screen.Screen(width, height, title, background, self.status.format(0, num_steps))
        self.__prev = []

        # this is for editing the model
        self.__selected = None
        self.__axis = []
        self.__moveaxis = None

        self.bind_keys()

        # store coordinates
        self.writePoints(points, displacement)
        self.flattened = []

        # store faces
        # self.elements = []
        # if lines is not None:
        #     self.writeLines(lines)
        # if triangles is not None:
        #     self.writeTriangles(triangles)
        # if quads is not None:
        #     self.writeQuads(quads)
        self.writeElements(elements)

        # displacement
        self.num_steps = num_steps
        self.tstep = 0
        self.dt = 1
        self.dscale = 1.
        self.prange = self.palette_range()

        self.deform = False
        self.__deform = self.deform

        # triad
        self.triad = []
        for vector in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]:
            self.triad.append(graphics.vertex.Vertex(vector))

    def clear(self):
        # clear display
        self.screen.clear()

    def rotate_view(self, axis, angle):
        if axis == 'x':
            T = np.array([[1,             0,              0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle),  np.cos(angle)]], dtype=float)
        elif axis == 'y':
            T = np.array([[np.cos(angle), 0, -np.sin(angle)],
                          [0,             1,              0],
                          [np.sin(angle), 0,  np.cos(angle)]], dtype=float)
        elif axis == 'z':
            T = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle),  np.cos(angle), 0],
                          [            0,              0, 1]], dtype=float)
        else:
          raise ValueError(f'invalid rotation axis {axis:s}')
        self.Tr = T @ self.Tr

    def rotate_model(self, axis, angle):
        if axis == 'x':
            T = np.array([[1,             0,              0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle),  np.cos(angle)]], dtype=float)
        elif axis == 'y':
            T = np.array([[np.cos(angle), 0, -np.sin(angle)],
                          [0,             1,              0],
                          [np.sin(angle), 0,  np.cos(angle)]], dtype=float)
        elif axis == 'z':
            T = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle),  np.cos(angle), 0],
                          [            0,              0, 1]], dtype=float)
        else:
          raise ValueError(f'invalid rotation axis {axis:s}')
        self.Tr = self.Tr @ T

    def render_triad(self):
        X = 0.9 * np.array([self.screen.width, self.screen.height], dtype=float).reshape(2,1)
        size = int(0.05 * min(self.screen.width, self.screen.height))

        if self.projection == 'ortho':
            distance = 0.
        else:
            distance = self.distance

        for vector, color in zip(self.triad, ['red', 'green', 'blue']):
            v = vector.flatten(size, distance, self.Tr, np.array([0, 0, 0], dtype=float).reshape(3,1))
            v = np.hstack((X, X + v))
            self.screen.createVector(v, color)

    def render(self):
        if self.deform:
            self.next_tstep()

        if self.projection == 'ortho':
            distance = 0.
        else:
            distance = self.distance

        self.dscale = self.displacement_scale()
        # calculate flattened coordinates (x, y)
        self.flattened = np.array([self.points[i].flatten(self.scale, distance, self.Tr, self.Tt, self.dscale) for i in range(self.points.shape[0])], dtype=float)

        # get coordinates to draw triangles and quads
        avgZ = np.zeros(self.elements.shape[0], dtype=float)
        color = np.zeros(self.elements.shape[0], dtype=float)
        for i in range(self.elements.shape[0]):
            avgZ[i] = np.mean([self.points[vertex].dist(self.Tr, self.Tt) for vertex in self.elements[i].v])
            color[i] = self.prange[0] + (np.mean([self.points[vertex].rms(self.dscale) for vertex in self.elements[i].v]) - self.prange[0]) / self.prange[1]

        # get element order from front to back
        Zorder = avgZ.argsort()

        # draw triangles and quads
        for i in range(Zorder.shape[0]):
            self.screen.createElement(self.flattened[self.elements[Zorder[i]].v].reshape(self.elements[Zorder[i]].v.shape[0],2), self.color(color[Zorder[i]]))

        self.render_triad()
        self.screen.label.config(text=self.status.format(self.tstep, self.num_steps))

