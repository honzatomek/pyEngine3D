import graphics.screen
import graphics.face
import graphics.vertex
import copy
import sys

import numpy as np

class Engine3D:
    def __resetDrag(self, event):
        self.__prev = []
        self.__restore_deform(event)

    def __drag(self, event):
        if self.__prev:
            self.rotate('y', (event.x - self.__prev[0]) / 20)
            self.rotate('x', (event.y - self.__prev[1]) / 20)
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
            self.__selected = self.flattened.index(found[0])

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
        self.scale += 2.5
        self.clear()
        deform = self.deform
        self.deform = False
        self.render()
        self.deform = deform

    def __zoomout(self, event):
        self.scale -= 2.5
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

    def __extents(self, points):
        points = np.array(points)
        minX, maxX = np.min(points[:,0]), np.max(points[:,0])
        minY, maxY = np.min(points[:,1]), np.max(points[:,1])
        minZ, maxZ = np.min(points[:,2]), np.max(points[:,2])
        return ((minX, minY, minZ), (maxX, maxY, maxZ)), ((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2)

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

    def dscale(self):
        scale = np.sin((self.tstep / (self.num_steps + 1)) * 2 * np.pi)
        return scale

    def palette_range(self):
        dmin = self.point[0].square(self.dscale)
        dmax = dmin
        for point in self.points:
            dmin = dmin if point.square(self.dscale) < dmin else dmin
            dmax = dmax if point.square(self.dscale) > dmax else dmax
        return (dmin, dmax)

    def palette(self, fraction):
        range = len(self.p)
        i = int((range - 1) * fraction)
        col1 = self.p[i]
        col2 = self.p[i+1]
        f = (fraction - i * range) * range

        return (int((col2[0] - col1[0]) * f), int((col2[1] - col1[1]) * f), int((col2[2] - col1[2]) * f))

    def writePoints(self, points, displacement):
        self.points = []
        if displacement is not None:
            for point, defor in zip(points, displacement):
                self.points.append(graphics.vertex.Vertex(point, displacement=defor))
        else:
            for point in points:
                self.points.append(graphics.vertex.Vertex(point))

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

    def __init__(self, points, lines=None, triangles=None, quads=None, displacement=None, width=1000, height=700, distance=6, scale=100, title='3D', background='white', num_steps=11, projection='perspective'):
        # object parameters
        self.distance = distance
        self.extents, self.offset = self.__extents(points)
        self.scale = 0.6 * min(height, width) / max(self.extents[1][0]-self.extents[0][0], self.extents[1][1]-self.extents[0][1], self.extents[1][2]-self.extents[0][2])
        self.p = np.array([[0., 0., 255.], [0., 255., 0.], [255., 255., 0.], [255., 0., 0.], [255., 0., 255.]], dtype=float)
        self.projection = projection

        # transformation matrix
        # self.Tr = np.eye(3, dtype=float)
        # self.rotate('x', np.pi)
        self.Tr = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
        self.Tt = np.zeros((3, 1), dtype=float)

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
        self.elements = []
        if lines is not None:
            self.writeLines(lines)
        if triangles is not None:
            self.writeTriangles(triangles)
        if quads is not None:
            self.writeQuads(quads)

        # displacement
        self.num_steps = num_steps
        self.tstep = 0
        self.dt = 1
        # self.dscale = 1.

        self.deform = False
        self.__deform = self.deform

        # triad
        self.triad = []
        for vector in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]:
            self.triad.append(graphics.vertex.Vertex(vector))

    def clear(self):
        # clear display
        self.screen.clear()

    def rotate(self, axis, angle):
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

    def render_triad(self):
        x = 0.9 * self.screen.width
        y = 0.9 * self.screen.height
        s = 0.05 * min(self.screen.width, self.screen.height)

        if self.projection == 'ortho':
            distance = 0.
        else:
            distance = self.distance

        for vector, color in zip(self.triad, ['red', 'green', 'blue']):
            v = vector.flatten(s, distance, self.Tr, self.Tt)
            self.screen.createVector([[x, y], [x + v[0], y + v[1]]], color)

    def render(self):
        if self.deform:
            self.next_tstep()

        if self.projection == 'ortho':
            distance = 0.
        else:
            distance = self.distance

        # calculate flattened coordinates (x, y)
        self.flattened = []
        print(self.Tr)
        for point in self.points:
            self.flattened.append(point.flatten(self.scale, distance, self.Tr, self.Tt, self.dscale()))

        # get coordinates to draw triangles and quads
        elements = []
        for element in self.elements:
            if type(element) is graphics.face.Line:
                avgZ = -(self.points[element.a].d(self.Tr, self.Tt) + self.points[element.b].d(self.Tr, self.Tt)) / 2
                elements.append((self.flattened[element.a], self.flattened[element.b], element.color, avgZ))
            elif type(element) is graphics.face.Face3:
                avgZ = -(self.points[element.a].d(self.Tr, self.Tt) + self.points[element.b].d(self.Tr, self.Tt) + self.points[element.c].d(self.Tr, self.Tt)) / 3
                elements.append((self.flattened[element.a], self.flattened[element.b], self.flattened[element.c], element.color, avgZ))
            elif type(element) is graphics.face.Face4:
                avgZ = -(self.points[element.a].d(self.Tr, self.Tt) + self.points[element.b].d(self.Tr, self.Tt) + self.points[element.c].d(self.Tr, self.Tt) + self.points[element.d].d(self.Tr, self.Tt)) / 4
                elements.append((self.flattened[element.a], self.flattened[element.b], self.flattened[element.c], self.flattened[element.d], element.color, avgZ))

        # sort elements from furthest back to closest
        elements = sorted(elements,key=lambda x: x[-1])

        # draw triangles and quads
        for element in elements:
            if len(element) == 4:
                self.screen.createBeam(element[0:2], element[2])
            elif len(element) == 5:
                self.screen.createTriangle(element[0:3], element[3])
            elif len(element) == 6:
                self.screen.createQuad(element[0:4], element[4])

        self.render_triad()
        self.screen.label.config(text=self.status.format(self.tstep, self.num_steps))

