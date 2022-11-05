import tkinter
import numpy as np

class Screen:
    def __init__(self, width, height, title, background, status):
        # store size
        self.width = width
        self.height = height

        #calculate center of screen
        self.zeros = np.array([width/2, height/2], dtype='int32')

        # initialize tkinter window for displaying graphics
        self.window = tkinter.Tk()
        self.window.title(title)
        self.image = tkinter.Canvas(self.window, width=width, height=height, bg=background)
        self.image.pack()

        # initialise status line
        self.label = tkinter.Label(self.image, text=status)
        self.label.place(relx=0.5, rely=0.95, anchor='center')

    def createElement(self, points: np.ndarray, color):
        """
        In:
            points - [[x1, y1], .. , [xn, yn]] (flattened coordinates)
            color  - color name
        """
        # create coordinates starting in center of screen
        coors = (np.array(points, dtype='int32') + self.zeros).flatten().tolist()
        print(coors)
        if type(color) in (np.ndarray, tuple, list):
            color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        # draw element on screen
        if len(points) > 2:
            self.image.create_polygon(*coors, fill=color, outline="black")
        # draw beam
        elif len(points) == 2:
            self.image.create_line(*coors, fill=color, arrow=tkinter.NONE)
        # draw mass element
        else:
            r = 10
            # coors = np.hstack((coors + 10, coors - 10)).flatten()
            self.image.create_oval(coors[0]-r, coors[1]-r, coors[0]+r, coors[1]+r, fill=color, outline="black")

    def createSupport(self, point: np.ndarray, size: float, color='black'):
        # create coordinates starting in center of screen
        coors = np.hstack((point - size/2, point + size/2)).flatten()
        # draw circle on screen
        self.image.create_oval(*coors, fill=color, outline="black")

    def createVertex(self, point: np.ndarray, size: float, color='blue'):
        # create coordinates starting in center of screen
        coors = np.hstack((point - size/2, point + size/2)).flatten()
        # draw circle on screen
        self.image.create_oval(*coors, fill=color, outline="black")

    def createTriangle(self, points, color):
        """
        In:
            points - [[x1, y1], [x2, y2], [x3, y3]] (flattened coordinates)
            color  - color name
        """
        a, b, c = points[0], points[1], points[2]
        # create coordinates starting in center of screen
        coords = [a[0] + self.zeros[0], a[1] + self.zeros[1], b[0] + self.zeros[0], b[1] + self.zeros[1], c[0] + self.zeros[0], c[1] + self.zeros[1]]
        # draw triangle on screen
        self.image.create_polygon(coords, fill=color, outline="black")

    def createQuad(self, points, color):
        a, b, c, d = points[0], points[1], points[2], points[3]
        # create coordinates starting in center of screen
        coords = [a[0] + self.zeros[0], a[1] + self.zeros[1], b[0] + self.zeros[0], b[1] + self.zeros[1], c[0] + self.zeros[0], c[1] + self.zeros[1], d[0] + self.zeros[0], d[1] + self.zeros[1]]
        # draw triangle on screen
        self.image.create_polygon(coords, fill=color, outline="black")

    def createBeam(self, points, color):
        a, b = points[0], points[1]
        self.image.create_line(a[0] + self.zeros[0], a[1] + self.zeros[1], b[0] + self.zeros[0], b[1] + self.zeros[1], fill=color, arrow=tkinter.NONE)

    def createLine(self, points, color):
        a, b = points[0], points[1]
        return self.image.create_line(a[0], a[1], b[0], b[1], fill=color, arrow=tkinter.BOTH)

    def createVector(self, points, color):
        # a, b = points[0], points[1]
        # self.image.create_line(a[0], a[1], b[0], b[1], fill=color, arrow=tkinter.LAST)
        self.image.create_line(*points.T.flatten(), fill=color, arrow=tkinter.LAST)

    def clear(self):
        #clear display
        self.image.delete('all')

    def delete(self, item):
        self.image.delete(item)
        return None

    def after(self, time, function):
        #call tk.Tk's after() method
        self.window.after(time, function)
