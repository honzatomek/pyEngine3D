class Line:
    def __init__(self, vertices):
        #store point indexes
        (a, b, color) = vertices
        self.a = a
        self.b = b
        self.color = color

class Face3:
    def __init__(self, vertices):
        #store point indexes
        (a, b, c, color) = vertices
        self.a = a
        self.b = b
        self.c = c
        self.color = color

class Face4:
    def __init__(self, vertices):
        #store point indexes
        (a, b, c, d, color) = vertices
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.color = color
