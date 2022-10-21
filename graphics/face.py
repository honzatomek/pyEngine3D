import numpy as np

class Line:
    def __init__(self, vertices):
        #store point indexes
        self.v = np.array(vertices[:-1], dtype='int32')
        self.color = vertices[-1]

class Face3:
    def __init__(self, vertices):
        #store point indexes
        self.v = np.array(vertices[:-1], dtype='int32')
        self.color = vertices[-1]

class Face4:
    def __init__(self, vertices):
        #store point indexes
        self.v = np.array(vertices[:-1], dtype='int32')
        self.color = vertices[-1]

class Element:
    def __init__(self, vertices):
        #store point indexes
        self.v = np.array(vertices, dtype='int32')

    def __getitem__(self, index):
        return self.v[index]


