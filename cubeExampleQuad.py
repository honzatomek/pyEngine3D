### Cube ###

import graphics.engine

points = [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1],[1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1]]
triangles = [] # [[0,1,2],[0,2,3],[2,3,7],[2,7,6],[1,2,5],[2,5,6],[0,1,4],[1,4,5],[4,5,6],[4,6,7],[3,7,4],[4,3,0]]
quads = [[0,1,2,3],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]]

test = graphics.engine.Engine3D(points, triangles, quads, title='Cube')

def animation():
    test.clear()
    test.rotate('y', 0.01)
    test.rotate('x', 0.01)
    test.render()
    test.screen.after(1, animation)

animation()
test.screen.window.mainloop()

############
