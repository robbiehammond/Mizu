# potentially good reference: https://experiments.withgoogle.com/fluid-particles

# Note: If we want our simulation to be really accurate, we should not do it in real time (i.e. precompute the particle positions and then render them seamlessly)
# Basically, run the program, give it time to generate particle positions at each time value, then render these results 
# Doing things in real time is possible of course, but we'll be limited when it comes to the number of particles we can simulate since it will have to generate positions on the fly
# Since we have an HPC, we prolly could use an insane amount of particles, especially if we precompute the positions

# This code below moves a single particle through a 2D field of force vectors
# Particle is denoted as 1 in grid
class VecField2D:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.grid = []
        for i in range(w):
            self.grid.append([])
            for j in range(h):
                self.grid[i].append(vec2D(0, 0))


    def __str__(self):
        str = ""
        for i in range(self.w):
            for j in range(self.h):
                str += self.grid[i][j].__str__() + " "
            if i != self.w - 1:
                str += "\n"
        return str


    def setVec(self, row, col, vec):
        self.grid[row][col] = vec


    def modifyVec(self, row, col, vec):
        self.grid[row][col] += vec


    def getVec(self, row, col):
        return self.grid[row][col]


class vec2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    

    def __add__(self, other):
        return vec2D(self.x + other.x, self.y + other.y)


    def __sub__(self, other):
        return vec2D(self.x - other.x, self.y - other.y)


    def dot(self, other):
        return self.x * other.x + self.y * other.y


    def mag(self):
        return (self.x**2 + self.y**2)**0.5


    def __str__(self):
        return f"({self.x}, {self.y})"


def prettyPrint(grid):
    for row in grid:
        for e in row:
            print(e, end=" ")
        print()
w = 5
h = 5
flowField = VecField2D(w, h)
flowField.setVec(0, 0, vec2D(1, 1)) # There's a gradient at the top left corner, pushing a particle to the bottom right.
flowField.setVec(1, 1, vec2D(1, 1))

grid = []
for i in range(h):
    grid.append([0] * w)

grid[0][0] = 1

print("Step 0")
prettyPrint(grid)

steps = 5
# -1 denotes a particle was here, but has left (next step, this area will be 0)
# -2 denotes a particle has arrived here (next step, this area will have the particle)
for t in range(steps):
    for j in range(h):
        for k in range(w):
            if grid[j][k] == 1:
                vec = flowField.getVec(j, k)
                if j + vec.y < h and k + vec.x < w:
                    grid[j][k] = -1
                    grid[j + vec.y][k + vec.x] = -2

    for j in range(h):
        for k in range(w):
            if grid[j][k] == -1:
                grid[j][k] = 0
            elif grid[j][k] == -2:
                grid[j][k] = 1

    print("Step " + str(t + 1))
    prettyPrint(grid)

