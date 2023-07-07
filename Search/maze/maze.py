
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw


# sys.path.append("..")
from queue import LifoQueue, PriorityQueue, Queue


class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        
        

    def __lt__(self, other):  # 优先级队列元组比较可能用到（第一个数字相等）
        return 0


class Maze():

    def __init__(self, filename, method):

        self.method = method

        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)
        
        self.cell_size = 50
        self.cell_border = 2

        # Create a blank canvas
        self.img = Image.new(
            "RGBA",
            (self.width * self.cell_size, self.height * self.cell_size),
            "black"
        )
        self.draw = ImageDraw.Draw(self.img)

        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def solve(self):
        """Finds a solution to maze, if one exists."""

        # Keep track of number of states explored
        self.num_explored = 0

        # Initialize frontier to just the starting position
        start = Node(state=self.start, parent=None, action=None)

        if self.method == "DFS":
            # frontier = StackFrontier()
            frontier = LifoQueue()
            frontier.put(start)
        elif self.method == "BFS":
            # frontier = QueueFrontier()
            frontier = Queue()
            frontier.put(start)
        else:
            frontier = PriorityQueue()
            frontier.put((0, start))
            cost_so_far = dict()
            cost_so_far[start.state] = 0

        # Initialize an empty explored set,探测过而不是访问过
        self.explored = set()
        self.explored.add(start.state)

        # Keep looping until solution found
        while True:

            # If nothing left in frontier, then no path
            if frontier.empty():
                raise Exception("no solution")

            # Choose a node from the frontier
            if self.method == "DFS" or self.method == "BFS":
                node = frontier.get()
            else:
                node = frontier.get()[1]
            self.num_explored += 1
            self.draw_image(show_solution=False,show_explored=True)
            yield self.img

            # If node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                self.draw_image(show_solution=True,show_explored=True)
                yield self.img
                return
                

            # Add neighbors to frontier
            for action, state in self.neighbors(node.state):
                if self.method == "A" or self.method == "D":
                    new_cost = cost_so_far[node.state] + 1
                    # 新节点和旧节点的新路径
                    if state not in cost_so_far or new_cost < cost_so_far[state]:
                        child = Node(state=state, parent=node, action=action)
                        self.explored.add(state)  # 这里不必要，用于图片输出上色
                        cost_so_far[state] = new_cost
                        if self.method == "A":
                            priority = new_cost+self.heuristic(state)
                        else:
                            priority = new_cost

                        frontier.put((priority, child))
                else:
                    if state not in self.explored:
                        child = Node(state=state, parent=node, action=action)
                        self.explored.add(state)
                        if self.method == "DFS" or self.method == "BFS":
                            frontier.put(child)
                        else:  # 'H'
                            priority = self.heuristic(state)
                            frontier.put((priority, child))

    def heuristic(self, state):
        return abs(self.goal[0]-state[0])+abs(self.goal[1]-state[1])

    def draw_image(self,show_solution=False,show_explored=False):
        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                self.draw.rectangle(
                    ([(j * self.cell_size + self.cell_border, i * self.cell_size + self.cell_border),
                      ((j + 1) * self.cell_size - self.cell_border, (i + 1) * self.cell_size - self.cell_border)]),
                    fill=fill
                )
        # self.img.save(filename)
        
        
    
        
        


# if len(sys.argv) != 3:
#     sys.exit("Usage: python maze.py maze.txt BFS/DFS/D/H/A")

# m = Maze(sys.argv[1], sys.argv[2])
m=Maze("C:\\Users\\nmnmnmnm\\Desktop\\AI-PYTHON\\Search\\maze\\maze.txt","A")
# print("Maze:")
# m.print()
# print("Solving...")
# m.solve()
# print("States Explored:", m.num_explored)
# print("Solution:")
# m.print()
# m.draw_image("maze.png", show_explored=True)

fig, ax = plt.subplots()

def update(img):
    ax.clear()
    ax.imshow(img)
ani = animation.FuncAnimation(fig,update, m.solve ,interval=100,repeat=False)
plt.show()#放在后面gif会出问题
ani.save("demo1.gif")
