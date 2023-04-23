import tkinter as tk
import math

class Vec2:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def len(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

class VecM:
    @staticmethod
    def sub(u: Vec2, v: Vec2) -> Vec2:
        return Vec2(u.x - v.x, u.y - v.y)

    @staticmethod
    def dot(v1: Vec2, v2: Vec2):
        return v1.x * v2.x + v1.y * v2.y

    @staticmethod
    def det(v1: Vec2, v2: Vec2):
        return v1.x * v2.y - v1.y * v2.x

    @staticmethod
    def full_angle(v1: Vec2, v2: Vec2):
        return math.atan2(VecM.det(v1, v2), VecM.dot(v1, v2))

    @staticmethod
    def half_angle(v1: Vec2, v2: Vec2): 
        return math.acos(VecM.dot(v1, v2) / (v1.len() * v2.len()))

# node is currently just a wrapper for Vec2 with better naming
class Node: 
    def __init__(self, x, y) -> None:
        self.coords = Vec2(x, y)

class Edge:
    def __init__(self, start: str, end: str) -> None:
        self.start = start
        self.end = end
        #self.cost = cost

class Graph:
    def __init__(self) -> None:
        self.sources: list[str] = []
        self.targets: list[str] = []
        self.costs: list[int] = []
        self.offsets: list[int] = []
        self.nodes: dict[str, Node] = {}

    def add_edge(self, edge: Edge) -> None:
        if not self.sources or (edge.start != self.sources[-1]):
            self.offsets.append(len(self.sources))
        self.sources.append(edge.start)
        self.targets.append(edge.end)
        # the coords of nodes are defined on a unit coordinate system (x and y from 0 to 1)
        # calculate the length of an edge via euclidean distance, scaled by 10 and converted to int 
        cost: int = int(VecM.sub(self.nodes[edge.end].coords, self.nodes[edge.start].coords).len() * 10) 
        self.costs.append(cost)

    def add_node(self, name: str, node: Node):
        self.nodes[name] = node

class DrawHandler:
    def __init__(self, graph: Graph) -> None:
        self.graph: Graph = graph
        self.window = tk.Tk()
        self.canvas_x: int = 800
        self.canvas_y: int = 800
        self.node_size = 15
        self.canvas = tk.Canvas(self.window, width=self.canvas_x, height=self.canvas_y)
        self.canvas.pack()
        self.draw_graph()
        self.window.mainloop()
        # self.l: float = math.sqrt((self.canvas_x * self.canvas_y) / len(self.graph.sources))
        # self.displacements: list[Vec2] = []
        # for v in graph.offsets:
        #     self.displacements.append(Vec2(0, 0))

        # self.layout()
    
    def convert_to_image_coords(self, local: Vec2):
        return Vec2(local.x * self.canvas_x, local.y * self.canvas_y)

    def draw_graph(self):
        for name, node in self.graph.nodes.items():
            coords = self.convert_to_image_coords(node.coords)
            self.canvas.create_oval(coords.x - self.node_size, coords.y - self.node_size, coords.x + self.node_size, coords.y + self.node_size)
            self.canvas.create_text(coords.x, coords.y, text=name)
        
        for idx, start in enumerate(self.graph.sources):
            end = self.graph.targets[idx]
            coord_start = self.convert_to_image_coords(self.graph.nodes[start].coords)
            coord_end = self.convert_to_image_coords(self.graph.nodes[end].coords)
            # calculate the angle wrt to a vertical reference line so that the line starts outside of the node
            ref_line = Vec2(1, 0)
            line = VecM.sub(self.graph.nodes[end].coords, self.graph.nodes[start].coords) 
            angle = VecM.full_angle(ref_line, line)
            x_offset = math.cos(angle) * self.node_size
            y_offset = math.sin(angle) * self.node_size
            self.canvas.create_line(coord_start.x + x_offset, coord_start.y + y_offset, coord_end.x - x_offset, coord_end.y - y_offset, arrow=tk.LAST)
            
            # attach the cost as text, after 20% of the line (closer to start vertex)
            cost = self.graph.costs[idx]
            part_angle = VecM.half_angle(ref_line, line)
            orientation_offset_x = math.sin(part_angle) * 10
            orientation_offset_y = math.cos(part_angle) * 10
            line_offset = self.convert_to_image_coords(Vec2(line.x * 0.3, line.y * 0.3))
            offset = Vec2(line_offset.x + orientation_offset_x, line_offset.y + orientation_offset_y)
            self.canvas.create_text(coord_start.x + offset.x, coord_start.y + offset.y, text=cost)

    # fruchterman reingold graph layouting stub, not used 

    # def layout(self):
    #     for it in range(30):
    #         self.fdl(graph, it)

    # def cool(self, iteration):
    #     maxCool = 2
    #     coolProgress = iteration / 1200.0
    #     coolProgress = math.abs(math.pow(10, -coolProgress) - 0.1)
    #     return coolProgress * maxCool

    # def fr(self, d):
    #     if d == 0:
    #         return 0
    #     return math.abs(-(self.l ** 2) / d)

    # def fa(self, d):
    #     return math.abs((d ** 2) / self.l)

    # def fdl(self):
    #     iteration = self.cool(iteration)
    #     for v, _ in enumerate(self.graph.offsets):
    #         self.displacements[v] = Vec2(0, 0)
    #         for u, _ in enumerate(self.graph.offsets):
    #             if u != v:
    #                 delta: Vec2 = sub(u, v)
    #                 deltaLength: float = v_length(delta)
    #                 self.displacements[v]

graph = Graph()

graph.add_node("0", Node(0.1, 0.1))
graph.add_node("1", Node(0.65, 0.2))
graph.add_node("2", Node(0.22, 0.58))
graph.add_node("3", Node(0.86, 0.52))

# graph.add_node("0", Node(0.5, 0.6))
# graph.add_node("1", Node(0.5, 0.4))
# graph.add_edge(Edge("0", "1"))

graph.add_edge(Edge("0", "1"))
graph.add_edge(Edge("1", "2"))
graph.add_edge(Edge("1", "3"))
graph.add_edge(Edge("2", "1"))
graph.add_edge(Edge("3", "0"))

dh = DrawHandler(graph)