import tkinter as tk
import math

class Vec2:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

def sub(u: Vec2, v: Vec2) -> Vec2:
    return Vec2(u.x - v.x, u.y - v.y)

def v_length(vec: Vec2):
    return math.sqrt(vec.x ** 2 + vec.y ** 2)

class Node: 
    def __init__(self, x, y) -> None:
        self.coords = Vec2(x, y)

class Edge:
    def __init__(self, start: int, end: int, cost: int) -> None:
        self.start = start
        self.end = end
        self.cost = cost

class Graph:
    def __init__(self) -> None:
        self.sources: list[int] = []
        self.targets: list[int] = []
        self.costs: list[int] = []
        self.offsets: list[int] = []
        self.nodes: dict[str, Node] = {}

    def add_edge(self, edge: Edge) -> None:
        if not self.sources or (edge.start != self.sources[-1]):
            self.offset.append(len(self.sources))
        self.sources.append(edge.start)
        self.targets.append(edge.end)
        self.costs.append(edge.cost)

    # supply a flat list of nodes first
    def add_node(self, name: str, node: Node):
        self.nodes.append(node)

class DrawHandler:
    def __init__(self, graph: Graph) -> None:
        self.graph: Graph = graph
        self.window = tk.Tk()
        self.canvas_x: int = 800
        self.canvas_y: int = 800
        self.canvas = tk.Canvas(self.window, width=self.canvas_x, height=self.canvas_y)
        self.l: float = math.sqrt((self.canvas_x * self.canvas_y) / len(self.graph.sources))
        self.displacements: list[Vec2] = []
        for v in graph.offsets:
            self.displacements.append(Vec2(0, 0))

        self.layout()
    
    def layout(self):
        for it in range(30):
            self.fdl(graph, it)

    def cool(self, iteration):
        maxCool = 2
        coolProgress = iteration / 1200.0
        coolProgress = math.abs(math.pow(10, -coolProgress) - 0.1)
        return coolProgress * maxCool

    def fr(self, d):
        if d == 0:
            return 0
        return math.abs(-(self.l ** 2) / d)

    def fa(self, d):
        return math.abs((d ** 2) / self.l)

    def fdl(self):
        iteration = self.cool(iteration)
        for v, _ in enumerate(self.graph.offsets):
            self.displacements[v] = Vec2(0, 0)
            for u, _ in enumerate(self.graph.offsets):
                if u != v:
                    delta: Vec2 = sub(u, v)
                    deltaLength: float = v_length(delta)
                    self.displacements[v]

graph = Graph()
graph.add_edge(Edge(0, 1, 1))
graph.add_edge(Edge(1, 2, 2))
graph.add_edge(Edge(1, 3, 5))
graph.add_edge(Edge(2, 1, 3))
graph.add_edge(Edge(3, 0, 2))