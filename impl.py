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
        self.coords: Vec2 = Vec2(x, y)
        self.potential: int = 0

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
        self.scale_factor = 50

    def add_edge(self, edge: Edge) -> None:
        if not self.sources or (edge.start != self.sources[-1]):
            self.offsets.append(len(self.sources))
        self.sources.append(edge.start)
        self.targets.append(edge.end)
        # the coords of nodes are defined on a unit coordinate system (x and y from 0 to 1)
        # calculate the length of an edge via euclidean distance, scaled by 50 and converted to int 
        cost: int = int(VecM.sub(self.nodes[edge.end].coords, self.nodes[edge.start].coords).len() * self.scale_factor) 
        self.costs.append(cost)

    def add_node(self, name: str, node: Node):
        self.nodes[name] = node
    
    def compute_potentials(self):
        for _, node in self.nodes.items():
            potential: int = int(VecM.sub(node.coords, self.nodes.get("t").coords).len() * self.scale_factor)
            node.potential = potential

class DrawHandler:
    def __init__(self, graph: Graph) -> None:
        self.graph: Graph = graph
        self.graph.compute_potentials()
        self.window = tk.Tk()
        self.canvas_x: int = 800
        self.canvas_y: int = 800
        self.node_size = 15
        self.canvas = tk.Canvas(self.window, width=self.canvas_x, height=self.canvas_y)
        self.canvas.pack()
        self.draw_graph()
        self.window.mainloop()
    
    def convert_to_image_coords(self, local: Vec2):
        return Vec2(local.x * self.canvas_x, self.canvas_y - local.y * self.canvas_y)


    def draw_nodes(self):
        for name, node in self.graph.nodes.items():
            coords = self.convert_to_image_coords(node.coords)
            f_color = ""
            if name == "s":
                f_color = "red"
            elif name == "t":
                f_color = "green"
            self.canvas.create_oval(coords.x - self.node_size, coords.y - self.node_size, coords.x + self.node_size, coords.y + self.node_size, fill=f_color)
            self.canvas.create_text(coords.x, coords.y, text=name)
            # add potential drawing
            self.canvas.create_text(coords.x, coords.y + 10, fill="blue", text=node.potential)

    def calc_angle(self, u: Vec2, v: Vec2, half: bool):
        ref_line = Vec2(1, 0)
        line = VecM.sub(v, u)

        if half: 
            return VecM.half_angle(ref_line, line)
        else: 
            return VecM.full_angle(ref_line, line)

    def draw_edge_line(self, u: Vec2, v: Vec2):
        coord_start = self.convert_to_image_coords(u)
        coord_end = self.convert_to_image_coords(v)
        # calculate the angle wrt to a vertical reference line so that the line starts outside of the node
        angle = self.calc_angle(u, v, False) 
        x_offset = math.cos(angle) * self.node_size
        y_offset = -math.sin(angle) * self.node_size
        self.canvas.create_line(coord_start.x + x_offset, coord_start.y + y_offset, coord_end.x - x_offset, coord_end.y - y_offset, arrow=tk.LAST)

    def draw_edge_cost(self, u: Vec2, v: Vec2, idx: int):
        cost = self.graph.costs[idx]
        coord_start = self.convert_to_image_coords(u)
        line = VecM.sub(v, u)
        angle = self.calc_angle(u, v, True)  
        orientation_offset_x = math.sin(angle) * 10
        orientation_offset_y = math.cos(angle) * 10
        line_offset = Vec2(self.canvas_x * line.x * 0.3, -self.canvas_y * line.y * 0.3)
        offset = Vec2(line_offset.x + orientation_offset_x, line_offset.y + orientation_offset_y)
        self.canvas.create_text(coord_start.x + offset.x, coord_start.y + offset.y, text=cost)

    def draw_edges(self):
        for idx, start in enumerate(self.graph.sources):
            end = self.graph.targets[idx]
            u = self.graph.nodes[start].coords
            v = self.graph.nodes[end].coords
            self.draw_edge_line(u, v)
            
            # attach the cost as text, after 20% of the line (closer to start vertex)
            self.draw_edge_cost(u, v, idx)

    def draw_graph(self):
        self.draw_nodes()
        self.draw_edges()


def initialize_demo_graph():
    graph = Graph()
    graph.add_node("A", Node(0.1, 0.65))    #0
    graph.add_node("B", Node(0.2, 0.75))    #1
    graph.add_node("C", Node(0.2, 0.5))     #2
    graph.add_node("D", Node(0.3, 0.9))     #3  
    graph.add_node("E", Node(0.3, 0.8))     #4
    graph.add_node("F", Node(0.4, 0.9))     #5
    graph.add_node("G", Node(0.4, 0.75))    #6 
    graph.add_node("H", Node(0.6, 0.8))     #7
    graph.add_node("I", Node(0.5, 0.6))     #8
    graph.add_node("J", Node(0.65, 0.65))   #9
    graph.add_node("K", Node(0.35, 0.45))   #10
    graph.add_node("L", Node(0.5, 0.4))     #11
    graph.add_node("M", Node(0.25, 0.3))    #12 
    graph.add_node("N", Node(0.6, 0.3))     #13
    graph.add_node("O", Node(0.7, 0.5))     #14
    graph.add_node("P", Node(0.4, 0.2))     #15
    graph.add_node("s", Node(0.3, 0.65))
    graph.add_node("t", Node(0.8, 0.2))

    graph.add_edge(Edge("A","B"))
    graph.add_edge(Edge("B","E"))
    graph.add_edge(Edge("B","D"))
    graph.add_edge(Edge("B","C"))
    graph.add_edge(Edge("C","A"))
    graph.add_edge(Edge("C","M"))
    graph.add_edge(Edge("D","F"))
    graph.add_edge(Edge("E","D"))
    graph.add_edge(Edge("E","G"))
    graph.add_edge(Edge("G","F"))
    graph.add_edge(Edge("G","H"))
    graph.add_edge(Edge("G","I"))
    graph.add_edge(Edge("H","J"))
    graph.add_edge(Edge("I","O"))
    graph.add_edge(Edge("I","L"))
    graph.add_edge(Edge("J","I"))
    graph.add_edge(Edge("L","K"))
    graph.add_edge(Edge("L","N"))
    graph.add_edge(Edge("M","P"))
    graph.add_edge(Edge("M","L"))
    graph.add_edge(Edge("M","K"))
    graph.add_edge(Edge("N","O"))
    graph.add_edge(Edge("N","t"))
    graph.add_edge(Edge("O","t"))
    graph.add_edge(Edge("P","N"))
    graph.add_edge(Edge("s","B"))
    graph.add_edge(Edge("s","E"))
    graph.add_edge(Edge("s","I"))
    graph.add_edge(Edge("s","K"))
    graph.add_edge(Edge("s","C"))
    graph.add_edge(Edge("s","G"))

    return graph

graph = initialize_demo_graph()

dh = DrawHandler(graph)