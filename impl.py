import tkinter as tk
import math
from enum import Enum
from queue import PriorityQueue

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

class NodeState(Enum):
    UNREACHED = 1
    LABELED = 2
    SCANNED = 3

class Node: 
    def __init__(self, x, y) -> None:
        self.coords: Vec2 = Vec2(x, y)
        self.potential: int = 0
        self.state: NodeState = NodeState.UNREACHED
    
    def append_idx(self, idx) -> None:
        self.idx = idx

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
        self.updated_costs: list[int] = []
        self.offsets: list[int] = []
        self.nodes: dict[str, Node] = {}
        self.scale_factor = 50

    def add_edge(self, edge: Edge) -> None:
        if not self.sources or (edge.start != self.sources[-1]):
            self.offsets[self.nodes.get(edge.start).idx] = len(self.sources)
        self.sources.append(edge.start)
        self.targets.append(edge.end)
        # the coords of nodes are defined on a unit coordinate system (x and y from 0 to 1)
        # calculate the length of an edge via euclidean distance, scaled by 50 and converted to int 
        cost: int = math.ceil(VecM.sub(self.nodes[edge.end].coords, self.nodes[edge.start].coords).len() * self.scale_factor) 
        self.costs.append(cost)

    def add_node(self, name: str, node: Node):
        idx = len(self.nodes)
        node.append_idx(idx)
        self.nodes[name] = node
        self.offsets.append(0)

    def compute_euclidean_potential(self, coords: Vec2):
        return math.ceil(VecM.sub(coords, self.nodes.get("t").coords).len() * self.scale_factor)
    
    def compute_potentials(self):
        self.updated_costs = self.costs.copy()
        for _, node in self.nodes.items():
            potential: int = self.compute_euclidean_potential(node.coords)
            node.potential = potential
            start_offset = self.offsets[node.idx]
            if len(self.offsets) > node.idx + 1:
                end_offset = self.offsets[node.idx + 1]
            else:
                end_offset = len(self.sources) 

            for edge_idx, cost in enumerate(self.costs[start_offset:end_offset]):
                source: Node = self.nodes.get(self.sources[edge_idx])
                target: Node = self.nodes.get(self.targets[edge_idx])
                source_potential = self.compute_euclidean_potential(source.coords)
                target_potential = self.compute_euclidean_potential(target.coords)
                self.updated_costs[edge_idx] = cost + target_potential - source_potential
    
    def a_star(self):
        openlist = PriorityQueue()
        closedlist = set()
        # initialize start element
        openlist.put(0, self.nodes.get("s"))
        while openlist:
            currentNode: Node = openlist.get()
            if currentNode == self.nodes.get("t"):
                return path
            closedlist.add(currentNode)
        self.expandNode(currentNode)

    def expandNode(self, currentNode: Node, openlist: PriorityQueue, closedlist: set):
        idx: int = currentNode.idx
        start_offset: int = self.offsets[idx]
        if self.offsets[idx + 1]:
            end_offset = self.offsets[idx + 1]
        else:
            end_offset = len(self.sources) 
        outgoing_edges = self.targets[start_offset:end_offset]
        for node_name in outgoing_edges:
            node = self.nodes.get(node_name)
            if closedlist[node]:
                continue
            node.state = NodeState.LABELED

class Step:
    def __init__(self, fwd, bwd) -> None:
        self.forward = fwd
        self.backward = bwd

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
        self.init_sequence()
        self.window.bind("<Right>", self.fwd_event)
        self.window.bind("<Left>", self.bwd_event)
        self.cost_elements = []
        self.updated_cost_elements = []
        self.potential_elements = []
        self.draw_graph()
        self.window.mainloop()

    def init_sequence(self):
        self.sequence: list[Step] = []
        self.current_step = 0
        cost_step: Step = Step(self.draw_costs, self.remove_cost_elements)
        potential_step: Step = Step(self.draw_potentials, self.remove_potential_elements)
        updated_cost_step: Step = Step(self.draw_updated_costs, self.remove_updated_cost_elements)
        self.sequence.append(cost_step)
        self.sequence.append(potential_step)
        self.sequence.append(updated_cost_step)

    def fwd_event(self, event):
        self.current_step = min(self.current_step, len(self.sequence) - 1)
        self.sequence[self.current_step].forward()
        self.current_step = min(self.current_step + 1, len(self.sequence))
    
    def bwd_event(self, event):
        self.current_step = max(0, self.current_step - 1)
        self.sequence[self.current_step].backward()

    def remove_cost_elements(self):
        for cost_element in self.cost_elements:
            self.canvas.delete(cost_element)
        self.cost_elements = []
    
    def remove_updated_cost_elements(self):
        for cost_element in self.updated_cost_elements:
            self.canvas.delete(cost_element)
        self.updated_cost_elements = []

    def remove_potential_elements(self):
        for potential_element in self.potential_elements:
            self.canvas.delete(potential_element)
        self.potential_elements = []

    def convert_to_image_coords(self, local: Vec2):
        return Vec2(local.x * self.canvas_x, self.canvas_y - local.y * self.canvas_y)

    def draw_nodes(self):
        for name, node in self.graph.nodes.items():
            coords = self.convert_to_image_coords(node.coords)
            f_color = ""
            if name == "s" or name == "t":
                f_color = "yellow"
            self.canvas.create_oval(coords.x - self.node_size, coords.y - self.node_size, coords.x + self.node_size, coords.y + self.node_size, fill=f_color)
            self.canvas.create_text(coords.x, coords.y, text=name)

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

    def draw_edge_cost(self, u: Vec2, v: Vec2, idx: int, updated: bool):
        coord_start = self.convert_to_image_coords(u)
        line = VecM.sub(v, u)
        angle = self.calc_angle(u, v, True)
        orientation_offset_x = math.sin(angle) * 10
        orientation_offset_y = math.cos(angle) * 10
        line_offset = Vec2(self.canvas_x * line.x * 0.3, -self.canvas_y * line.y * 0.3)
        if updated:
            color = "red"
            cost = self.graph.updated_costs[idx]
            offset = Vec2(line_offset.x - orientation_offset_x, line_offset.y - orientation_offset_y)
        else:
            color = "black"
            cost = self.graph.costs[idx]
            offset = Vec2(line_offset.x + orientation_offset_x, line_offset.y + orientation_offset_y)
        
        return self.canvas.create_text(coord_start.x + offset.x, coord_start.y + offset.y, text=cost, fill=color)


    def draw_potentials(self):
        # add potential drawing
        for _, node in self.graph.nodes.items():
            coords = self.convert_to_image_coords(node.coords)
            potential_elements = self.canvas.create_text(coords.x, coords.y + 10, fill="blue", text=node.potential)
            self.potential_elements.append(potential_elements)


    def draw_costs(self):
        for idx, start in enumerate(self.graph.sources):
            end = self.graph.targets[idx]
            u = self.graph.nodes[start].coords
            v = self.graph.nodes[end].coords

            text_element = self.draw_edge_cost(u, v, idx, False)
            self.cost_elements.append(text_element)

    def draw_updated_costs(self):
        for idx, start in enumerate(self.graph.sources):
            end = self.graph.targets[idx]
            u = self.graph.nodes[start].coords
            v = self.graph.nodes[end].coords

            text_element = self.draw_edge_cost(u, v, idx, True)
            self.updated_cost_elements.append(text_element)


    def draw_edges(self):
        for idx, start in enumerate(self.graph.sources):
            end = self.graph.targets[idx]
            u = self.graph.nodes[start].coords
            v = self.graph.nodes[end].coords
            self.draw_edge_line(u, v)        

    def draw_graph(self):
        self.draw_nodes()
        self.draw_edges()

def initialize_demo_graph():
    graph = Graph()
    graph.add_node("s", Node(0.3, 0.65))
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
    graph.add_node("t", Node(0.8, 0.2))

    graph.add_edge(Edge("s","B"))
    graph.add_edge(Edge("s","E"))
    graph.add_edge(Edge("s","I"))
    graph.add_edge(Edge("s","K"))
    graph.add_edge(Edge("s","C"))
    graph.add_edge(Edge("s","G"))
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

    return graph

graph = initialize_demo_graph()

dh = DrawHandler(graph)