import tkinter as tk
import math
from enum import Enum
import heapq
import numpy as np

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
    
    def __lt__(self, other):
        return self.idx < other.idx

    def __le__(self, other):
        return self.idx <= other.idx

class Edge:
    def __init__(self, start: str, end: str) -> None:
        self.start = start
        self.end = end

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
        self.sources.append(edge.start)
        self.targets.append(edge.end)
        # bidirectional
        self.sources.append(edge.end)
        self.targets.append(edge.start)
        # the coords of nodes are defined on a unit coordinate system (x and y from 0 to 1)
        # calculate the length of an edge via euclidean distance, scaled by 50 and converted to int 
        cost: int = math.ceil(VecM.sub(self.nodes[edge.end].coords, self.nodes[edge.start].coords).len() * self.scale_factor) 
        self.costs.append(cost)
        self.costs.append(cost)

    def add_node(self, name: str, node: Node):
        idx = len(self.nodes)
        node.append_idx(idx)
        self.nodes[name] = node

    def init_demo_graph(self):
        self.add_node("s", Node(0.3, 0.65))
        self.add_node("A", Node(0.1, 0.65))    #0
        self.add_node("B", Node(0.2, 0.75))    #1
        self.add_node("C", Node(0.2, 0.5))     #2
        self.add_node("D", Node(0.3, 0.9))     #3  
        self.add_node("E", Node(0.3, 0.8))     #4
        self.add_node("F", Node(0.4, 0.9))     #5
        self.add_node("G", Node(0.4, 0.75))    #6 
        self.add_node("H", Node(0.6, 0.8))     #7
        self.add_node("I", Node(0.5, 0.6))     #8
        self.add_node("J", Node(0.65, 0.65))   #9
        self.add_node("K", Node(0.35, 0.45))   #10
        self.add_node("L", Node(0.5, 0.4))     #11
        self.add_node("M", Node(0.25, 0.3))    #12 
        self.add_node("N", Node(0.6, 0.25))     #13
        self.add_node("O", Node(0.7, 0.5))     #14
        self.add_node("P", Node(0.4, 0.2))     #15
        self.add_node("t", Node(0.8, 0.2))
        self.add_node("Q", Node(0.9, 0.1))

        self.add_edge(Edge("s","B"))
        self.add_edge(Edge("s","E"))
        self.add_edge(Edge("s","I"))
        self.add_edge(Edge("s","K"))
        self.add_edge(Edge("s","C"))
        self.add_edge(Edge("A","B"))
        self.add_edge(Edge("B","E"))
        self.add_edge(Edge("B","D"))
        self.add_edge(Edge("B","C"))
        self.add_edge(Edge("C","A"))
        self.add_edge(Edge("C","M"))
        self.add_edge(Edge("D","F"))
        self.add_edge(Edge("E","D"))
        self.add_edge(Edge("E","G"))
        self.add_edge(Edge("G","F"))
        self.add_edge(Edge("G","H"))
        self.add_edge(Edge("G","I"))
        self.add_edge(Edge("G","s"))
        self.add_edge(Edge("H","J"))
        self.add_edge(Edge("I","O"))
        self.add_edge(Edge("I","L"))
        self.add_edge(Edge("J","I"))
        self.add_edge(Edge("L","K"))
        self.add_edge(Edge("L","N"))
        self.add_edge(Edge("M","P"))
        self.add_edge(Edge("M","L"))
        self.add_edge(Edge("M","K"))
        self.add_edge(Edge("N","O"))
        self.add_edge(Edge("N","t"))
        self.add_edge(Edge("O","t"))
        self.add_edge(Edge("P","N"))
        self.add_edge(Edge("t","Q"))

        self.postprocess()

    def postprocess(self):
        # sort sources array
        sorted_args = sorted(enumerate(self.sources), key=lambda x: self.nodes.get(x[1]).idx)
        sort_args = [i[0] for i in sorted_args]
        self.sources = [i[1] for i in sorted_args]
        self.targets = [self.targets[i] for i in sort_args]
        self.costs = [self.costs[i] for i in sort_args]
        self.add_offset()

    def add_offset(self):
        offsets = np.zeros(len(self.nodes), dtype=np.int32)
        edge_index = 0 
        for name, node in self.nodes.items():
            offsets[node.idx] = edge_index
            while edge_index < len(self.sources) and name == self.sources[edge_index]:
                edge_index += 1

        self.offsets = list(offsets)

    def compute_euclidean_potential(self, coords: Vec2):
        return math.ceil(VecM.sub(coords, self.nodes.get("t").coords).len() * self.scale_factor)
    
    def distance_based_potential(self):
        for _, node in self.nodes.items():
            potential: int = self.compute_euclidean_potential(node.coords)
            node.potential = potential
            start_offset = self.offsets[node.idx]
            if len(self.offsets) > node.idx + 1:
                end_offset = self.offsets[node.idx + 1]
            else:
                end_offset = len(self.sources) 

            for local_idx, cost in enumerate(self.costs[start_offset:end_offset]):
                edge_idx = start_offset + local_idx
                source: Node = self.nodes.get(self.sources[edge_idx])
                target: Node = self.nodes.get(self.targets[edge_idx])
                source_potential = self.compute_euclidean_potential(source.coords)
                target_potential = self.compute_euclidean_potential(target.coords)
                self.updated_costs[edge_idx] = cost + target_potential - source_potential


    def compute_potentials(self, euclidean: bool):
        self.updated_costs = self.costs.copy()
        if euclidean == True:
            self.distance_based_potential()
        else:
            self.landmark_potential()

    def dijkstra(self, s: Node):
        node_state = np.full(len(self.nodes), NodeState.UNREACHED) 
        dist = np.full(len(self.nodes), float('inf'))
        parent = np.full(len(self.nodes), None)
        queue = []
        heapq.heapify(queue)
        heapq.heappush(queue, (0, s))
        dist[s.idx] = 0
        while len(queue) > 0:
            node: Node = heapq.heappop(queue)[1]
            node_state[node.idx] = NodeState.SCANNED
            start_offset = self.offsets[node.idx]
            try:
                end_offset = self.offsets[node.idx + 1]
            except:
                end_offset = len(self.sources)
            edges = self.targets[start_offset:end_offset]
            for edge_idx, endpoint in enumerate(edges):
                target = self.nodes.get(endpoint)
                if node_state[target.idx] == NodeState.SCANNED:
                    continue
                start_offset = self.offsets[node.idx]
                node_state[target.idx] = NodeState.LABELED
                d = dist[node.idx] + self.costs[start_offset + edge_idx]
                if dist[target.idx] >= d:
                    dist[target.idx] = d
                    parent[target.idx] = node.idx
                    heapq.heappush(queue, (dist[target.idx], target))
        return dist

    def landmark_potential(self):
        landmark = self.nodes.get("Q")
        self.updated_costs = self.costs.copy()
        lm_dists = []
        for name, node in self.nodes.items():
            dist = self.dijkstra(node)
            lm_dists.append(dist[landmark.idx])
        
        for name, node in self.nodes.items():
            node.potential = int(lm_dists[node.idx] - lm_dists[self.nodes.get("t").idx])
            start_offset = self.offsets[node.idx]
            if len(self.offsets) > node.idx + 1:
                end_offset = self.offsets[node.idx + 1]
            else:
                end_offset = len(self.sources) 

            for local_idx, cost in enumerate(self.costs[start_offset:end_offset]):
                edge_idx = start_offset + local_idx
                source: Node = self.nodes.get(self.sources[edge_idx])
                target: Node = self.nodes.get(self.targets[edge_idx])
                source_potential = int(lm_dists[source.idx] - lm_dists[self.nodes.get("t").idx]) 
                target_potential = int(lm_dists[target.idx] - lm_dists[self.nodes.get("t").idx]) 
                self.updated_costs[edge_idx] = cost + target_potential - source_potential
    
class StateType(Enum):
    OUTER = 1
    INNER = 2

class AlgState:
    def __init__(self, node: Node) -> None:
        self.state_type = StateType.OUTER
        self.node = node
        self.edges: list[str] = []
        self.edge_idx:int = 0

    def reset(self, state_type: StateType, node: Node, edges: list[str], edge_idx: int):
        self.state_type = state_type
        self.node = node
        self.edges = edges
        self.edge_idx = edge_idx

    def increment(self):
        self.edge_idx += 1

class IterativeDijkstra:
    def __init__(self, graph: Graph, costs: list[int]) -> None:
        self.graph = graph
        self.costs = costs
        self.dist = np.full(len(self.graph.nodes), float("inf"))
        self.parent = np.full(len(self.graph.nodes), None)
        self.node_state = np.full(len(self.graph.nodes), NodeState.UNREACHED)
        start_node: Node = self.graph.nodes.get("s")
        self.dist[start_node.idx] = 0
        self.queue = []
        heapq.heapify(self.queue)
        heapq.heappush(self.queue, (0, start_node))
        self.state = AlgState(start_node)
        self.finished = False

    def outer_loop(self):
        if len(self.queue) > 0:
            node: Node = heapq.heappop(self.queue)[1]
            node_idx: int = node.idx
            self.node_state[node_idx] = NodeState.SCANNED
            start_offset = self.graph.offsets[node_idx]
            try:
                end_offset = self.graph.offsets[node_idx + 1]
            except:
                end_offset = len(self.graph.sources)
            edges = self.graph.targets[start_offset:end_offset]
            self.state.reset(StateType.INNER, node, edges, 0)
        

    def inner_loop(self):
        end_node = self.state.edges[self.state.edge_idx]
        target = self.graph.nodes.get(end_node)
        if self.node_state[target.idx] == NodeState.SCANNED:
            self.state.increment()
            self.single_step()
            return
        start_offset = self.graph.offsets[self.state.node.idx]
        self.node_state[target.idx] = NodeState.LABELED
        target_idx = target.idx
        d = self.dist[self.state.node.idx] + self.costs[start_offset + self.state.edge_idx]
        if self.dist[target_idx] >= d:
            self.dist[target_idx] = d
            self.parent[target_idx] = self.state.node.idx
            heapq.heappush(self.queue, (self.dist[target_idx], target))
        self.state.increment()

    def single_step(self):
        if self.finished:
            return
        if self.state.edge_idx >= len(self.state.edges):
            self.state.reset(StateType.OUTER, self.state.node, [], 0)
        if self.state.state_type == StateType.OUTER:
            self.outer_loop()
        else:
            self.inner_loop()
        if self.graph.nodes.get("t").state == NodeState.SCANNED:
            self.finished = True
    
# class Step:
#     def __init__(self, fwd, bwd) -> None:
#         self.forward = fwd
#         self.backward = bwd

class DrawHandler:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.canvas_x: int = 900
        self.canvas_y: int = 900
        self.node_size = 15
        self.canvas = tk.Canvas(self.window, width=self.canvas_x, height=self.canvas_y)
        self.canvas.pack()
        #self.init_sequence()
        self.window.bind("<Right>", self.fwd_event)
        b1 = tk.Button(self.window, text = "Dijkstra", command=self.start_dijkstra)
        b2 = tk.Button(self.window, text = "A* Distance", command=self.init_a_star_dist)
        b3  = tk.Button(self.window, text = "A* Landmark", command=self.init_a_star_lm)
        b1.pack(side=tk.LEFT)
        b2.pack(side=tk.LEFT)
        b3.pack(side=tk.LEFT)
        self.cost_elements = []
        self.updated_cost_elements = []
        self.potential_elements = []
        self.node_elements = []
        self.node_name_elements = []
        self.edge_elements = []
        self.dist_elements = []
        self.start_dijkstra()
        self.window.mainloop()

    def reset_graph(self):
        self.graph: Graph = Graph()
        self.graph.init_demo_graph()
    
    def init_a_star_dist(self):
        self.start_a_star(True)

    def init_a_star_lm(self):
        self.start_a_star(False)

    def start_a_star(self, euclidean):
        self.reset_graph()
        self.graph.compute_potentials(euclidean)
        self.it_dijkstra = IterativeDijkstra(self.graph, self.graph.updated_costs)
        self.redraw_graph()
        self.init_sequence(True)

    def start_dijkstra(self):
        self.reset_graph()
        self.it_dijkstra = IterativeDijkstra(self.graph, self.graph.costs)        
        self.redraw_graph()
        self.init_sequence(False)

    def redraw_graph(self):
        self.remove_cost_elements()
        self.remove_node_drawing()
        self.remove_potential_elements()
        self.remove_updated_cost_elements()
        self.remove_edge_elements()
        self.remove_dist_labels()
        self.draw_graph()

    def init_sequence(self, a_star: bool):
        self.sequence = []
        self.current_step = 0
        self.sequence.append(self.draw_costs)
        if a_star:
            self.sequence.append(self.draw_potentials)
            self.sequence.append(self.draw_updated_costs)

    def fwd_event(self, event):
        self.current_step = min(self.current_step, len(self.sequence) - 1)
        self.sequence[self.current_step]()
        if self.current_step >= len(self.sequence) - 1 and not self.it_dijkstra.finished:
            self.sequence.append(self.single_step_dijkstra)
        self.current_step = min(self.current_step + 1, len(self.sequence))
    
    def single_step_dijkstra(self):
        self.it_dijkstra.single_step()
        self.redraw_nodes()

    def redraw_nodes(self):
        self.remove_node_drawing()
        self.remove_dist_labels()
        self.draw_nodes()

    def remove_dist_labels(self):
        for dist_element in self.dist_elements:
            self.canvas.delete(dist_element)

    def remove_node_drawing(self):
        for node_element in self.node_elements:
            self.canvas.delete(node_element)
        for name_element in self.node_name_elements:
            self.canvas.delete(name_element)
        self.node_elements = []
        self.node_name_elements = []
        
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
    
    def remove_edge_elements(self):
        for edge_element in self.edge_elements:
            self.canvas.delete(edge_element)
        self.edge_elements = []

    def convert_to_image_coords(self, local: Vec2):
        return Vec2(local.x * self.canvas_x, self.canvas_y - local.y * self.canvas_y)

    def draw_nodes(self):
        for name, node in self.graph.nodes.items():
            coords = self.convert_to_image_coords(node.coords)
            f_color = ""
            if name == "s" or name == "t":
                f_color = "yellow"
            if self.it_dijkstra.node_state[node.idx] == NodeState.LABELED:
                f_color = "cyan"
            if self.it_dijkstra.state.node == node:
                f_color = "green"
            elif self.it_dijkstra.node_state[node.idx] == NodeState.SCANNED:
                f_color = "magenta"
            node_elem = self.canvas.create_oval(coords.x - self.node_size, coords.y - self.node_size, coords.x + self.node_size, coords.y + self.node_size, fill=f_color)
            self.node_elements.append(node_elem)
            dist_element = self.canvas.create_text(coords.x - 25, coords.y, text = str(self.it_dijkstra.dist[node.idx]), font=('Helvetica 10 bold'))
            self.dist_elements.append(dist_element)
            node_text = self.canvas.create_text(coords.x, coords.y, text=name)
            self.node_name_elements.append(node_text)

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
        line = self.canvas.create_line(coord_start.x + x_offset, coord_start.y + y_offset, coord_end.x - x_offset, coord_end.y - y_offset, arrow=tk.LAST)
        self.edge_elements.append(line)

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

dh = DrawHandler()