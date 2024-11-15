#!/usr/bin/env python3

import abc
import math
from heapq import heapify, heappop, heappush
from typing import Dict, List, Tuple


class Node:
    def __init__(self, id: str, x: float = 0, y: float = 0) -> None:
        self.id = id.lower()
        self.x = x
        self.y = y

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"<Node id:{self.id} x: {self.x} y: {self.y}>"

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def __hash__(self) -> int:
        return hash(self.id)


class Graph:
    nodes: Dict[Node, Node]
    edges: Dict[Node, Dict[Node, float]]

    def __init__(
        self,
        nodes: List[Node] = [],
        edges: List[Tuple[Node, Node, float]] = [],
        undirected: bool = True,
    ) -> None:
        self.nodes = {}
        self.edges = {}
        self.undirected = undirected

        [self.add_node(node) for node in nodes]
        [self.add_edge(edge) for edge in edges]

    def add_node(self, node: Node) -> None:
        self.nodes[node] = node

    def add_edge(self, edge: Tuple[Node, Node, float]) -> None:
        source, dest, weight = edge

        self.add_node(source)
        self.add_node(dest)

        if source not in self.edges:
            self.edges[source] = {}
            self.edges[source][source] = 0.0

        if dest not in self.edges:
            self.edges[dest] = {}
            self.edges[dest][dest] = 0.0

        self.edges[source][dest] = weight

        if self.undirected:
            self.edges[dest][source] = weight


class PathFinderInterface(abc.ABC):
    @abc.abstractmethod
    def shortest_path(self, origin: Node, destination: Node) -> List[Node]:
        pass


class Dijsktra(PathFinderInterface):
    def __init__(self, graph: Graph):
        self.graph = graph

    def shortest_path(self, origin: Node, destination: Node) -> List[Node]:
        distances = {node: float("inf") for node in self.graph.nodes}
        predecessors: dict[Node, Node | None] = {
            node: None for node in self.graph.nodes
        }
        distances[origin] = 0.0

        pq = [(0.0, origin)]
        heapify(pq)

        visited = set()

        while pq:
            current_distance, current_node = heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor, weight in self.graph.edges[current_node].items():
                tentative_distance = current_distance + weight

                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    predecessors[neighbor] = current_node
                    heappush(pq, (tentative_distance, neighbor))

        path = []
        current_node = destination

        while current_node:
            path.append(current_node)
            current_node = predecessors[current_node]

        return path[::-1]


class GBFS(PathFinderInterface):
    def __init__(
        self,
        graph: Graph,
        heuristics: dict[Node, float] | None = None,
        include_origin: bool = False,
        include_destination: bool = False,
    ):
        self.graph = graph
        if heuristics is None:
            heuristics = {}

        self.heuristics = heuristics
        self.include_origin = include_origin
        self.include_destination = include_destination

    def shortest_path(self, origin: Node, destination: Node) -> List[Node]:
        assert origin in self.graph.nodes
        assert destination in self.graph.nodes

        # Just to make sure that the dude have the right coordinates from the calculate heuristics
        origin = self.graph.nodes[origin]
        destination = self.graph.nodes[destination]
        predecessors: dict[Node, Node | None] = {
            node: None for node in self.graph.nodes
        }

        if not self.heuristics:
            self.__calculate_heuristics(destination)

        pq = [(self.heuristics[origin], origin)]
        heapify(pq)

        visited = set()

        while pq:
            current_node = heappop(pq)[1]

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == destination:
                break

            for neighbor in self.graph.edges[current_node]:
                tentative_distance = self.heuristics[neighbor]
                if neighbor not in visited:
                    heappush(pq, (tentative_distance, neighbor))
                    predecessors[neighbor] = current_node

        path: List[Node] = []
        current_node = predecessors[destination]

        while current_node and current_node != origin:
            path.append(current_node)
            current_node = predecessors[current_node]

        path.reverse()

        if self.include_origin:
            path.insert(0, origin)

        if self.include_destination:
            path.append(destination)

        return path

    def __calculate_heuristics(self, destination: Node) -> None:
        for node in self.graph.nodes:
            euclidean_distance = math.sqrt(
                (node.x - destination.x) ** 2 + (node.y - destination.y) ** 2
            )
            self.heuristics[node] = euclidean_distance


class AStar(PathFinderInterface):
    def __init__(
        self,
        graph: Graph,
        heuristics: dict[Node, float] | None = None,
        include_origin: bool = False,
        include_destination: bool = False,
    ):
        self.graph = graph
        if heuristics is None:
            heuristics = {}

        self.heuristics = heuristics
        self.include_origin = include_origin
        self.include_destination = include_destination

    def shortest_path(self, origin: Node, destination: Node) -> List[Node]:
        assert origin in self.graph.nodes
        assert destination in self.graph.nodes

        # Just to make sure that the dude have the right coordinates from the calculate heuristics
        origin = self.graph.nodes[origin]
        destination = self.graph.nodes[destination]
        predecessors: dict[Node, Node | None] = {
            node: None for node in self.graph.nodes
        }

        if not self.heuristics:
            self.__calculate_heuristics(destination)

        pq = [(0 + self.heuristics[origin], origin)]
        heapify(pq)

        visited = set()

        while pq:
            current_distance, current_node = heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == destination:
                break

            for neighbor, weight in self.graph.edges[current_node].items():
                tentative_distance = (
                    current_distance + weight + self.heuristics[neighbor]
                )
                if neighbor in visited:
                    continue

                heappush(pq, (tentative_distance, neighbor))
                predecessors[neighbor] = current_node

        path: List[Node] = []
        current_node = predecessors[destination]

        while current_node and current_node != origin:
            path.append(current_node)
            current_node = predecessors[current_node]

        path.reverse()

        if self.include_origin:
            path.insert(0, origin)

        if self.include_destination:
            path.append(destination)

        return path

    def __calculate_heuristics(self, destination: Node) -> None:
        for node in self.graph.nodes:
            euclidean_distance = math.sqrt(
                (node.x - destination.x) ** 2 + (node.y - destination.y) ** 2
            )
            self.heuristics[node] = euclidean_distance


def greedy_beverage_machine(user_cash_input: float, beverage_cost: float) -> float:
    assert user_cash_input > 0
    assert beverage_cost > 0
    assert user_cash_input >= beverage_cost

    available_coins = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    change = user_cash_input - beverage_cost

    for coin in available_coins:
        if coin <= change:
            return coin

    # User cash input must be equal to beverage cost
    assert user_cash_input == beverage_cost

    return 0


def first_question():
    print("First question")

    g = Graph()
    node_a = Node("A")
    node_b = Node("B")
    node_c = Node("C")
    node_d = Node("D")
    node_e = Node("E")
    node_f = Node("F")
    node_g = Node("G")

    g.add_edge((node_a, node_b, 3.0))
    g.add_edge((node_a, node_c, 3.0))

    g.add_edge((node_b, node_d, 3.5))
    g.add_edge((node_b, node_e, 2.8))

    g.add_edge((node_c, node_e, 2.8))
    g.add_edge((node_c, node_f, 3.5))

    g.add_edge((node_d, node_e, 3.1))
    g.add_edge((node_d, node_g, 10.0))

    g.add_edge((node_e, node_g, 7.0))
    g.add_edge((node_f, node_g, 2.5))

    dj = Dijsktra(graph=g)

    path = dj.shortest_path(node_b, node_f)
    print(f"shortest path from B to F is {path}")


def second_question():
    print("Second question")
    user_cash_input = 1
    beverage_cost = 0.7
    return_value = greedy_beverage_machine(user_cash_input, beverage_cost)

    print(
        f"for {user_cash_input} dolar of input and a cost of {beverage_cost}, the return from the machine is {return_value}"
    )


def third_question():
    print("Third question")

    g = Graph()

    arad = Node("Arad")
    bucharest = Node("Bucharest")
    craiova = Node("Craiova")
    dobreta = Node("Dobreta")
    eforie = Node("Eforie")
    fagaras = Node("Fagaras")
    giurgiu = Node("Giurgiu")
    hirsova = Node("Hirsova")
    iasi = Node("Iasi")
    lugoj = Node("Lugoj")
    mehadia = Node("Mehadia")
    neamt = Node("Neamt")
    oradea = Node("Oradea")
    pitesti = Node("Pitesti")
    rimnicu_vilcea = Node("Rimnicu Vilcea")
    sibiu = Node("Sibiu")
    timisoara = Node("Timisoara")
    urziceni = Node("Urziceni")
    vaslui = Node("Vaslui")
    zerind = Node("Zerind")

    g.add_edge((arad, zerind, 75.0))
    g.add_edge((zerind, oradea, 71.0))
    g.add_edge((oradea, sibiu, 151.0))
    g.add_edge((sibiu, arad, 140.0))
    g.add_edge((arad, timisoara, 118.0))
    g.add_edge((timisoara, lugoj, 111.0))
    g.add_edge((lugoj, mehadia, 70.0))
    g.add_edge((mehadia, dobreta, 75.0))
    g.add_edge((dobreta, craiova, 120.0))
    g.add_edge((craiova, rimnicu_vilcea, 146.0))
    g.add_edge((rimnicu_vilcea, sibiu, 80.0))
    g.add_edge((sibiu, fagaras, 99.0))
    g.add_edge((fagaras, bucharest, 211.0))
    g.add_edge((bucharest, pitesti, 101.0))
    g.add_edge((pitesti, rimnicu_vilcea, 97.0))
    g.add_edge((craiova, pitesti, 138.0))
    g.add_edge((pitesti, bucharest, 101.0))
    g.add_edge((bucharest, giurgiu, 90.0))
    g.add_edge((bucharest, urziceni, 85.0))
    g.add_edge((urziceni, hirsova, 98.0))
    g.add_edge((hirsova, eforie, 86.0))
    g.add_edge((urziceni, vaslui, 142.0))
    g.add_edge((vaslui, iasi, 92.0))
    g.add_edge((iasi, neamt, 87.0))

    heuristics = {
        arad: 366.0,
        bucharest: 0,
        craiova: 160,
        dobreta: 242,
        eforie: 161,
        fagaras: 176,
        giurgiu: 77,
        hirsova: 151,
        iasi: 226,
        lugoj: 244,
        mehadia: 241,
        neamt: 234,
        oradea: 380,
        pitesti: 100,
        rimnicu_vilcea: 193,
        sibiu: 253,
        timisoara: 329,
        urziceni: 80,
        vaslui: 199,
        zerind: 374,
    }

    print("Greedy Best First Search")

    gbfs = GBFS(
        graph=g,
        heuristics=heuristics,
        include_origin=True,
        include_destination=True,
    )

    path = gbfs.shortest_path(Node("arad"), Node("bucharest"))

    print(f"Shortest path from Arad to Bucharest using GBFS is {path}")

    print("A* Search")

    astar = AStar(
        graph=g,
        heuristics=heuristics,
        include_origin=True,
        include_destination=True,
    )

    path = astar.shortest_path(Node("arad"), Node("bucharest"))

    print(f"Shortest path from Arad to Bucharest using A* is {path}")


def main():
    first_question()
    second_question()
    third_question()


if __name__ == "__main__":
    main()
