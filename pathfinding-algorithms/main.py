#!/usr/bin/env python3

import abc
from typing import Dict, List, Tuple
from heapq import heapify, heappop, heappush


type Node = str


class Graph:
    nodes: List[Node]
    edges: Dict[Node, Dict[Node, float]]

    def __init__(
        self,
        nodes: List[Node] = [],
        edges: List[Tuple[Node, Node, float]] = [],
        undirected: bool = True,
    ) -> None:
        self.nodes = []
        self.edges = {}
        self.undirected = undirected

        [self.add_node(node) for node in nodes]
        [self.add_edge(edge) for edge in edges]

    def add_node(self, node: Node) -> None:
        if node not in self.nodes:
            self.nodes.append(node)

    def add_edge(self, edge: Tuple[Node, Node, float]) -> None:
        source, dest, weight = edge

        self.add_node(source)
        self.add_node(dest)

        if source not in self.edges:
            self.edges[source] = {}

        if dest not in self.edges:
            self.edges[dest] = {}

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


def first_question():
    print("First question")

    g = Graph()
    g.add_edge(("A", "B", 3.0))
    g.add_edge(("A", "C", 3.0))

    g.add_edge(("B", "D", 3.5))
    g.add_edge(("B", "E", 2.8))

    g.add_edge(("C", "E", 2.8))
    g.add_edge(("C", "F", 3.5))

    g.add_edge(("D", "E", 3.1))
    g.add_edge(("D", "G", 10.0))

    g.add_edge(("E", "G", 7.0))
    g.add_edge(("F", "G", 2.5))

    dj = Dijsktra(graph=g)

    path = dj.shortest_path("B", "F")
    print(f"shortest path from B to F is {path}")


def main():
    first_question()


if __name__ == "__main__":
    main()
