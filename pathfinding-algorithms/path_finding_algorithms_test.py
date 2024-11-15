from typing import List

import pytest
from path_finding_algorithms import GBFS, Dijsktra, Graph, Node, greedy_beverage_machine


def test_djkastra():
    # Arrange
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

    # Act
    path = dj.shortest_path(node_b, node_f)
    expected = [node_b, node_e, node_c, node_f]

    # Assert
    assert path == expected


@pytest.mark.parametrize(
    "user_cash_input,beverage_cost,expected_return",
    [
        (1.0, 0.7, 0.2),  # Small change
        (1.0, 1.0, 0.0),  # Exact payment
        (2.0, 1.99, 0.01),  # Smallest possible change
        (3.0, 2.0, 1.0),  # Large change, single coin
        (1.0, 0.68, 0.2),  # Change not representable with single coin
        (10.0, 0.01, 1.0),  # Minimal cost and high input
        (0.05, 0.04, 0.01),  # Smallest possible coin
        (5.0, 4.83, 0.1),  # All change coins available
    ],
)
def test_greedy_beverage_machine(
    user_cash_input: float, beverage_cost: float, expected_return: float
):
    assert greedy_beverage_machine(user_cash_input, beverage_cost) == expected_return


@pytest.mark.parametrize(
    "source,destination,expected_path",
    [
        (
            Node("arad"),
            Node("bucharest"),
            [
                (Node("sibiu")),
                (Node("fagaras")),
            ],
        ),
        (
            Node("arad"),
            Node("oradea"),
            [
                (Node("zerind")),
            ],
        ),
        (
            Node("arad"),
            Node("arad"),
            [],
        ),
    ],
)
def test_gbfs(source: Node, destination: Node, expected_path: List[Node]):
    # Arrange
    g = Graph()

    arad = Node("Arad", 29.0, 192.0)
    bucharest = Node("Bucharest", 268.0, 55.0)
    craiova = Node("Craiova", 163.0, 22.0)
    dobreta = Node("Dobreta", 91.0, 32.0)
    eforie = Node("Eforie", 420.0, 28.0)
    fagaras = Node("Fagaras", 208.0, 157.0)
    giurgiu = Node("Giurgiu", 264.0, 8.0)
    hirsova = Node("Hirsova", 396.0, 74.0)
    iasi = Node("Iasi", 347.0, 204.0)
    lugoj = Node("Lugoj", 91.0, 98.0)
    mehadia = Node("Mehadia", 93.0, 65.0)
    neamt = Node("Neamt", 290.0, 229.0)
    oradea = Node("Oradea", 62.0, 258.0)
    pitesti = Node("Pitesti", 220.0, 88.0)
    rimnicu_vilcea = Node("Rimnicu Vilcea", 147.0, 124.0)
    sibiu = Node("Sibiu", 126.0, 164.0)
    timisoara = Node("Timisoara", 32.0, 124.0)
    urziceni = Node("Urziceni", 333.0, 74.0)
    vaslui = Node("Vaslui", 376.0, 153.0)
    zerind = Node("Zerind", 44.0, 225.0)

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

    gbfs = GBFS(g)

    # Act
    path = gbfs.shortest_path(source, destination)

    # Assert
    assert path == expected_path
