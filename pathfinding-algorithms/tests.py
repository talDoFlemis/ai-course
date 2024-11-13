from main import Dijsktra, Graph
import unittest


class TestDjKastra(unittest.TestCase):
    def test_djkastra(self):
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
        expected = ["B", "E", "C", "F"]
        self.assertListEqual(expected, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
