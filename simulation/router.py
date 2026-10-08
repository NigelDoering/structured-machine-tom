import networkx as nx
import numpy as np
import osmnx as ox
import gc
from pathlib import Path


class Router:
    """
    Shortest-path router over the UCSD pedestrian graph.

    Finds the k shortest simple paths between two nodes and samples one
    probabilistically using Boltzmann weighting: p(path) ∝ exp(-length / tau).
    Shorter paths receive higher probability; tau controls how sharply the
    distribution peaks on the shortest path.

    The underlying MultiDiGraph is converted once to a simple DiGraph (keeping
    the minimum-length edge for each node pair) so that
    nx.shortest_simple_paths runs efficiently.
    """

    def __init__(
        self,
        graph_path: str | Path,
        k: int = 5,
        temperature: float = 100.0,
    ):
        """
        Parameters
        ----------
        graph_path  : path to the .graphml walk graph
        k           : number of candidate shortest paths to generate
        temperature : Boltzmann temperature in metres. Lower = more peaked on
                      the shortest path. 100m is a reasonable default: a path
                      that is 100m longer than the shortest is exp(-1) ≈ 37%
                      as likely to be chosen.
        """
        self.k = k
        self.temperature = temperature

        try:
            raw = ox.load_graphml(graph_path)
        except TypeError:
            gc.collect()
            raw = ox.load_graphml(graph_path)
        self.G = self._to_simple_digraph(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        origin: int,
        destination: int,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """
        Return a sampled path from origin to destination as a list of node ids.

        If fewer than k paths exist only those are used. Returns a single-node
        list if origin == destination. Returns an empty list if no path exists.
        """
        if origin == destination:
            return [origin]

        if rng is None:
            rng = np.random.default_rng()

        paths, lengths = self._k_shortest(origin, destination)
        if not paths:
            return []

        idx = self._sample_index(lengths, rng)
        return paths[idx]

    def shortest_length(self, origin: int, destination: int) -> float | None:
        """Return the length of the shortest path in metres, or None."""
        try:
            return nx.shortest_path_length(self.G, origin, destination, weight='length')
        except nx.NetworkXNoPath:
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_simple_digraph(G_multi: nx.MultiDiGraph) -> nx.DiGraph:
        """
        Collapse MultiDiGraph to DiGraph, keeping the minimum-length edge
        for each (u, v) pair. nx.shortest_simple_paths runs orders of
        magnitude faster on a simple DiGraph.
        """
        G = nx.DiGraph()
        G.add_nodes_from(G_multi.nodes(data=True))
        for u, v, data in G_multi.edges(data=True):
            length = float(data.get('length', 1.0))
            if G.has_edge(u, v):
                if length < G[u][v]['length']:
                    G[u][v]['length'] = length
            else:
                G.add_edge(u, v, length=length)
        return G

    def _k_shortest(
        self, origin: int, destination: int
    ) -> tuple[list[list[int]], list[float]]:
        """Return up to k shortest simple paths and their lengths."""
        if self.k <= 1:
            try:
                path = nx.shortest_path(self.G, origin, destination, weight='length')
                length = nx.path_weight(self.G, path, weight='length')
                return [path], [length]
            except (nx.NetworkXNoPath, nx.NodeNotFound, AttributeError, TypeError, SystemError):
                return [], []

        paths, lengths = [], []
        try:
            gen = nx.shortest_simple_paths(self.G, origin, destination, weight='length')
            for path in gen:
                paths.append(path)
                lengths.append(nx.path_weight(self.G, path, weight='length'))
                if len(paths) >= self.k:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound, AttributeError, TypeError, SystemError):
            pass
        return paths, lengths

    def _sample_index(
        self, lengths: list[float], rng: np.random.Generator
    ) -> int:
        """Boltzmann sample over path indices: p(i) ∝ exp(-length_i / tau)."""
        arr = np.array(lengths, dtype=float)
        log_w = -arr / self.temperature
        log_w -= log_w.max()          # numerical stability
        w = np.exp(log_w)
        w /= w.sum()
        return int(rng.choice(len(lengths), p=w))
