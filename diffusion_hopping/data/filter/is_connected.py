import itertools
from typing import List, Set

import numpy as np
from scipy import spatial

from diffusion_hopping.data import ProteinLigandComplex


class IsConnectedFilter:
    def __init__(self, cutoff: int = 5) -> None:
        self.cutoff = cutoff

    def _is_connected(self, adjacency_matrix: np.ndarray) -> bool:
        n = len(adjacency_matrix)
        if n < 2:
            return True
        adjacency_list = [
            frozenset(np.argwhere(adjacency_matrix[i]).flatten()) for i in range(n)
        ]
        return len(self._bfs(adjacency_list, 0)) == n

    def _bfs(self, adj: List[frozenset[int]], source: int) -> Set[int]:
        seen = set()
        next_nodes = {source}
        while next_nodes:
            frontier = next_nodes
            next_nodes = set()
            for v in frontier:
                if v not in seen:
                    seen.add(v)
                    next_nodes.update(adj[v])
        return seen

    def __call__(self, complex: ProteinLigandComplex) -> bool:
        protein = complex.protein.pandas_pdb()
        df = protein.df["ATOM"]

        chain_coords = [
            chain[["x_coord", "y_coord", "z_coord"]].values
            for _, chain in df.groupby("chain_id")
        ]
        chain_count = len(chain_coords)
        distance = np.zeros((chain_count, chain_count))
        for i, j in itertools.combinations(range(chain_count), 2):
            distance[i][j] = distance[j][i] = spatial.distance.cdist(
                chain_coords[i], chain_coords[j]
            ).min(initial=np.inf)

        adjancency_matrix = distance < self.cutoff
        np.fill_diagonal(adjancency_matrix, False)
        return self._is_connected(adjancency_matrix)

    def __repr__(self) -> str:
        return f"IsConnectedFilter(cutoff={self.cutoff})"

    def __str__(self) -> str:
        return self.__repr__()
