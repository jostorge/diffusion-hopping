import torch


def get_squared_distance(pos, edge_index):
    row, col = edge_index
    return torch.sum((pos[row] - pos[col]) ** 2, dim=-1, keepdim=True)
