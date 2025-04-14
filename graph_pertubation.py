import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj
import random

class EdgePerturbationTransform(BaseTransform):
    """
    Edge perturbation transform: randomly adds and removes edges with given probabilities.
    """

    def __init__(self, add_prob=0.1, drop_prob=0.1):
        self.add_prob = add_prob
        self.drop_prob = drop_prob

    def __call__(self, data):
        # Create a copy of the data object
        new_data = data.clone()

        edge_index = data.edge_index
        num_nodes = data.x.size(0)

        # Drop edges
        mask = torch.rand(edge_index.size(1)) > self.drop_prob
        new_edge_index = edge_index[:, mask]

        # Add random edges
        if self.add_prob > 0:
            # Calculate how many edges to add
            num_edges_to_add = int(edge_index.size(1) * self.add_prob)
            random_edge_source = torch.randint(0, num_nodes, (num_edges_to_add,))
            random_edge_target = torch.randint(0, num_nodes, (num_edges_to_add,))
            random_edges = torch.stack([random_edge_source, random_edge_target], dim=0)

            # Combine existing and new edges
            new_edge_index = torch.cat([new_edge_index, random_edges], dim=1)

        # Update edge index
        new_data.edge_index = new_edge_index

        return new_data


class FeatureMaskTransform(BaseTransform):
    """
    Feature masking transform: randomly masks node features with given probability.
    """

    def __init__(self, mask_prob=0.1):
        self.mask_prob = mask_prob

    def __call__(self, data):
        # Create a copy of the data object
        new_data = data.clone()

        # Create mask for features
        feature_mask = torch.rand(data.x.shape) > self.mask_prob

        # Apply mask (set masked features to 0)
        new_data.x = data.x * feature_mask

        return new_data


class FeatureDropoutTransform(BaseTransform):
    """
    Feature dropout transform: applies dropout to node features.
    """

    def __init__(self, dropout_prob=0.1):
        self.dropout_prob = dropout_prob

    def __call__(self, data):
        # Create a copy of the data object
        new_data = data.clone()

        # Apply dropout to features
        mask = torch.rand(data.x.shape) > self.dropout_prob
        new_data.x = data.x * mask / (1 - self.dropout_prob)  # Scale the remaining features

        return new_data


class SubgraphTransform(BaseTransform):
    """
    Subgraph sampling transform: samples a connected subgraph from the original graph.
    """

    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def __call__(self, data):
        # Create a copy of the data object
        new_data = data.clone()

        num_nodes = data.x.size(0)
        num_sampled_nodes = max(int(num_nodes * self.ratio), 1)

        # Randomly select starting node
        start_idx = random.randint(0, num_nodes - 1)

        # Simple BFS to get connected subgraph
        visited = {start_idx}
        queue = [start_idx]
        edge_index = data.edge_index.t().numpy()

        while len(visited) < num_sampled_nodes and queue:
            current = queue.pop(0)

            # Find neighbors
            neighbors = edge_index[edge_index[:, 0] == current, 1]
            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < num_sampled_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Convert visited to a list and sort
        nodes = sorted(list(visited))
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[nodes] = True

        # Map old indices to new ones
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes)}

        # Filter edges that have both endpoints in the subgraph
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        subgraph_edge_index = data.edge_index[:, edge_mask]

        # Remap node indices
        for i in range(subgraph_edge_index.size(1)):
            subgraph_edge_index[0, i] = old_to_new[subgraph_edge_index[0, i].item()]
            subgraph_edge_index[1, i] = old_to_new[subgraph_edge_index[1, i].item()]

        # Update data
        new_data.x = data.x[nodes]
        new_data.edge_index = subgraph_edge_index

        if hasattr(data, 'y') and data.y is not None:
            new_data.y = data.y

        return new_data


class DiffusionTransform(BaseTransform):
    """
    Diffusion transform: applies heat kernel diffusion to node features.
    """

    def __init__(self, alpha=0.2, num_steps=10):
        self.alpha = alpha
        self.num_steps = num_steps

    def __call__(self, data):
        # Create a copy of the data object
        new_data = data.clone()

        edge_index = data.edge_index
        num_nodes = data.x.size(0)

        # Create adjacency matrix
        adj = to_dense_adj(edge_index)[0]

        # Add self-loops
        adj = adj + torch.eye(num_nodes)

        # Normalize adjacency matrix
        degrees = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(degrees, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt_matrix = torch.diag(deg_inv_sqrt)
        norm_adj = torch.mm(torch.mm(deg_inv_sqrt_matrix, adj), deg_inv_sqrt_matrix)

        # Apply diffusion
        x = data.x
        for _ in range(self.num_steps):
            x = (1 - self.alpha) * x + self.alpha * torch.mm(norm_adj, x)

        new_data.x = x

        return new_data
