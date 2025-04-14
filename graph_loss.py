import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.utils import dist

class SpectralGraphMatchingLoss(nn.Module):
    def __init__(
            self,
            similarity_threshold=0.5,
            temperature=0.1,
            k_eigvals=10,
            gather_distributed=False
    ):
        """
        Spectral graph matching loss that compares the structural properties
        of graphs constructed from two views of the data.

        Parameters:
        - similarity_threshold (float): Threshold for constructing edges in the similarity graph.
                                       Values above this threshold create an edge.
        - temperature (float): Temperature for the similarity calculations.
        - k_eigvals (int): Number of smallest eigenvalues to use for spectral comparison.
        - gather_distributed (bool): Whether to gather data across multiple distributed processes.
        """
        super(SpectralGraphMatchingLoss, self).__init__()
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.k_eigvals = k_eigvals
        self.use_distributed = gather_distributed and dist.world_size() > 1

    def _construct_adjacency_matrix(self, embeddings):
        """
        Construct adjacency matrix from embeddings using cosine similarity.

        Parameters:
        - embeddings (torch.Tensor): Normalized embeddings, shape (batch_size, embedding_dim).

        Returns:
        - torch.Tensor: Adjacency matrix.
        """
        # Compute similarity matrix
        similarities = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Apply threshold to create adjacency matrix (undirected graph)
        adj_matrix = (similarities > self.similarity_threshold).float()

        # Set diagonal to zero (no self-loops)
        adj_matrix = adj_matrix - torch.eye(adj_matrix.size(0), device=adj_matrix.device) * adj_matrix.diagonal()

        return adj_matrix

    def _compute_laplacian(self, adj_matrix):
        """
        Compute the normalized graph Laplacian matrix.

        Parameters:
        - adj_matrix (torch.Tensor): Adjacency matrix.

        Returns:
        - torch.Tensor: Normalized Laplacian matrix.
        """
        # Compute degree matrix
        degrees = adj_matrix.sum(dim=1)

        # Add small epsilon to avoid division by zero
        degrees = degrees + 1e-10

        # Compute D^(-1/2)
        d_inv_sqrt = torch.pow(degrees, -0.5)
        d_inv_sqrt_matrix = torch.diag(d_inv_sqrt)

        # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        laplacian = torch.eye(adj_matrix.size(0), device=adj_matrix.device) - torch.matmul(
            torch.matmul(d_inv_sqrt_matrix, adj_matrix), d_inv_sqrt_matrix
        )

        return laplacian

    def _compute_spectral_loss(self, lap1, lap2):
        """
        Compute spectral loss between two Laplacian matrices.

        Parameters:
        - lap1 (torch.Tensor): First Laplacian matrix.
        - lap2 (torch.Tensor): Second Laplacian matrix.

        Returns:
        - torch.Tensor: Spectral loss value.
        """
        # Compute eigenvalues (sorted in ascending order)
        eigvals1, _ = torch.linalg.eigh(lap1)
        eigvals2, _ = torch.linalg.eigh(lap2)

        # Use k smallest eigenvalues (excluding zero eigenvalues if possible)
        k = min(self.k_eigvals, len(eigvals1) - 1, len(eigvals2) - 1)
        eigvals1_k = eigvals1[1:k + 1]  # Skip the first eigenvalue (should be zero or near zero)
        eigvals2_k = eigvals2[1:k + 1]

        # Compute mean squared error between eigenvalues
        spectral_loss = F.mse_loss(eigvals1_k, eigvals2_k)

        return spectral_loss

    def forward(self, out0, out1, labels=None):
        """
        Forward pass to compute the spectral graph matching loss.

        Parameters:
        - out0 (torch.Tensor): Embeddings from first view, shape (batch_size, embedding_dim).
        - out1 (torch.Tensor): Embeddings from second view, shape (batch_size, embedding_dim).
        - labels (torch.Tensor, optional): Class labels for use in distributed gathering.

        Returns:
        - torch.Tensor: The computed spectral loss.
        """
        # Normalize embeddings
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        # Handle distributed case
        if self.use_distributed:
            out0_large = torch.cat(dist.gather(out0), 0)
            out1_large = torch.cat(dist.gather(out1), 0)
        else:
            out0_large = out0
            out1_large = out1

        # Construct adjacency matrices
        adj_matrix1 = self._construct_adjacency_matrix(out0_large)
        adj_matrix2 = self._construct_adjacency_matrix(out1_large)

        # Compute Laplacian matrices
        laplacian1 = self._compute_laplacian(adj_matrix1)
        laplacian2 = self._compute_laplacian(adj_matrix2)

        # Compute spectral loss
        spectral_loss = self._compute_spectral_loss(laplacian1, laplacian2)

        return spectral_loss


class AdaptiveSpectralGraphMatchingLoss(SpectralGraphMatchingLoss):
    """
    Adaptive version of the spectral graph matching loss that automatically
    adjusts the similarity threshold based on batch statistics.
    """

    def __init__(
            self,
            percentile=90,
            min_edges_percent=10,
            max_edges_percent=50,
            temperature=0.5,
            k_eigvals=10,
            gather_distributed=False
    ):
        """
        Parameters:
        - percentile (float): Percentile of similarity values to use as threshold.
        - min_edges_percent (float): Minimum percentage of edges to maintain.
        - max_edges_percent (float): Maximum percentage of edges to allow.
        - temperature (float): Temperature for the similarity calculations.
        - k_eigvals (int): Number of smallest eigenvalues to use for spectral comparison.
        - gather_distributed (bool): Whether to gather data across multiple distributed processes.
        """
        super(AdaptiveSpectralGraphMatchingLoss, self).__init__(
            similarity_threshold=0.0,  # Will be determined adaptively
            temperature=temperature,
            k_eigvals=k_eigvals,
            gather_distributed=gather_distributed
        )
        self.percentile = percentile
        self.min_edges_percent = min_edges_percent / 100.0
        self.max_edges_percent = max_edges_percent / 100.0

    def _adaptive_threshold(self, similarities):
        """
        Adaptively determine similarity threshold.

        Parameters:
        - similarities (torch.Tensor): Similarity matrix.

        Returns:
        - float: Adaptive threshold.
        """
        # Flatten the upper triangular part (excluding diagonal)
        n = similarities.size(0)
        indices = torch.triu_indices(n, n, 1)
        sim_values = similarities[indices[0], indices[1]]

        # Compute the threshold based on percentile
        threshold = torch.quantile(sim_values, self.percentile / 100.0)

        # Check if the threshold creates too few or too many edges
        mask = (similarities > threshold).float()
        edge_density = mask.sum() / (n * (n - 1))

        if edge_density < self.min_edges_percent:
            # Lower threshold to include more edges
            threshold = torch.quantile(sim_values, 1.0 - self.min_edges_percent)
        elif edge_density > self.max_edges_percent:
            # Raise threshold to include fewer edges
            threshold = torch.quantile(sim_values, 1.0 - self.max_edges_percent)

        return threshold

    def _construct_adjacency_matrix(self, embeddings):
        """
        Construct adjacency matrix with adaptive threshold.

        Parameters:
        - embeddings (torch.Tensor): Normalized embeddings.

        Returns:
        - torch.Tensor: Adjacency matrix.
        """
        # Compute similarity matrix
        similarities = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Determine threshold adaptively
        threshold = self._adaptive_threshold(similarities)

        # Apply threshold to create adjacency matrix
        adj_matrix = (similarities > threshold).float()

        # Set diagonal to zero (no self-loops)
        adj_matrix = adj_matrix - torch.eye(adj_matrix.size(0), device=adj_matrix.device) * adj_matrix.diagonal()

        return adj_matrix