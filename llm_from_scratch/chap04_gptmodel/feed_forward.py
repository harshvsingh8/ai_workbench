import torch.nn as nn
import gelu

class FeedForward(nn.Module):
    """
    A simple feed-forward neural network module.
    This is a standard component in Transformer architectures, applied after the
    self-attention layer.
    """
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        # nn.Sequential is a container that passes data through a sequence of modules.
        # The setup follows the standard Transformer architecture for the feed-forward network.
        self.layers = nn.Sequential(
            # 1. First Linear Layer (Expansion):
            # This layer expands the input embedding dimension (`emb_dim`) by a factor of 4.
            # This expansion creates a higher-dimensional space where the model can
            # learn more complex representations and interactions between features.
            # It's a common practice in Transformer models (e.g., in the paper "Attention Is All You Need").
            nn.Linear(emb_dim, 4 * emb_dim),

            # 2. GELU Activation:
            # The non-linear activation function is applied element-wise to the output
            # of the first linear layer. This is crucial for the model to learn
            # non-linear patterns in the data. Without this, the two linear layers
            # would collapse into a single linear transformation.
            gelu.GELU(),

            # 3. Second Linear Layer (Projection):
            # This layer projects the high-dimensional representation from the GELU activation
            # back down to the original embedding dimension (`emb_dim`). This prepares the
            # output to be added to the output of the self-attention layer (via a residual connection).
            nn.Linear(4 * emb_dim, emb_dim),
        )
    
    def forward(self, x):
        """
        The forward pass simply passes the input tensor `x` through the sequential layers.
        """
        return self.layers(x)

