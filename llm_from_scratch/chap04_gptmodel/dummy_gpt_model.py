import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    """
    Dummy implementation of a GPT-style language model.
    This serves as a scaffold for building the full GPT model architecture.
    
    The model follows the standard transformer architecture with:
    - Token and positional embeddings
    - Multiple transformer blocks (attention + feedforward)
    - Final layer normalization
    - Output projection to vocabulary
    """
    def __init__(self, cfg):
        """
        Initialize the GPT model with the given configuration.
        
        Args:
            cfg (dict): Configuration dictionary containing:
                - vocab_size: Size of the vocabulary 
                - emb_dim: Embedding dimension
                - context_length: Maximum sequence length the model can handle
                - drop_rate: Dropout rate for regularization
                - n_layers: Number of transformer blocks
        """
        super().__init__()
        
        # Token embeddings: Convert token IDs to dense vectors
        # Maps each token in vocabulary to a learned embedding vector
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Position embeddings: Add positional information to tokens
        # Since transformers have no inherent notion of sequence order,
        # we add learned positional embeddings to give the model position awareness
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Embedding dropout: Regularization to prevent overfitting
        # Applied to the sum of token and positional embeddings
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Transformer blocks: Core of the model - will contain attention and feedforward layers
        # Each block will implement self-attention and feed-forward networks
        # Currently using dummy blocks that will be replaced with full implementation
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization: Stabilizes training and improves convergence
        # Applied before the output projection
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # Output head: Projects final hidden states back to vocabulary space
        # No bias term is used following GPT convention
        # This will produce logits for next token prediction
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias = False
        )

    def forward(self, in_idx):
        """
        Forward pass through the GPT model.
        
        Args:
            in_idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Logits for next token prediction of shape (batch_size, seq_len, vocab_size)
        """
        # Get input dimensions
        batch_size, seq_len = in_idx.shape
        
        # Convert token indices to embeddings
        # Shape: (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        
        # Get positional embeddings for the sequence
        # We create a range tensor for positions and embed them
        # Shape: (seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # Combine token and positional embeddings
        # Broadcasting handles the batch dimension automatically
        # Shape: (batch_size, seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        
        # Apply dropout for regularization
        x = self.drop_emb(x)
        
        # Pass through all transformer blocks
        # Each block will apply self-attention and feed-forward transformations
        # Currently dummy blocks that just pass through input unchanged
        x = self.trf_blocks(x)
        
        # Apply final layer normalization
        # Helps with training stability
        x = self.final_norm(x)

        # Project to vocabulary space to get logits
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)
        return logits


class DummyLayerNorm(nn.Module):
    """
    Dummy implementation of Layer Normalization.
    
    Layer normalization will normalize activations across the feature dimension,
    helping with training stability and convergence. It's applied after attention
    and feed-forward layers in transformers.
    
    TODO: Implement actual layer normalization with:
    - Mean and variance calculation across the last dimension
    - Learnable scale (gamma) and shift (beta) parameters
    - Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize Layer Normalization.
        
        Args:
            normalized_shape (int): Size of the feature dimension to normalize
            eps (float): Small constant for numerical stability
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # TODO: Add learnable parameters
        # self.gamma = nn.Parameter(torch.ones(normalized_shape))  # Scale parameter
        # self.beta = nn.Parameter(torch.zeros(normalized_shape))  # Shift parameter

    def forward(self, x):
        """
        Apply layer normalization to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., normalized_shape)
            
        Returns:
            torch.Tensor: Normalized tensor (currently just returns input unchanged)
        """
        # TODO: Implement actual layer normalization
        # mean = x.mean(dim=-1, keepdim=True)
        # var = x.var(dim=-1, keepdim=True, unbiased=False)
        # normalized = (x - mean) / torch.sqrt(var + self.eps)
        # return normalized * self.gamma + self.beta
        
        # For now, just return input unchanged (dummy implementation)
        return x


class DummyTransformerBlock(nn.Module):
    """
    Dummy implementation of a Transformer block.
    
    A complete transformer block will consist of:
    1. Multi-head self-attention mechanism
    2. Residual connection around attention
    3. Layer normalization after attention
    4. Feed-forward network (MLP)
    5. Residual connection around feed-forward
    6. Layer normalization after feed-forward
    
    This follows the "Pre-LN" architecture where layer norm is applied
    before the attention and feed-forward layers, not after.
    
    TODO: Implement the full transformer block with:
    - MultiHeadAttention layer
    - Feed-forward network (typically 2 linear layers with activation)
    - Layer normalization layers
    - Residual connections
    - Dropout for regularization
    """
    def __init__(self, cfg):
        """
        Initialize a transformer block.
        
        Args:
            cfg (dict): Configuration dictionary containing:
                - emb_dim: Embedding/hidden dimension
                - n_heads: Number of attention heads
                - drop_rate: Dropout rate
                - context_length: Maximum sequence length (for attention masking)
        """
        super().__init__()
        self.cfg = cfg
        
        # TODO: Initialize the actual components:
        # self.ln1 = LayerNorm(cfg["emb_dim"])  # Layer norm before attention
        # self.attn = MultiHeadAttention(cfg)    # Multi-head self-attention
        # self.ln2 = LayerNorm(cfg["emb_dim"])  # Layer norm before feed-forward
        # self.ff = FeedForward(cfg)             # Feed-forward network
        # self.drop_resid = nn.Dropout(cfg["drop_rate"])  # Residual dropout

    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)
            
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # TODO: Implement the actual forward pass:
        # # Pre-layer norm architecture
        # # Attention block with residual connection
        # attn_out = self.attn(self.ln1(x))
        # x = x + self.drop_resid(attn_out)
        # 
        # # Feed-forward block with residual connection  
        # ff_out = self.ff(self.ln2(x))
        # x = x + self.drop_resid(ff_out)
        # 
        # return x
        
        # For now, just return input unchanged (dummy implementation)
        return x