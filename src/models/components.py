import torch
import torch.nn as nn

import configs.config


class MultiHeadTransformerAggregator(nn.Module):
    """
    Replaces a simple linear attention aggregator with a small multi-head
    self-attention Transformer.

    We prepend a learnable [CLS] token to the microbe embeddings, run them
    through Transformer layers, and then take the final hidden state of [CLS]
    as the sample representation.
    """
    def __init__(self, config:configs.config.Config):
        super().__init__()
        # A learnable [CLS] token (one vector of size embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.EMBEDDING_DIM))

        # Define a small Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.NHEAD_CLASSIFIER,
            dim_feedforward=4 * config.EMBEDDING_DIM,  # typical feedforward size is 2-4x d_model
            dropout=config.AGGREGATOR_DROPOUT_CLASSIFIER,
            batch_first=True  # We'll use (batch_size, seq_len, emb_dim) ordering
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS_CLASSIFIER)

        self.dropout = nn.Dropout(config.DROPOUT_LINEAR_CLASSIFIER)

    def forward(self, microbe_embeddings,attn_mask):
        """
        microbe_embeddings: shape (batch_size, num_bacteria, embedding_dim)
        Returns:
            cls_embedding (batch_size, embedding_dim): The final hidden state of the [CLS] token
            hidden_states (batch_size, num_bacteria+1, embedding_dim):
                The entire sequence of final hidden states (if you need it)
        """
        B, N, D = microbe_embeddings.shape

        # Expand [CLS] token for each sample in the batch
        # shape => (batch_size, 1, embedding_dim)
        cls_token_expanded = self.cls_token.expand(B, -1, -1)

        # Concatenate [CLS] token at the start of the sequence
        microbe_embeddings= self.dropout(microbe_embeddings)
        # => (batch_size, N+1, embedding_dim)
        x = torch.cat([cls_token_expanded, microbe_embeddings], dim=1)

        # add in the attention mask the [CLS] token
        # => (batch_size, N+1)
        attn_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool).to(microbe_embeddings.device), attn_mask], dim=1)

        # Pass through Transformer encoder
        # => (batch_size, N+1, embedding_dim)
        x = self.transformer(x,src_key_padding_mask=attn_mask)

        # Final hidden state at [CLS] index = x[:, 0, :]
        cls_embedding = x[:, 0, :]
        return cls_embedding, x

