import jax.numpy as jnp
from jax import lax
import haiku as hk
import jax
import jax.numpy as jnp
import functools

def gather_edges_jax(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    B, N, _, C = edges.shape
    _, _, K = neighbor_idx.shape
    ii = jnp.arange(B)[:, None, None]
    jj = jnp.arange(N)[:, None]
    neighbor_idx_expanded = neighbor_idx[:, :, :, None].repeat(C, axis=3)
    return edges[ii, jj, neighbor_idx_expanded]

def gather_nodes_jax(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    B, N, C = nodes.shape
    _, _, K = neighbor_idx.shape
    ii = jnp.arange(B)[:, None, None]
    jj = jnp.arange(N)[:, None]
    neighbor_idx_expanded = neighbor_idx[:, :, :, None].repeat(C, axis=3)
    return nodes[ii, jj, neighbor_idx_expanded]

def gather_nodes_t_jax(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    B, K = neighbor_idx.shape
    C = nodes.shape[-1]
    ii = jnp.arange(B)[:, None]
    neighbor_idx_expanded = neighbor_idx[:, :, None].repeat(C, axis=2)
    return nodes[ii, neighbor_idx_expanded]

def cat_neighbors_nodes_jax(h_nodes, h_neighbors, E_idx):
    h_nodes_gathered = gather_nodes_jax(h_nodes, E_idx)
    h_nn = jnp.concatenate([h_neighbors, h_nodes_gathered], axis=-1)
    return h_nn


class PositionWiseFeedForwardJax(hk.Module):
    def __init__(self, num_hidden, num_ff, name=None):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_ff = num_ff
        self.act = functools.partial(jax.nn.gelu, approximate=False)

    def __call__(self, h_V):
        W_in = hk.Linear(self.num_ff, name="W_in")
        W_out = hk.Linear(self.num_hidden, name="W_out")
        h = self.act(W_in(h_V))
        h = W_out(h)
        return h


class DecLayerJ(hk.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout_rate = dropout
        self.act = jax.nn.gelu

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
        # Define the linear transformations within the call method
        W1 = hk.Linear(self.num_hidden, name="W1")
        W2 = hk.Linear(self.num_hidden, name="W2")
        W3 = hk.Linear(self.num_hidden, name="W3")
        dense = PositionWiseFeedForwardJax(self.num_hidden, self.num_hidden * 4)

        # Concatenate h_V_i to h_E_ij
        h_V_expand = jnp.expand_dims(h_V, axis=-2)
        h_EV = jnp.concatenate([h_V_expand, h_E], axis=-1)

        # Apply transformations
        h_message = W3(self.act(W2(self.act(W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = jnp.sum(h_message, axis=-2) / self.scale

        # Apply dropout and normalization
        dropout1 = hk.dropout(hk.next_rng_key(), self.dropout_rate, dh)
        h_V = h_V + dropout1

        # Position-wise feedforward
        dh = dense(h_V)
        dropout2 = hk.dropout(hk.next_rng_key(), self.dropout_rate, dh)
        h_V = h_V + dropout2

        if mask_V is not None:
            mask_V = mask_V[..., None]
            h_V = mask_V * h_V

        return h_V

class PositionalEncodingsJax(hk.Module):
    def __init__(self, num_embeddings, max_relative_feature=32, name=None):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        # Initialize the linear layer for embedding the one-hot encoded positions
        self.linear = hk.Linear(num_embeddings, name="positional_embedding_linear")

    def __call__(self, offset, mask):
        # Clip the offset values and apply the mask
        d = jnp.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        d = d * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        # One-hot encode the clipped and masked offsets
        d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        # Pass the one-hot encoded offsets through the linear layer
        E = self.linear(d_onehot.astype(jnp.float32))
        return E

class DecLayer(hk.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout_rate = dropout
        self.act = jax.nn.gelu

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
        # Define the linear transformations within the call method
        W1 = hk.Linear(self.num_hidden, name="W1")
        W2 = hk.Linear(self.num_hidden, name="W2")
        W3 = hk.Linear(self.num_hidden, name="W3")
        dense = PositionWiseFeedForwardJax(self.num_hidden, self.num_hidden * 4)

        # Concatenate h_V_i to h_E_ij
        h_V_expand = jnp.expand_dims(h_V, axis=-2)
        h_EV = jnp.concatenate([h_V_expand, h_E], axis=-1)

        # Apply transformations
        h_message = W3(self.act(W2(self.act(W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = jnp.sum(h_message, axis=-2) / self.scale

        # Apply dropout and normalization
        dropout1 = hk.dropout(hk.next_rng_key(), self.dropout_rate, dh)
        h_V = h_V + dropout1

        # Position-wise feedforward
        dh = dense(h_V)
        dropout2 = hk.dropout(hk.next_rng_key(), self.dropout_rate, dh)
        h_V = h_V + dropout2

        if mask_V is not None:
            mask_V = mask_V[..., None]
            h_V = mask_V * h_V

        return h_V

class EncLayer(hk.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None):
        super().__init__(name=name)
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout_rate = dropout
        self.act = jax.nn.gelu

    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        W1 = hk.Linear(self.num_hidden, name="W1")
        W2 = hk.Linear(self.num_hidden, name="W2")
        W3 = hk.Linear(self.num_hidden, name="W3")
        dense = PositionWiseFeedForwardJax(self.num_hidden, self.num_hidden * 4)

        # Concatenate h_V_i to h_E_ij
        B, N, _ = h_V.shape
        _, _, K, _ = h_E.shape
        ii = jnp.arange(B)[:, None, None]
        jj = jnp.arange(N)[:, None]
        kk = jnp.arange(K)[None, None, :]
        neighbor_idx_expanded = E_idx[:, :, :, None].repeat(self.num_hidden, axis=3)
        h_V_neighbors = h_V[ii, neighbor_idx_expanded, kk]
        h_EV = jnp.concatenate([h_V_neighbors, h_E], axis=-1)

        # Apply transformations
        h_message = W3(self.act(W2(self.act(W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = jnp.sum(h_message, axis=-2) / self.scale

        # Apply dropout and normalization
        dropout1 = hk.dropout(hk.next_rng_key(), self.dropout_rate, dh)
        h_V = h_V + dropout1

        # Position-wise feedforward
        dh = dense(h_V)
        dropout2 = hk.dropout(hk.next_rng_key(), self.dropout_rate, dh)
        h_V = h_V + dropout2

        if mask_V is not None:
            mask_V = mask_V[..., None]
            h_V = mask_V * h_V

        return h_V

