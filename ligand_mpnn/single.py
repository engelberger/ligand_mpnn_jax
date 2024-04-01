import jax.numpy as jnp
from jax import lax
import haiku as hk
import jax
import jax.numpy as jnp
import functools
import itertools

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
    def __init__(
        self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None
    ):
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
        d = jnp.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        )
        d = d * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        # One-hot encode the clipped and masked offsets
        d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        # Pass the one-hot encoded offsets through the linear layer
        E = self.linear(d_onehot.astype(jnp.float32))
        return E


class DecLayer(hk.Module):
    def __init__(
        self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None
    ):
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
    def __init__(
        self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, name=None
    ):
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


class ProteinFeaturesLigand(hk.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        atom_context_num=16,
        use_side_chains=False,
        name=None,
    ):
        super().__init__(name=name)
        self.use_side_chains = use_side_chains
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.atom_context_num = atom_context_num

        self.embeddings = PositionalEncodingsJax(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = hk.Linear(edge_features, name="edge_embedding")
        self.node_project_down = hk.Linear(
            5 * num_rbf + 64 + 4, node_features, name="node_project_down"
        )
        self.type_linear = hk.Linear(147, 64, name="type_linear")
        self.y_nodes = hk.Linear(147, node_features, name="y_nodes")
        self.y_edges = hk.Linear(num_rbf, node_features, name="y_edges")

        self.side_chain_atom_types = jnp.array(
            [
                6,
                6,
                6,
                8,
                8,
                16,
                6,
                6,
                6,
                7,
                7,
                8,
                8,
                16,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                8,
                8,
                6,
                7,
                7,
                8,
                6,
                6,
                6,
                7,
                8,
            ]
        )

        self.periodic_table_features = jnp.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    84,
                    85,
                    86,
                    87,
                    88,
                    89,
                    90,
                    91,
                    92,
                    93,
                    94,
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                ],
                [
                    0,
                    1,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                ],
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ],
            ],
            dtype=jnp.int32,
        )

    def _make_angle_features(self, A, B, C, Y):
        v1 = A - B
        v2 = C - B
        e1 = v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True)
        e1_v2_dot = jnp.einsum("bli,bli->bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot
        e2 = u2 / jnp.linalg.norm(u2, axis=-1, keepdims=True)
        e3 = jnp.cross(e1, e2, axis=-1)
        R_residue = jnp.concatenate(
            (e1[..., None], e2[..., None], e3[..., None]), axis=-1
        )

        local_vectors = jnp.einsum("blqp,blyq->blyp", R_residue, Y - B[:, :, None, :])

        rxy = jnp.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = jnp.linalg.norm(local_vectors, axis=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = jnp.concatenate(
            [f1[..., None], f2[..., None], f3[..., None], f4[..., None]], axis=-1
        )
        return f

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask[:, None, :] * mask[:, :, None]
        dX = X[:, None, :, :] - X[:, :, None, :]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=3) + eps)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = jax.lax.top_k(
            D_adjust, jnp.minimum(self.top_k, X.shape[1]), axis=-1
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu[None, None, None, :]
        D_sigma = (D_max - D_min) / D_count
        D_expand = D[..., None]
        RBF = jnp.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = jnp.sqrt(
            jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, axis=-1) + 1e-6
        )
        D_A_B_neighbors = gather_edges_jax(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(self, input_features):
        Y = input_features["Y"]
        Y_m = input_features["Y_m"]
        Y_t = input_features["Y_t"]
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * jax.random.normal(hk.next_rng_key(), X.shape)
            Y = Y + self.augment_eps * jax.random.normal(hk.next_rng_key(), Y.shape)

        B, L, _, _ = X.shape

        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        b = Ca - N
        c = C - Ca
        a = jnp.cross(b, c, axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = jnp.concatenate(RBF_all, axis=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges_jax(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains = (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        E_chains = gather_edges_jax(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.astype(jnp.int32), E_chains)
        E = jnp.concatenate((E_positional, RBF_all), axis=-1)
        E = self.edge_embedding(E)

        if self.use_side_chains:
            xyz_37 = input_features["xyz_37"]
            xyz_37_m = input_features["xyz_37_m"]
            E_idx_sub = E_idx[:, :, :16]
            mask_residues = input_features["chain_mask"]
            xyz_37_m = xyz_37_m * (1 - mask_residues[:, :, None])
            R_m = gather_nodes_jax(xyz_37_m[:, :, 5:], E_idx_sub)

            X_sidechain = xyz_37[:, :, 5:, :].reshape(B, L, -1)
            R = gather_nodes_jax(X_sidechain, E_idx_sub).reshape(
                B, L, E_idx_sub.shape[2], -1, 3
            )
            R_t = self.side_chain_atom_types[None, None, None, :].repeat(
                B, L, E_idx_sub.shape[2], 0
            )

            R = R.reshape(B, L, -1, 3)
            R_m = R_m.reshape(B, L, -1)
            R_t = R_t.reshape(B, L, -1)

            Y = jnp.concatenate((R, Y), axis=2)
            Y_m = jnp.concatenate((R_m, Y_m), axis=2)
            Y_t = jnp.concatenate((R_t, Y_t), axis=2)

            Cb_Y_distances = jnp.sum((Cb[:, :, None, :] - Y) ** 2, axis=-1)
            mask_Y = mask[:, :, None] * Y_m
            Cb_Y_distances_adjusted = Cb_Y_distances * mask_Y + (1.0 - mask_Y) * 10000.0
            _, E_idx_Y = jax.lax.top_k(
                Cb_Y_distances_adjusted, self.atom_context_num, axis=-1
            )

            Y = jnp.take_along_axis(Y, E_idx_Y[..., None].repeat(3, axis=-1), axis=2)
            Y_t = jnp.take_along_axis(Y_t, E_idx_Y, axis=2)
            Y_m = jnp.take_along_axis(Y_m, E_idx_Y, axis=2)

        Y_t = Y_t.astype(jnp.int32)
        Y_t_g = self.periodic_table_features[1, Y_t]
        Y_t_p = self.periodic_table_features[2, Y_t]

        Y_t_g_1hot_ = jax.nn.one_hot(Y_t_g, 19)
        Y_t_p_1hot_ = jax.nn.one_hot(Y_t_p, 8)
        Y_t_1hot_ = jax.nn.one_hot(Y_t, 120)

        Y_t_1hot_ = jnp.concatenate([Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], axis=-1)
        Y_t_1hot = self.type_linear(Y_t_1hot_.astype(jnp.float32))

        D_N_Y = self._rbf(
            jnp.sqrt(jnp.sum((N[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6)
        )
        D_Ca_Y = self._rbf(
            jnp.sqrt(jnp.sum((Ca[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6)
        )
        D_C_Y = self._rbf(
            jnp.sqrt(jnp.sum((C[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6)
        )
        D_O_Y = self._rbf(
            jnp.sqrt(jnp.sum((O[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6)
        )
        D_Cb_Y = self._rbf(
            jnp.sqrt(jnp.sum((Cb[:, :, None, :] - Y) ** 2, axis=-1) + 1e-6)
        )

        f_angles = self._make_angle_features(N, Ca, C, Y)

        D_all = jnp.concatenate(
            (D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot, f_angles), axis=-1
        )
        V = self.node_project_down(D_all)

        Y_edges = self._rbf(
            jnp.sqrt(
                jnp.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, axis=-1)
                + 1e-6
            )
        )

        Y_edges = self.y_edges(Y_edges)
        Y_nodes = self.y_nodes(Y_t_1hot_.astype(jnp.float32))

        return V, E, E_idx, Y_nodes, Y_edges, Y_m


class ProteinFeatures(hk.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        top_k=48,
        augment_eps=0.0,
        num_rbf=16,
        num_positional_embeddings=16,
        name=None,
    ):
        super().__init__(name=name)
        self.node_features = node_features
        self.edge_features = edge_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodingsJax(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = hk.Linear(edge_features, name="edge_embedding")

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask[:, None, :] * mask[:, :, None]
        dX = X[:, None, :, :] - X[:, :, None, :]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=3) + eps)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = jax.lax.top_k(
            D_adjust, jnp.minimum(self.top_k, X.shape[1]), axis=-1
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu[None, None, None, :]
        D_sigma = (D_max - D_min) / D_count
        D_expand = D[..., None]
        RBF = jnp.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = jnp.sqrt(
            jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, axis=-1) + 1e-6
        )
        D_A_B_neighbors = gather_edges_jax(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(self, input_features):
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * jax.random.normal(hk.next_rng_key(), X.shape)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = jnp.cross(b, c, axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = jnp.concatenate(tuple(RBF_all), axis=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges_jax(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains = (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        E_chains = gather_edges_jax(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.astype(jnp.int32), E_chains)
        E = jnp.concatenate((E_positional, RBF_all), axis=-1)
        E = self.edge_embedding(E)

        return E, E_idx


class ProteinFeaturesMembrane(hk.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        top_k=48,
        augment_eps=0.0,
        num_rbf=16,
        num_positional_embeddings=16,
        num_classes=3,
        name=None,
    ):
        super().__init__(name=name)
        self.node_features = node_features
        self.edge_features = edge_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.num_classes = num_classes

        self.embeddings = PositionalEncodingsJax(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = hk.Linear(edge_features, name="edge_embedding")
        self.node_embedding = hk.Linear(node_features, name="node_embedding")

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask[:, None, :] * mask[:, :, None]
        dX = X[:, None, :, :] - X[:, :, None, :]
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, axis=3) + eps)
        D_max = jnp.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = jax.lax.top_k(
            D_adjust, jnp.minimum(self.top_k, X.shape[1]), axis=-1
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu[None, None, None, :]
        D_sigma = (D_max - D_min) / D_count
        D_expand = D[..., None]
        RBF = jnp.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = jnp.sqrt(
            jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, axis=-1) + 1e-6
        )
        D_A_B_neighbors = gather_edges_jax(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(self, input_features):
        X = input_features["X"]
        mask = input_features["mask"]
        R_idx = input_features["R_idx"]
        chain_labels = input_features["chain_labels"]
        membrane_per_residue_labels = input_features["membrane_per_residue_labels"]

        if self.augment_eps > 0:
            X = X + self.augment_eps * jax.random.normal(hk.next_rng_key(), X.shape)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = jnp.cross(b, c, axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = jnp.concatenate(tuple(RBF_all), axis=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges_jax(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains = (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        E_chains = gather_edges_jax(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.astype(jnp.int32), E_chains)
        E = jnp.concatenate((E_positional, RBF_all), axis=-1)
        E = self.edge_embedding(E)

        C_1hot = jax.nn.one_hot(membrane_per_residue_labels, self.num_classes)
        V = self.node_embedding(C_1hot)

        return V, E, E_idx


class ProteinMPNN(hk.Module):
    def __init__(
        self,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.0,
        atom_context_num=0,
        model_type="protein_mpnn",
        ligand_mpnn_use_side_chain_context=False,
        name=None,
    ):
        super().__init__(name=name)
        self.model_type = model_type
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        if self.model_type == "ligand_mpnn":
            self.features = ProteinFeaturesLigand(
                node_features,
                edge_features,
                top_k=k_neighbors,
                augment_eps=augment_eps,
                atom_context_num=atom_context_num,
                use_side_chains=ligand_mpnn_use_side_chain_context,
            )
            self.W_v = hk.Linear(hidden_dim, name="W_v")
            self.W_c = hk.Linear(hidden_dim, name="W_c")
            self.W_nodes_y = hk.Linear(hidden_dim, name="W_nodes_y")
            self.W_edges_y = hk.Linear(hidden_dim, name="W_edges_y")
            self.V_C = hk.Linear(hidden_dim, name="V_C")
            self.context_encoder_layers = [
                DecLayerJ(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(2)
            ]
            self.y_context_encoder_layers = [
                DecLayerJ(hidden_dim, hidden_dim, dropout=dropout) for _ in range(2)
            ]
        elif self.model_type == "protein_mpnn" or self.model_type == "soluble_mpnn":
            self.features = ProteinFeatures(
                node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
            )
        elif (
            self.model_type == "per_residue_label_membrane_mpnn"
            or self.model_type == "global_label_membrane_mpnn"
        ):
            self.W_v = hk.Linear(hidden_dim, name="W_v")
            self.features = ProteinFeaturesMembrane(
                node_features,
                edge_features,
                top_k=k_neighbors,
                augment_eps=augment_eps,
                num_classes=3,
            )
        else:
            raise ValueError("Choose --model_type flag from currently available models")

        self.W_e = hk.Linear(hidden_dim, name="W_e")
        self.W_s = hk.Embed(vocab, hidden_dim, name="W_s")
        self.dropout_rate = dropout

        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ]
        self.decoder_layers = [
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ]
        self.W_out = hk.Linear(num_letters, name="W_out")

    def encode(self, feature_dict):
        B, L = feature_dict["S"].shape
        device = feature_dict["S"].device

        if self.model_type == "ligand_mpnn":
            V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(feature_dict)
            h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
            h_E = self.W_e(E)
            h_E_context = self.W_v(V)

            mask_attend = gather_nodes_jax(
                feature_dict["mask"][:, :, None], E_idx
            ).squeeze(-1)
            mask_attend = feature_dict["mask"][:, :, None] * mask_attend
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, feature_dict["mask"], mask_attend)

            h_V_C = self.W_c(h_V)
            Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
            Y_nodes = self.W_nodes_y(Y_nodes)
            Y_edges = self.W_edges_y(Y_edges)
            for i in range(len(self.context_encoder_layers)):
                Y_nodes = self.y_context_encoder_layers[i](
                    Y_nodes, Y_edges, Y_m, Y_m_edges
                )
                h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
                h_V_C = self.context_encoder_layers[i](
                    h_V_C, h_E_context_cat, feature_dict["mask"], Y_m
                )

            h_V_C = self.V_C(h_V_C)
            h_V = h_V + hk.dropout(hk.next_rng_key(), self.dropout_rate, h_V_C)
        elif self.model_type == "protein_mpnn" or self.model_type == "soluble_mpnn":
            E, E_idx = self.features(feature_dict)
            h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
            h_E = self.W_e(E)

            mask_attend = gather_nodes_jax(
                feature_dict["mask"][:, :, None], E_idx
            ).squeeze(-1)
            mask_attend = feature_dict["mask"][:, :, None] * mask_attend
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, feature_dict["mask"], mask_attend)
        elif (
            self.model_type == "per_residue_label_membrane_mpnn"
            or self.model_type == "global_label_membrane_mpnn"
        ):
            V, E, E_idx = self.features(feature_dict)
            h_V = self.W_v(V)
            h_E = self.W_e(E)

            mask_attend = gather_nodes_jax(
                feature_dict["mask"][:, :, None], E_idx
            ).squeeze(-1)
            mask_attend = feature_dict["mask"][:, :, None] * mask_attend
            for layer in self.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, feature_dict["mask"], mask_attend)

        return h_V, h_E, E_idx

    def sample(self, feature_dict):
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict["S"]
        mask = feature_dict["mask"]
        chain_mask = feature_dict["chain_mask"]
        randn = feature_dict["randn"]
        B, L = S_true.shape
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        chain_mask = mask * chain_mask
        decoding_order = jnp.argsort((chain_mask + 0.0001) * (jnp.abs(randn)))

        if (
            len(feature_dict["symmetry_residues"][0]) == 0
            and len(feature_dict["symmetry_residues"]) == 1
        ):
            E_idx = jnp.repeat(E_idx, B_decoder, axis=0)
            permutation_matrix_reverse = jax.nn.one_hot(decoding_order, num_classes=L)
            order_mask_backward = jnp.einsum(
                "ij,biq,bjp->bqp",
                (1 - jnp.triu(jnp.ones((L, L)))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = gather_edges_jax(order_mask_backward[:, :, :, None], E_idx)[
                :, :, :, 0
            ]
            mask_1D = mask[:, :, None, None]
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            S_true = jnp.repeat(S_true, B_decoder, axis=0)
            h_V = jnp.repeat(h_V, B_decoder, axis=0)
            h_E = jnp.repeat(h_E, B_decoder, axis=0)
            chain_mask = jnp.repeat(chain_mask, B_decoder, axis=0)
            mask = jnp.repeat(mask, B_decoder, axis=0)
            feature_dict["bias"] = jnp.repeat(feature_dict["bias"], B_decoder, axis=0)

            all_probs = jnp.zeros((B_decoder, L, 20))
            all_log_probs = jnp.zeros((B_decoder, L, 21))
            h_S = jnp.zeros_like(h_V)
            S = 20 * jnp.ones((B_decoder, L), dtype=jnp.int32)
            h_V_stack = [h_V] + [
                jnp.zeros_like(h_V) for _ in range(len(self.decoder_layers))
            ]

            h_EX_encoder = cat_neighbors_nodes_jax(jnp.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes_jax(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for t_ in range(L):
                t = decoding_order[:, t_]
                chain_mask_t = jnp.take_along_axis(chain_mask, t[:, None], axis=1)[:, 0]
                mask_t = jnp.take_along_axis(mask, t[:, None], axis=1)[:, 0]
                bias_t = jnp.take_along_axis(
                    feature_dict["bias"], t[:, None, None], axis=1
                )[:, 0, :]

                E_idx_t = jnp.take_along_axis(E_idx, t[:, None, None], axis=1)
                h_E_t = jnp.take_along_axis(h_E, t[:, None, None, None], axis=1)
                h_ES_t = cat_neighbors_nodes_jax(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = jnp.take_along_axis(
                    h_EXV_encoder_fw, t[:, None, None, None], axis=1
                )

                mask_bw_t = jnp.take_along_axis(mask_bw, t[:, None, None, None], axis=1)

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes_jax(
                        h_V_stack[l], h_ES_t, E_idx_t
                    )
                    h_V_t = jnp.take_along_axis(h_V_stack[l], t[:, None, None], axis=1)
                    h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l + 1] = jax.ops.index_update(
                        h_V_stack[l + 1],
                        jax.ops.index[t[:, None, None]],
                        layer(h_V_t, h_ESV_t, mask_V=mask_t[:, None]),
                    )

                h_V_t = jnp.take_along_axis(h_V_stack[-1], t[:, None, None], axis=1)[
                    :, 0
                ]
                logits = self.W_out(h_V_t)
                log_probs = jax.nn.log_softmax(logits, axis=-1)

                probs = jax.nn.softmax(
                    (logits + bias_t) / feature_dict["temperature"], axis=-1
                )
                probs_sample = probs[:, :20] / jnp.sum(
                    probs[:, :20], axis=-1, keepdims=True
                )
                S_t = jax.random.categorical(hk.next_rng_key(), probs_sample, axis=-1)

                all_probs = jax.ops.index_update(
                    all_probs,
                    jax.ops.index[t[:, None, None]],
                    (chain_mask_t[:, None, None] * probs_sample[:, None, :]),
                )
                all_log_probs = jax.ops.index_update(
                    all_log_probs,
                    jax.ops.index[t[:, None, None]],
                    (chain_mask_t[:, None, None] * log_probs[:, None, :]),
                )
                S_true_t = jnp.take_along_axis(S_true, t[:, None], axis=1)[:, 0]
                S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).astype(
                    jnp.int32
                )
                h_S = jax.ops.index_update(
                    h_S, jax.ops.index[t[:, None, None]], self.W_s(S_t)[:, None, :]
                )
                S = jax.ops.index_update(S, jax.ops.index[t[:, None]], S_t[:, None])

            output_dict = {
                "S": S,
                "sampling_probs": all_probs,
                "log_probs": all_log_probs,
                "decoding_order": decoding_order,
            }
        else:
            symmetry_weights = jnp.ones(L)
            for i1, item_list in enumerate(feature_dict["symmetry_residues"]):
                for i2, item in enumerate(item_list):
                    symmetry_weights = jax.ops.index_update(
                        symmetry_weights,
                        jax.ops.index[item],
                        feature_dict["symmetry_weights"][i1][i2],
                    )

            new_decoding_order = []
            for t_dec in list(decoding_order[0,].data.numpy()):
                if t_dec not in list(itertools.chain(*new_decoding_order)):
                    list_a = [
                        item
                        for item in feature_dict["symmetry_residues"]
                        if t_dec in item
                    ]
                    if list_a:
                        new_decoding_order.append(list_a[0])
                    else:
                        new_decoding_order.append([t_dec])

            decoding_order = jnp.array(list(itertools.chain(*new_decoding_order)))[
                None,
            ].repeat(B, axis=0)

            permutation_matrix_reverse = jax.nn.one_hot(decoding_order, num_classes=L)
            order_mask_backward = jnp.einsum(
                "ij,biq,bjp->bqp",
                (1 - jnp.triu(jnp.ones((L, L)))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = gather_edges_jax(order_mask_backward[:, :, :, None], E_idx)[
                :, :, :, 0
            ]
            mask_1D = mask[:, :, None, None]
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            E_idx = jnp.repeat(E_idx, B_decoder, axis=0)
            mask_fw = jnp.repeat(mask_fw, B_decoder, axis=0)
            mask_bw = jnp.repeat(mask_bw, B_decoder, axis=0)
            decoding_order = jnp.repeat(decoding_order, B_decoder, axis=0)

            S_true = jnp.repeat(S_true, B_decoder, axis=0)
            h_V = jnp.repeat(h_V, B_decoder, axis=0)
            h_E = jnp.repeat(h_E, B_decoder, axis=0)
            mask = jnp.repeat(mask, B_decoder, axis=0)

            h_S = self.W_s(S_true)
            h_ES = cat_neighbors_nodes_jax(h_S, h_E, E_idx)

            h_EX_encoder = cat_neighbors_nodes_jax(jnp.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes_jax(h_V, h_EX_encoder, E_idx)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            if not use_sequence:
                for layer in self.decoder_layers:
                    h_V = layer(h_V, h_EXV_encoder_fw, mask)
            else:
                for layer in self.decoder_layers:
                    h_ESV = cat_neighbors_nodes_jax(h_V, h_ES, E_idx)
                    h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                    h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            output_dict = {
                "S": S_true,
                "log_probs": log_probs,
                "logits": logits,
                "decoding_order": decoding_order,
            }
        return output_dict

    def single_aa_score(self, feature_dict, use_sequence: bool):
        B_decoder = feature_dict["batch_size"]
        S_true_enc = feature_dict["S"]
        mask_enc = feature_dict["mask"]
        chain_mask_enc = feature_dict["chain_mask"]
        randn = feature_dict["randn"]
        B, L = S_true_enc.shape
        device = S_true_enc.device

        h_V_enc, h_E_enc, E_idx_enc = self.encode(feature_dict)
        log_probs_out = jnp.zeros((B_decoder, L, 21))
        logits_out = jnp.zeros((B_decoder, L, 21))
        decoding_order_out = jnp.zeros((B_decoder, L, L))

        for idx in range(L):
            h_V = jnp.array(h_V_enc)
            E_idx = jnp.array(E_idx_enc)
            mask = jnp.array(mask_enc)
            S_true = jnp.array(S_true_enc)
            if not use_sequence:
                order_mask = jnp.zeros(chain_mask_enc.shape[1])
                order_mask = jax.ops.index_update(order_mask, jax.ops.index[idx], 1.0)
            else:
                order_mask = jnp.ones(chain_mask_enc.shape[1])
                order_mask = jax.ops.index_update(order_mask, jax.ops.index[idx], 0.0)
            decoding_order = jnp.argsort((order_mask + 0.0001) * (jnp.abs(randn)))
            E_idx = jnp.repeat(E_idx, B_decoder, axis=0)
            permutation_matrix_reverse = jax.nn.one_hot(decoding_order, num_classes=L)
            order_mask_backward = jnp.einsum(
                "ij,biq,bjp->bqp",
                (1 - jnp.triu(jnp.ones((L, L)))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = gather_edges_jax(order_mask_backward[:, :, :, None], E_idx)[
                :, :, :, 0
            ]
            mask_1D = mask[:, :, None, None]
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)
            S_true = jnp.repeat(S_true, B_decoder, axis=0)
            h_V = jnp.repeat(h_V, B_decoder, axis=0)
            h_E = jnp.repeat(h_E_enc, B_decoder, axis=0)
            mask = jnp.repeat(mask, B_decoder, axis=0)

            h_S = self.W_s(S_true)
            h_ES = cat_neighbors_nodes_jax(h_S, h_E, E_idx)

            h_EX_encoder = cat_neighbors_nodes_jax(jnp.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes_jax(h_V, h_EX_encoder, E_idx)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes_jax(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            log_probs_out = jax.ops.index_update(
                log_probs_out, jax.ops.index[:, idx, :], log_probs[:, idx, :]
            )
            logits_out = jax.ops.index_update(
                logits_out, jax.ops.index[:, idx, :], logits[:, idx, :]
            )
            decoding_order_out = jax.ops.index_update(
                decoding_order_out, jax.ops.index[:, idx, :], decoding_order
            )

        output_dict = {
            "S": S_true,
            "log_probs": log_probs_out,
            "logits": logits_out,
            "decoding_order": decoding_order_out,
        }
        return output_dict

    def score(self, feature_dict, use_sequence: bool):
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict["S"]
        mask = feature_dict["mask"]
        chain_mask = feature_dict["chain_mask"]
        randn = feature_dict["randn"]
        symmetry_list_of_lists = feature_dict["symmetry_residues"]
        B, L = S_true.shape
        device = S_true.device

        h_V, h_E, E_idx = self.encode(feature_dict)

        chain_mask = mask * chain_mask
        decoding_order = jnp.argsort((chain_mask + 0.0001) * (jnp.abs(randn)))
        if len(symmetry_list_of_lists[0]) == 0 and len(symmetry_list_of_lists) == 1:
            E_idx = jnp.repeat(E_idx, B_decoder, axis=0)
            permutation_matrix_reverse = jax.nn.one_hot(decoding_order, num_classes=L)
            order_mask_backward = jnp.einsum(
                "ij,biq,bjp->bqp",
                (1 - jnp.triu(jnp.ones((L, L)))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = gather_edges_jax(order_mask_backward[:, :, :, None], E_idx)[
                :, :, :, 0
            ]
            mask_1D = mask[:, :, None, None]
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)
        else:
            new_decoding_order = []
            for t_dec in list(decoding_order[0,].data.numpy()):
                if t_dec not in list(itertools.chain(*new_decoding_order)):
                    list_a = [item for item in symmetry_list_of_lists if t_dec in item]
                    if list_a:
                        new_decoding_order.append(list_a[0])
                    else:
                        new_decoding_order.append([t_dec])

            decoding_order = jnp.array(list(itertools.chain(*new_decoding_order)))[
                None,
            ].repeat(B, axis=0)

            permutation_matrix_reverse = jax.nn.one_hot(decoding_order, num_classes=L)
            order_mask_backward = jnp.einsum(
                "ij,biq,bjp->bqp",
                (1 - jnp.triu(jnp.ones((L, L)))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = gather_edges_jax(order_mask_backward[:, :, :, None], E_idx)[
                :, :, :, 0
            ]
            mask_1D = mask[:, :, None, None]
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            E_idx = jnp.repeat(E_idx, B_decoder, axis=0)
            mask_fw = jnp.repeat(mask_fw, B_decoder, axis=0)
            mask_bw = jnp.repeat(mask_bw, B_decoder, axis=0)
            decoding_order = jnp.repeat(decoding_order, B_decoder, axis=0)

        S_true = jnp.repeat(S_true, B_decoder, axis=0)
        h_V = jnp.repeat(h_V, B_decoder, axis=0)
        h_E = jnp.repeat(h_E, B_decoder, axis=0)
        mask = jnp.repeat(mask, B_decoder, axis=0)

        h_S = self.W_s(S_true)
        h_ES = cat_neighbors_nodes_jax(h_S, h_E, E_idx)

        h_EX_encoder = cat_neighbors_nodes_jax(jnp.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes_jax(h_V, h_EX_encoder, E_idx)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        if not use_sequence:
            for layer in self.decoder_layers:
                h_V = layer(h_V, h_EXV_encoder_fw, mask)
        else:
            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes_jax(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        output_dict = {
            "S": S_true,
            "log_probs": log_probs,
            "logits": logits,
            "decoding_order": decoding_order,
        }
        return output_dict
