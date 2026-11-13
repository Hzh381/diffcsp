"""DiffCSP adapter for the EquiformerV3 denoising backbone.

The equivariant building blocks are kept compatible with the official
EquiformerV3 implementation. This file only adds the crystal-specific
interface used by ``diffcsp.pl_modules.diffusion.CSPDiffusion``.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter

from .edge_rot_mat import init_edge_rot_mat
from .envelope import PolynomialEnvelope
from .input_block import EdgeDegreeEmbedding
from .layer_norm import get_normalization_layer
from .radial_function import GaussianSmearing, RadialFunction
from .so3 import SO3Linear, SO3Rotation
from .transformer_block import TransBlockV3


_OFFSETS = torch.tensor(
    [
        [-1, -1, -1],
        [-1, -1, 0],
        [-1, -1, 1],
        [-1, 0, -1],
        [-1, 0, 0],
        [-1, 0, 1],
        [-1, 1, -1],
        [-1, 1, 0],
        [-1, 1, 1],
        [0, -1, -1],
        [0, -1, 0],
        [0, -1, 1],
        [0, 0, -1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, -1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=torch.float32,
)


def _radius_graph_pbc(cart_coords, lattices, num_atoms, radius, max_neighbors):
    """Build directed neighbor edges with periodic images.

    The topology search is intentionally local to the decoder package so it
    does not pull the full pymatgen preprocessing stack into model import.
    Selection indices are computed without gradients; edge vectors retain
    gradients with respect to coordinates and lattice matrices.
    """
    sources = []
    targets = []
    offsets = []
    start = 0
    offsets_all = _OFFSETS.to(cart_coords.device)

    with torch.no_grad():
        atom_counts = (
            num_atoms.tolist() if hasattr(num_atoms, "tolist") else list(num_atoms)
        )
        for graph_idx, num_atom in enumerate(atom_counts):
            end = start + num_atom
            graph_coords = cart_coords[start:end].detach()
            cell = lattices[graph_idx].detach()
            shifts = offsets_all @ cell

            # [target, source, image, 3]
            vectors = (
                graph_coords[None, :, None, :]
                + shifts[None, None, :, :]
                - graph_coords[:, None, None, :]
            )
            distances_sq = vectors.pow(2).sum(dim=-1)
            valid = distances_sq <= radius**2
            valid &= distances_sq > 1e-8

            for target_idx in range(num_atom):
                candidate_idx = valid[target_idx].nonzero(as_tuple=False)
                candidate_dist = distances_sq[target_idx][valid[target_idx]]
                if candidate_idx.numel() == 0:
                    continue
                keep = min(max_neighbors, candidate_idx.shape[0])
                nearest = torch.topk(
                    candidate_dist,
                    k=keep,
                    largest=False,
                ).indices
                chosen = candidate_idx[nearest]
                src_idx = chosen[:, 0] + start
                image_idx = chosen[:, 1]
                sources.append(src_idx)
                targets.append(
                    torch.full(
                        (keep,),
                        target_idx + start,
                        dtype=torch.long,
                        device=cart_coords.device,
                    )
                )
                offsets.append(offsets_all[image_idx])
            start = end

    if not sources:
        empty_index = torch.empty((2, 0), dtype=torch.long, device=cart_coords.device)
        empty_offsets = torch.empty(
            (0, 3),
            dtype=cart_coords.dtype,
            device=cart_coords.device,
        )
        return empty_index, empty_offsets

    edge_index = torch.stack(
        [torch.cat(sources), torch.cat(targets)],
        dim=0,
    )
    cell_offsets = torch.cat(offsets, dim=0)
    return edge_index, cell_offsets


class EquiformerV3CSPDecoder(nn.Module):
    """Periodic crystal denoiser built from EquiformerV3 blocks.

    Parameters match the decoder interface expected by ``CSPDiffusion``:
    ``time_emb``, atom types, fractional coordinates, lattice matrices,
    atom counts and the node-to-crystal mapping.

    The output is ``(pred_lattice, pred_coord)``, where ``pred_lattice``
    has shape ``(B, 3, 3)`` and ``pred_coord`` has shape ``(N, 3)``.
    """

    def __init__(
        self,
        latent_dim=256,
        hidden_dim=128,
        num_layers=4,
        max_atoms=100,
        lmax=2,
        mmax=2,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        edge_channels=128,
        num_radial_basis=64,
        cutoff=7.0,
        max_neighbors=20,
        avg_degree=20.0,
        attn_grid_resolution_list=(8, 20),
        ffn_grid_resolution_list=(20, 20),
        norm_type="merge_layer_norm",
        attn_activation="sep-merge_gates2_swiglu",
        ffn_activation="sep-merge_gates2_swiglu",
        use_envelope=True,
        use_grid_mlp=True,
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        attn_eps=1e-16,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        **kwargs,
    ):
        super().__init__()

        # Keep the module usable with older Hydra configs that pass decoder
        # fields through the parent CSPDiffusion config.
        del kwargs

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_atoms = max_atoms
        self.lmax = lmax
        self.mmax = mmax
        self.cutoff = float(cutoff)
        self.max_neighbors = max_neighbors
        self.avg_degree = float(avg_degree)
        self.use_envelope = use_envelope

        grid_list = lambda value: list(value) if value is not None else None
        attn_grid_resolution_list = grid_list(attn_grid_resolution_list)
        ffn_grid_resolution_list = grid_list(ffn_grid_resolution_list)

        # EquiformerV3 uses atom-type embeddings on the scalar (L=0) channel.
        self.sphere_embedding = nn.Embedding(max_atoms + 1, hidden_dim)
        self.time_embedding = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(latent_dim, hidden_dim),
            )
            if latent_dim > 0
            else None
        )

        self.distance_expansion = GaussianSmearing(
            0.0,
            self.cutoff,
            num_radial_basis,
            2.0,
        )
        edge_channels_list = [
            int(self.distance_expansion.num_output),
            edge_channels,
            edge_channels,
        ]
        self.envelope_func = (
            PolynomialEnvelope(cutoff=self.cutoff, exponent=5)
            if self.use_envelope
            else None
        )

        self.so3_rotation = SO3Rotation(
            lmax=self.lmax,
            mmax=self.mmax,
            use_rotation_mask=False,
        )
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            num_channels=hidden_dim,
            lmax=self.lmax,
            mmax=self.mmax,
            so3_rotation=self.so3_rotation,
            max_num_elements=max_atoms + 1,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=True,
            rescale_factor=self.avg_degree,
        )

        self.blocks = nn.ModuleList(
            [
                TransBlockV3(
                    num_in_channels=hidden_dim,
                    attn_hidden_channels=attn_hidden_channels,
                    num_heads=num_heads,
                    attn_alpha_channels=attn_alpha_channels,
                    attn_value_channels=attn_value_channels,
                    ffn_hidden_channels=ffn_hidden_channels,
                    num_out_channels=hidden_dim,
                    lmax=self.lmax,
                    mmax=self.mmax,
                    so3_rotation=self.so3_rotation,
                    attn_grid_resolution_list=attn_grid_resolution_list,
                    ffn_grid_resolution_list=ffn_grid_resolution_list,
                    max_num_elements=max_atoms + 1,
                    edge_channels_list=edge_channels_list,
                    use_atom_edge_embedding=True,
                    attn_activation=attn_activation,
                    use_attn_renorm=use_attn_renorm,
                    use_add_merge=use_add_merge,
                    use_rad_l_parametrization=use_rad_l_parametrization,
                    softcap=softcap,
                    attn_eps=attn_eps,
                    ffn_activation=ffn_activation,
                    use_grid_mlp=use_grid_mlp,
                    norm_type=norm_type,
                    drop_path_rate=drop_path_rate,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = get_normalization_layer(
            norm_type,
            lmax=self.lmax,
            num_channels=hidden_dim,
        )

        # A single L=1 vector per atom is sufficient for the coordinate score.
        self.coord_head = SO3Linear(hidden_dim, 1, lmax=1)
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, SO3Linear)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, RadialFunction):
            module.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            std = 1.0 / module.in_features**0.5
            nn.init.uniform_(module.weight, -std, std)

    def _build_edges(self, frac_coords, lattices, num_atoms, node2graph):
        lattice_nodes = lattices[node2graph]
        cart_coords = torch.einsum("ni,nij->nj", frac_coords, lattice_nodes)

        edge_index, cell_offsets = _radius_graph_pbc(
            cart_coords,
            lattices,
            num_atoms,
            self.cutoff,
            self.max_neighbors,
        )

        edge_graph = node2graph[edge_index[1]]
        edge_lattice = lattices[edge_graph]
        edge_vec = cart_coords[edge_index[0]] - cart_coords[edge_index[1]]
        edge_vec = edge_vec + torch.einsum(
            "ei,eij->ej",
            cell_offsets.to(edge_lattice.dtype),
            edge_lattice,
        )
        edge_distance = edge_vec.norm(dim=-1).clamp_min(1e-6)
        return edge_index, edge_distance, edge_vec

    def _heads(self, x, lattices, node2graph):
        x = self.norm(x)

        coord_irreps = self.coord_head(x.narrow(1, 0, 4))
        pred_coord = coord_irreps[:, 1:4, 0]

        graph_features = scatter(
            x[:, 0, :],
            node2graph,
            dim=0,
            dim_size=lattices.shape[0],
            reduce="mean",
        )
        pred_lattice = self.lattice_head(graph_features).view(-1, 3, 3)
        pred_lattice = torch.bmm(pred_lattice, lattices)
        return pred_lattice, pred_coord

    def forward(
        self,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
    ):
        atom_types = atom_types.long()
        num_nodes = atom_types.shape[0]

        x = torch.zeros(
            (num_nodes, (self.lmax + 1) ** 2, self.hidden_dim),
            device=atom_types.device,
            dtype=time_emb.dtype if time_emb is not None else lattices.dtype,
        )
        x[:, 0, :] = self.sphere_embedding(atom_types)
        if self.time_embedding is not None:
            x[:, 0, :] = x[:, 0, :] + self.time_embedding(time_emb)[node2graph]

        edge_index, edge_distance, edge_vec = self._build_edges(
            frac_coords, lattices, num_atoms, node2graph
        )

        # Isolated atoms have no graph edges; returning the scalar graph path
        # keeps the decoder well-defined for unusual boundary cases.
        if edge_index.shape[1] == 0:
            return self._heads(x, lattices, node2graph)

        self.so3_rotation.set_wigner(
            init_edge_rot_mat(edge_vec, use_rotation_mask=False)
        )
        edge_envelope = (
            self.envelope_func(edge_distance)
            if self.envelope_func is not None
            else None
        )
        edge_distance = self.distance_expansion(edge_distance)

        x = x + self.edge_degree_embedding(
            atom_types,
            edge_distance,
            edge_index,
            edge_envelope,
        )
        source_types = atom_types[edge_index[0]]
        target_types = atom_types[edge_index[1]]

        for block in self.blocks:
            x = block(
                x,
                source_types,
                target_types,
                edge_distance,
                edge_index,
                edge_envelope,
                node2graph,
            )

        return self._heads(x, lattices, node2graph)
