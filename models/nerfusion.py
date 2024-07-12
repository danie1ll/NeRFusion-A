import numpy as np
import tinycudann as tcnn
import torch
import torch.nn.functional as F
import vren
from einops import rearrange
from kornia.utils.grid import create_meshgrid3d
from torch import nn

from models.fusion.neuralrecon import NeuralRecon

from .custom_functions import TruncExp
from .rendering import NEAR_DISTANCE


class NeRFusion2(nn.Module):
    def __init__(self, scale, grid_size=128, global_representation=None, cfg=None):
        super().__init__()

        # scene bounding box
        # TODO: this is a temp easy solution
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        self.grid_size = grid_size

        self.cascades = 1
        self.register_buffer('density_bitfield',
            torch.ones(self.grid_size**3//8, dtype=torch.uint8)) # dummy
        self.register_buffer('density_grid',
            torch.zeros(self.cascades, self.grid_size**3))
        self.register_buffer('grid_coords',
            create_meshgrid3d(self.grid_size, self.grid_size, self.grid_size, False, dtype=torch.int32).reshape(-1, 3))
        
        self.global_feature_volume = torch.nn.Parameter(
            torch.rand(1, 16, self.grid_size, self.grid_size, self.grid_size)  # Assuming 16 channels for features
        )

        self.mlp = tcnn.Network(
            n_input_dims=3,
            n_output_dims=16,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 2,
            },
        )

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    # "otype": "Grid",
                    # "type": "Dense",
                    # "n_levels": 3,
                    # "n_feature_per_level": 2,
                    # "base_resolution": 128,
                    # "per_level_scale": 2.0,
                    # "interpolation": "Linear",
                "otype": "Grid",
                "type": "Dense",
                "n_levels": 5,
                "n_feature_per_level": 2,
                "base_resolution": 32,
                # "base_resolution": 16,
                "per_level_scale": 2.0,
                "interpolation": "Linear",
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
            },
        )
        # DIRECTION ENCODER
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        # RGB NET 2D ENCODER
        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3/features of GRU) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
            h: (N, C) intermediate feature
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        h = self.xyz_encoder(x) # (N, 16)

        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """

        # x = self.get_global_feature(x) # (N, 16)
        # h = self.mlp(x)
        # sigmas = TruncExp.apply(h[:, 0])

        sigmas, h = self.density(x, return_feat=True)
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        return sigmas, rgbs


    def get_global_feature(self, x):
        """
        Extracts feature for location in global feature volume.

        Args:
            x: (N, 3) xyz coordinates in [-scale, scale]

        Returns:
            features: (N, 16) interpolated feature vectors
        """
        # Normalize coordinates to [0, 1] for grid sample
        x_normalized = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)

        # Prepare the normalized coordinates for grid_sample
        # Grid sample expects input of shape (N, H, W, D, 3) for 3D grid sampling
        x_normalized = x_normalized.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)  # Shape: (1, N, 1, 1, 3)

        # Permutation to make it compatible with grid_sample
        x_normalized = x_normalized.permute(0, 2, 3, 1, 4) # Shape: (1, 1, 1, N, 3)

        # Use grid_sample to interpolate features from global_feature_volume
        features = F.grid_sample(self.global_feature_volume, x_normalized, mode="nearest", align_corners=True)

        # Reshape the features to (N, 16)
        features = features.squeeze().t()  # Shape: (N, 16)

        return features

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def prune_cells(self, K, poses, img_wh, chunk=64 ** 3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a')  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i + chunk] / (self.grid_size - 1) * 2 - 1
                s = min(2 ** (c - 1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (uvd[:, 2] >= 0) & \
                           (uv[:, 0] >= 0) & (uv[:, 0] < img_wh[0]) & \
                           (uv[:, 1] >= 0) & (uv[:, 1] < img_wh[1])
                covered_by_cam = (uvd[:, 2] >= NEAR_DISTANCE) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i + chunk]] = \
                    count = covered_by_cam.sum(0) / N_cams

                too_near_to_cam = (uvd[:, 2] < NEAR_DISTANCE) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                self.density_grid[c, indices[i:i + chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size ** 3 // 4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2 ** (c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords / (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay ** (1 / self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid < 0,
                        self.density_grid,
                        torch.maximum(self.density_grid * decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid > 0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)


