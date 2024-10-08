import torch
from torch import nn
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """

    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
         ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3, depth_loss_w=0):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.depth_loss_w = depth_loss_w
        self.distortion_loss = DistortionLoss.apply

    def forward(self, results, target):
        # Add small epsilon to avoid log(0) when opacity is 0
        opacity = results['opacity'] + 1e-10

        loss_dict = dict(
            rgb=(results['rgb'] - target['rgb']) ** 2,
            opacity=self.lambda_opacity * (-opacity * torch.log(opacity))
        )

        if 'depth' in target:
            loss_dict['depth'] = self.depth_loss_w * (results['depth'] - target['depth']) ** 2

        # Add distortion loss
        if all(k in results for k in ['ws', 'deltas', 'ts', 'rays_a']):
            distortion = self.distortion_loss(results['ws'], results['deltas'], results['ts'], results['rays_a'])
            loss_dict['distortion'] = self.lambda_distortion * distortion

        return loss_dict
