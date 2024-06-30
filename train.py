import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
import argparse

# wandb
import wandb
from pytorch_lightning.loggers import WandbLogger

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from models.nerfusion import NeRFusion2
from models.rendering import render, MAX_SAMPLES

# gru fusion
from representations.grufusion.neucon_network import NeuConNet
from representations.grufusion.backbone import MnasMulti
from representations.grufusion.gru_fusion import GRUFusion
from utils import tocuda

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        self.model = NeRFusion2(scale=self.hparams.scale)
        
        # comments are default values
        cfg = {
            "MODEL": {
                "THRESHOLDS": [0, 0, 0],
                "VOXEL_SIZE": 0.04,
                "N_VOX": [96, 96, 96],  # Default: [128, 224, 192],     Train/Test: [96, 96, 96]
                "N_LAYER": 3,
                "TRAIN_NUM_SAMPLE": [4096, 16384, 65536],
                "TEST_NUM_SAMPLE": [4096, 16384, 65536],
                "PIXEL_MEAN": [103.53, 116.28, 123.675],
                "PIXEL_STD": [1.0, 1.0, 1.0],
                "FUSION": {"FUSION_ON": True, "FULL": True},  # Default: False,               Train/Test: 'True'
                "BACKBONE2D": {"ARC": "fpn-mnas-1"},
                "SPARSEREG":{"DROPOUT": False},
            }
        }

        self.cfg_gru = Config(cfg)
        self.pixel_mean = torch.Tensor(self.cfg_gru.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(self.cfg_gru.MODEL.PIXEL_STD).view(-1, 1, 1)

        alpha = float(self.cfg_gru.MODEL.BACKBONE2D.ARC.split("-")[-1])
        self.backbone2d = MnasMulti(alpha)
        self.neucon_net = NeuConNet(self.cfg_gru.MODEL)
        self.fuse_to_global = GRUFusion(cfg=self.cfg_gru.MODEL, direct_substitute=True)

        # Add a list to store validation images
        self.val_images = []

    def normalizer(self, x):
        """Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)
    
    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256

        outputs = {}
        W, H = self.train_dataset.img_wh  # (W, H) (1248, 920)
        rays = self.train_dataset.rays # (N_images, H*W, channels) (800, 1148160, 3)
        rgb_gt = batch["rgb"] # (batch size, channels) (8192, 3)

        # Construction of (batch size, N_images, C, H, W) tensor
        reshaped_images = rearrange(rays, "n (h w) c -> n c h w", h=H, w=W)  # torch.Size([800, 3, 920, 1248])
        # imgs = imgs.unsqueeze(0)
        reshaped_images = reshaped_images.unsqueeze(0)
        reshaped_images = reshaped_images.expand(8192, -1, -1, -1, -1)
        reshaped_images = tocuda(reshaped_images)
        imgs = torch.unbind(reshaped_images, 0)

        # image feature extraction
        # in: images; out: feature maps
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        print("WE ARE HERE")
        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)


        


        # --- GRU FUSION ---
        # channels = [96, 48, 24]
        # if self.cfg_gru.FUSION.FUSION_ON:
        #     self.gru_fusion = GRUFusion(self.cfg_gru, channels)
        # for i in range(self.cfg_gru.N_LAYER):

        #     if self.cfg_gru.FUSION.FUSION_ON:
        #         up_coords, feat, tsdf_target, occ_target = self.gru_fusion(up_coords, feat, inputs, i) # def forward part
        #         if self.cfg_gru.FUSION.FULL:
        #             grid_mask = torch.ones_like(feat[:, 0]).bool()

        #     tsdf = self.tsdf_preds[i](feat)
        #     occ = self.occ_preds[i](feat)


            #########################

            # rgb_gt = batch["rgb"]
            # inputs = rgb_gt
            # imgs = torch.unbind(rgb_gt, 1)

            # # image feature extraction
            # # in: images; out: feature maps
            # features = [self.backbone2d(self.normalizer(img)) for img in imgs]

            # # coarse-to-fine decoder: SparseConv and GRU Fusion.
            # # in: image feature; out: sparse coords and tsdf
            # outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        # --- GRU FUSION ---

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'num_frames_train': self.hparams.num_frames_train,
                  'num_frames_test': self.hparams.num_frames_test}
        
        if self.hparams.dataset_name == 'google_scanned':
            
            self.hparams['num_source_views'] = 3
            self.hparams['rectify_inplane_rotation'] = True
            print(self.hparams)
            self.train_dataset = dataset(split=self.hparams.split, args=self.hparams, **kwargs)
            self.test_dataset = dataset(split='test', args=self.hparams, **kwargs)
        else:
            self.train_dataset = dataset(split=self.hparams.split, **kwargs)
            self.test_dataset = dataset(split='test', **kwargs)

        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

    def configure_optimizers(self):
        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.prune_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            
            # Save the image for wandb logging
            self.val_images.append((idx, rgb_pred))
            
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)
        
        # Log images to wandb
        if self.logger.__class__.__name__ == 'WandbLogger':
            wandb_logger = self.logger.experiment
            test_images = []
            for idx, img in self.val_images:
                test_images.append(wandb.Image(img, caption=f"Test image {idx}"))
            wandb_logger.log({"test_images": test_images})
        
        # Clear the validation images list
        self.val_images.clear()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()

    if not hparams.debug:
        wandb.init(project="Nerfusion", name=hparams.exp_name)
        
        if hparams.use_sweep:
            # Override hyperparameters with wandb sweep config
            wandb_config = wandb.config
            for key, value in wandb_config.items():
                if hasattr(hparams, key):
                    setattr(hparams, key, value)
    
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    
    system = NeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=False,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    if not hparams.debug:
        # Get the wandb API key
        WANDB_API_KEY = os.getenv('WANDB_API_KEY')
        wandb.login(host="https://api.wandb.ai", key=WANDB_API_KEY)

        # Initialize wandb logger
        logger = WandbLogger(project="Nerfusion", name=hparams.exp_name, log_model=True)
    else:
        # Use TensorBoardLogger as default
        logger = TensorBoardLogger("tb_logs", name=hparams.exp_name)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only:  # save slimmed ckpt for the last epoch
        ckpt_ = slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                          save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if not hparams.no_save_test and hparams.save_video:
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        rgb_video_path = os.path.join(system.val_dir, 'rgb.mp4')
        depth_video_path = os.path.join(system.val_dir, 'depth.mp4')

        imageio.mimsave(rgb_video_path,
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(depth_video_path,
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)

        if not hparams.debug:
            # Log videos to wandb
            wandb.log({
                "rgb_video": wandb.Video(rgb_video_path, fps=30, format="mp4"),
                "depth_video": wandb.Video(depth_video_path, fps=30, format="mp4")
            })

    if not hparams.debug:
        wandb.finish()
