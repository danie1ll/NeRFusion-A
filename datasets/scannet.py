import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset

# because of this factor, normalized frames are slightly larger than original bounding box
# unclear why this is needed
SCANNET_FAR = 2.0


class ScanNetDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.unpad = 24
        self.num_frames_train = kwargs.get('num_frames_train', 800)
        self.num_frames_test = kwargs.get('num_frames_test', 80)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        # K is intrinsic matrix, which contains internal parameters of camera
        # contains focal lengths fx and fy in terms of pixels
        # contains principal points cx and cy, where optical axis intersects image plane
        # contains skew coefficient s, which accounts for the non-orthogonality between the x and y pixel axes. Ideally 0
        # needed to go from 3D point in camera coordinate system to 2D point in image
        K = np.loadtxt(os.path.join(self.root_dir, "./intrinsic/intrinsic_color.txt"))[:3, :3]
        # Scannet depth frames are at a resolution of 640×480 and color frames at a resolution of 1296×968 pixels
        # Scannet paper does not indicate that there is any padding applied, so below unpadding is surprising
        H, W = 968 - 2 * self.unpad, 1296 - 2 * self.unpad
        # adjusts the two principal points cx and cy to the padding
        K[:2, 2] -= self.unpad
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(H, W, self.K)
        # adjusted image width and height
        self.img_wh = (W, H)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train':
            with open(os.path.join(self.root_dir, "train.txt"), 'r') as f:
                frames = f.read().strip().split()
                frames = frames[:self.num_frames_train]
        else:
            with open(os.path.join(self.root_dir, f"{split}.txt"), 'r') as f:
                frames = f.read().strip().split()
                frames = frames[:self.num_frames_test]

        # cam_box contains the two corner points of the rectangular prism that contains all the camera positions
        cam_bbox = np.loadtxt(os.path.join(self.root_dir, f"cam_bbox.txt"))
        # calculates the scaling factor to be applied to all frames by adding the largest dimension of the bounding box and the margin
        # because of 2 * SCANNET_FAR, the normalized space is slightly larger than the actual bounding box
        sbbox_scale = (cam_bbox[1] - cam_bbox[0]).max() + 2 * SCANNET_FAR
        # mean of bounding box is equivalent to the center of the bounding box,
        # center is used to shift all camera positions so they are centered around the origin after normalization
        sbbox_shift = cam_bbox.mean(axis=0)

        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            # camera-to-world matrix contains pose information for specific frames
            # c2w matrix combines rotation and transformation 
            c2w = np.loadtxt(os.path.join(self.root_dir, f"pose/{frame}.txt"))[:3]

            # add shift
            # the translation part of the c2w matrix (last column) is adjusted by subtracting
            # sbbox_shift and dividing by sbbox_scale. This normalization is needed to ensure that
            # camera positions are consistent and fit within a normalized bbox.
            c2w[0, 3] -= sbbox_shift[0]
            c2w[1, 3] -= sbbox_shift[1]
            c2w[2, 3] -= sbbox_shift[2]
            c2w[:, 3] /= sbbox_scale

            # contains transformation matrices for all frames
            self.poses += [c2w]

            try:
                img_path = os.path.join(self.root_dir, f"color/{frame}.jpg")
                img = read_image(img_path, self.img_wh, unpad=self.unpad)
                # contains alpha-blended, unpadded, resized images in dimension ((height * width), channels)
                self.rays += [img]
            except: pass

        if len(self.rays)>0:
            # ?: depends on whether alpha-values for transparency are given.
            # Either (N_images, h*w, channels) or (N_images, h*w, channels, alpha)
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, h*w, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
