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


class ScanNetPPDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.transforms_undistorted_path = os.path.join(self.root_dir, "./dslr/nerfstudio/transforms_undistorted.json")
        self.unpad = 24

        self.read_intrinsics()
        # self.cam_box contains the two corner points of the rectangular prism that contains all the camera positions
        self.get_cam_bbox()

        self.skip_depth_loading = kwargs.get('skip_depth_loading')

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        # read undistorted intrinsics
        with open(self.transforms_undistorted_path, 'r') as file:
            transforms_undistorted_data = json.load(file)
        
        # extract intrinsics
        fl_x = transforms_undistorted_data.get('fl_x')
        fl_y = transforms_undistorted_data.get('fl_y')
        cx = transforms_undistorted_data.get('cx')
        cy = transforms_undistorted_data.get('cy')
        w = transforms_undistorted_data.get('w')
        h = transforms_undistorted_data.get('h')
        # we assume the skew-coefficient to be 0 for high-quality DSLR images
        s = 0

        # generate homogenous intrinsics matrix
        # K is intrinsic matrix, which contains internal parameters of camera
        # contains focal lengths fx and fy in terms of pixels
        # contains principal points cx and cy, where optical axis intersects image plane
        # contains skew coefficient s, which accounts for the non-orthogonality between the x and y pixel axes. Ideally 0
        # Needed to go from 3D point in camera coordinate system to 2D point in image
        self.K = torch.eye(3, dtype=torch.float32)
        self.K[0, 0] = fl_x
        self.K[0, 1] = s
        self.K[0, 2] = cx
        self.K[1, 1] = fl_y
        self.K[1, 2] = cy

        H, W = h - 2 * self.unpad, w - 2 * self.unpad
        # adjusts the two principal points cx and cy to the padding
        self.K[:2, 2] -= self.unpad
        self.directions = get_ray_directions(H, W, self.K)
        # adjusted image width and height
        self.img_wh = (W, H)

    def get_cam_bbox(self):
        with open(self.transforms_undistorted_path, 'r') as file:
            transforms_undistorted_data = json.load(file) 

        # Initialize lists to store xyz values
        xyzs = []

        # Iterate over all frames and extract the translation components
        for frame in transforms_undistorted_data['frames']:
            transform_matrix = np.array(frame['transform_matrix'])
            # Extract the translation component (last column, first three rows)
            xyz = transform_matrix[:3, 3]
            xyzs.append(xyz)

        # Convert the list of xyz values to a NumPy array for easy min/max computation
        xyzs = np.array(xyzs)

        # Compute the minimum and maximum values along each axis
        xyz_min = xyzs.min(axis=0)
        xyz_max = xyzs.max(axis=0)
        self.cam_bbox = np.array([xyz_min, xyz_max])

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        with open(self.transforms_undistorted_path, 'r') as file:
            transforms_undistorted_data = json.load(file) 

        if split == 'train':
            # TODO(mschneider): Scannet Dataset loads 800 frames for training, training might be less comparable to Scannet
            # -> could find a scene with 800 frames if necessary
            frames = transforms_undistorted_data['frames']
        else:
            # TODO(mschneider): Scannet Dataset loads 80 frames for testing, check if this makes difference in comparability between training
            frames = transforms_undistorted_data['test_frames']

        # calculates the scaling factor to be applied to all frames by adding the largest dimension of the bounding box and the margin
        # because of 2 * SCANNET_FAR, the normalized space is slightly larger than the actual bounding box
        sbbox_scale = (self.cam_bbox[1] - self.cam_bbox[0]).max() + 2 * SCANNET_FAR
        # mean of bounding box is equivalent to the center of the bounding box,
        # center is used to shift all camera positions so they are centered around the origin after normalization
        sbbox_shift = self.cam_bbox.mean(axis=0)

        # Iterate over all frames and extract the translation components
        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            # camera-to-world matrix == transform-matrix
            # remove the last row of transform_matrix to get (3, 4) poses
            c2w = np.array(frame['transform_matrix'])[:3, :]

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
                img_path = os.path.join(self.root_dir, f"./dslr/undistorted_images/{frame['file_path']}")
                # TODO(mschneider): check if we need a custom read_image to deal with Scannet++ data
                img = read_image(img_path, self.img_wh, unpad=self.unpad)
                # contains alpha-blended, unpadded, resized images in dimension ((height * width), channels)
                self.rays += [img]
            except: pass

        if len(self.rays)>0:
            # ?: depends on whether alpha-values for transparency are given.
            # Either (N_images, h*w, channels) or (N_images, h*w, channels, alpha)
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, h*w, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
