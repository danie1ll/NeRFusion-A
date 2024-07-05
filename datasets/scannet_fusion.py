from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import pickle
from PIL import Image
import cv2
import torch

from .ray_utils import get_ray_directions
from .color_utils import read_image, read_depth

# because of this factor, normalized frames are slightly larger than original bounding box
# unclear why this is needed
SCANNET_FAR = 2.0


class ScanNetFusion(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, nviews, n_scales, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample

        self.unpad = 24
        self.num_frames_train = kwargs.get('num_frames_train', 800)
        self.num_frames_test = kwargs.get('num_frames_test', 80)

        self.read_intrinsics()

        print('Depth being loaded:', not kwargs.get('skip_depth_loading', False))

        self.skip_depth_loading = kwargs.get('skip_depth_loading')

        if kwargs.get('read_meta', True):
            self.read_meta(split)

        ### FUSION
        self.n_views = nviews
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.split in ["train", "val", "test"]
        self.metas = self.build_list()
        if split == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans' 

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

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
        self.depths = []

        n_samples = self.num_frames_train if split == 'train' else self.num_frames_test

        with open(os.path.join(self.root_dir, f"{split}.txt"), 'r') as f:
            frames = f.read().strip().split()
            frames = frames[:n_samples]
       
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

                if not self.skip_depth_loading:
                    depth_path = os.path.join(self.root_dir, f"depth/{frame}.png")
                    depth_image = read_depth(depth_path, self.img_wh, unpad=self.unpad)
                    self.depths += [depth_image]

            except: pass

        if len(self.rays) > 0:
            # ?: depends on whether alpha-values for transparency are given.
            # Either (N_images, h*w, channels) or (N_images, h*w, channels, alpha)
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)

        if len(self.depths) > 0:
            self.depths = torch.FloatTensor(np.stack(self.depths))  # (N_images, hw, ?)

        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        # we're returning a single image/fram of a scene
        # we've already loaded all images/frames into the array and are just returning the image corresponding 
        # to a specific index
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]

            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]

            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs, 'rgb': rays[:, :3]}

            if not self.skip_depth_loading:
                depths = self.depths[img_idxs, pix_idxs]

                if len(depths) > 0:
                    sample['depth'] = depths

            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            # is this the pose for a single image or for several images?
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample

    def __getitem__(self, idx):
        # we're loading all images of a scene?
        meta = self.metas[idx]

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items