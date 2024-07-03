import os
import sys
import tqdm

import numpy as np

# Usage: python scannet_get_bbox.py <root_dir>
# root_dir: the root directory of a specific scene in the scannet dataset

if __name__ == '__main__':
    root_dir = sys.argv[1]
    xyzs = []
    for pose_file in tqdm.tqdm(os.listdir(os.path.join(root_dir, 'pose'))):
        # 4x4 transform-matrix to represent position and orientation of camera
        pose = np.loadtxt(os.path.join(root_dir, f'pose/{pose_file}'))
        # extract the 3x1 translation vector, which is equivalent to the position of the camera
        xyz = pose[:3, -1]
        xyzs.append(xyz)

    xyzs = np.array(xyzs)
    xyz_min = xyzs.min(axis=0)
    xyz_max = xyzs.max(axis=0)

    output = np.array([xyz_min, xyz_max])
    np.savetxt(os.path.join(root_dir, 'cam_bbox.txt'), output)

