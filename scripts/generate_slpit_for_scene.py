import os
import sys
import tqdm

import numpy as np

# Usage: python generate_split_for_scene.py <root_dir>
# root_dir: the root directory of a specific scene in the scannet dataset

if __name__ == '__main__':
    root_dir = sys.argv[1]
    all_frames = []
    for frame in tqdm.tqdm(os.listdir(os.path.join(root_dir, 'color'))):
        all_frames.append(frame)

    # create a train, test, val split
    train_split = all_frames[:int(len(all_frames) * 0.8)]
    test_split = all_frames[int(len(all_frames) * 0.8):int(len(all_frames) * 0.9)]
    val_split = all_frames[int(len(all_frames) * 0.9):]

    # Function to write split data to file
    def write_split(filename, split_data):
        filepath = os.path.join(root_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for frame in split_data:
                # append only the frame name to the file
                f.write(f'{frame.split(".")[0]}\n')

    # Write splits to files
    write_split('train.txt', train_split)
    write_split('trainval.txt', val_split)
    write_split('test.txt', test_split)
