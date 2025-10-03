import h5py
import os
import numpy as np
from tqdm import tqdm
import argparse



def check_episode_step(input_dir):
    episode_lengths = []
    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    for i in range(len(hdf5_files)):
        dataset_path = os.path.join(input_dir, "episode_" + str(i)+".hdf5")
        with h5py.File(dataset_path, "r") as root:
            steps = root['action'].shape[0]
            episode_lengths.append(steps)
    return episode_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check episode lengths in HDF5 files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing episode HDF5 files")
    args = parser.parse_args()
    input_dir = args.input_dir
    episode_lengths = check_episode_step(input_dir)
    print("Episode lengths:", episode_lengths)
    print("Percentiles:", np.percentile(episode_lengths, [50,75, 90, 95, 99, 100]))