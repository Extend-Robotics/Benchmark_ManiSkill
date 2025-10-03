import h5py
import os
import glob
import argparse

parser = argparse.ArgumentParser(description="Process HDF5 files")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input HDF5 file")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output episodes")
args = parser.parse_args()

input_file = args.input_file
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)
with h5py.File(input_file, 'r') as f:
    for traj in f.keys():
        traj_data = f[traj]

        output_file = os.path.join(output_dir, f"episode_{traj}.hdf5")

        with h5py.File(output_file, "w") as traj_file:
            f.copy(traj_data, traj_file)

matching_files = glob.glob(f"{output_dir}/episode_*.hdf5")
print(matching_files)