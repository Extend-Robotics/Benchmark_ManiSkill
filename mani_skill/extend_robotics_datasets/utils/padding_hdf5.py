import h5py
import os
import numpy as np
from tqdm import tqdm
import argparse



def check_max_step(input_dir):
    max_index = 0
    max_step = 0
    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    for i in range(len(hdf5_files)):
        dataset_path = os.path.join(input_dir, "episode_" + str(i)+".hdf5")
        with h5py.File(dataset_path, "r") as root:
            steps = root['/action'].shape[0]
            if steps > max_step:
                max_step = steps
                max_index = i
    return max_step, max_index

# def add_padding(input_file, max_step):
#     with h5py.File(input_file, "a") as root:
#         for dataset_name in ['/action', '/observations/qpos', '/observations/qvel']:
#             last_row = root[dataset_name][()][-1, :8]
#             padding = np.tile(last_row, (max_step-root[dataset_name][()].shape[0], 1))
#             expanded_data = np.vstack([root[dataset_name][()][:,:8], padding])
#             del root[dataset_name]
#             root.create_dataset(dataset_name, data=expanded_data)
#         success_data = root['/success'][()]
#         last_value = success_data[-1]  # Scalar
#         padding_len = max_step - success_data.shape[0]

#         # Create a padding array of shape (padding_len,)
#         padding = np.full((padding_len,), last_value, dtype=success_data.dtype)

#         # Stack to get the new fixed-length array
#         expanded_data = np.concatenate([success_data, padding])

#         # Overwrite the dataset
#         del root['/success']
#         root.create_dataset('/success', data=expanded_data)
        
#         for dataset_name in ['/observations/images/base_camera', '/observations/images/hand_camera']:
#             if dataset_name not in root:
#                 continue
#             last_row = root[dataset_name][()][-1, :,:,:]
#             padding = np.tile(last_row, ((max_step-root[dataset_name][()].shape[0],1,1,1)))
#             expanded_data = np.vstack([root[dataset_name][()], padding])
#             del root[dataset_name]
#             root.create_dataset(dataset_name, data=expanded_data)       
#         root.attrs['sim'] = True

import h5py
import numpy as np

def add_padding(input_file, max_step):
    with h5py.File(input_file, "a") as root:
        for dataset_name in ['/action', '/observations/qpos', '/observations/qvel']:
            data = root[dataset_name][()][:, :8]
            curr_len = data.shape[0]
            
            if curr_len < max_step:
                last_row = data[-1]
                padding = np.tile(last_row, (max_step - curr_len, 1))
                expanded_data = np.vstack([data, padding])
            else:
                expanded_data = data[:max_step]

            del root[dataset_name]
            root.create_dataset(dataset_name, data=expanded_data)

        # Handle /success
        success_data = root['/success'][()]
        curr_len = success_data.shape[0]
        last_value = success_data[-1]
        
        if curr_len < max_step:
            padding = np.full((max_step - curr_len,), last_value, dtype=success_data.dtype)
            expanded_data = np.concatenate([success_data, padding])
        else:
            expanded_data = success_data[:max_step]

        del root['/success']
        root.create_dataset('/success', data=expanded_data)

        for dataset_name in ['/observations/images/base_camera', '/observations/images/hand_camera']:
            if dataset_name not in root:
                continue

            data = root[dataset_name][()]
            curr_len = data.shape[0]

            if curr_len < max_step:
                last_frame = data[-1]
                padding = np.tile(last_frame[None, :, :, :], (max_step - curr_len, 1, 1, 1))
                expanded_data = np.concatenate([data, padding], axis=0)
            else:
                expanded_data = data[:max_step]

            del root[dataset_name]
            root.create_dataset(dataset_name, data=expanded_data)

        root.attrs['sim'] = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add padding to HDF5 files to ensure uniform length.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing episode HDF5 files")
    args = parser.parse_args()
    input_dir = args.input_dir
    max_step, max_index = check_max_step(input_dir)
    print("Max step:", max_step)
    print("Index", max_index)
    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    # ✅ Add tqdm progress bar
    for i in tqdm(range(len(hdf5_files)), desc="Adding padding"):
        dataset_path = os.path.join(input_dir, f"episode_{i}.hdf5")
        add_padding(dataset_path, 180)

    print("✅ Padding completed for all files.")
