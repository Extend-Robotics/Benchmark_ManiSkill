import os
import h5py
import re
import argparse

def format_hdf5(input_dir, output_dir):
   
    os.makedirs(output_dir, exist_ok=True)

    for files in sorted(os.listdir(input_dir)):
        if files.endswith('.hdf5'):
            input_file = os.path.join(input_dir, files)
            match = re.search(r'episode_traj_(\d+).hdf5', files)
            if match:
                index = int(match.group(1))
            else:
                print("No number found in filename.")
                break

            with h5py.File(input_file, 'r') as f:
                data = f[f'traj_{index}']
                output_file = os.path.join(output_dir, f'episode_{index}.hdf5')

                with h5py.File(output_file, 'w') as out_f:
                    # Prepare groups
                    obs_group = out_f.create_group('observations')
                    img_group = obs_group.create_group('images')

                    # Copy basic datasets
                    out_f.create_dataset('action', data=data['actions'])
                    obs_group.create_dataset('qpos', data=data['obs']['agent']['qpos'][:-1])
                    obs_group.create_dataset('qvel', data=data['obs']['agent']['qvel'][:-1])
                    out_f.create_dataset('tcp_pose', data=data['obs']['extra']['tcp_pose'][:-1])
                    out_f.create_dataset('success', data=data['success'])

                    # Copy images
                    base_img = data['obs']['sensor_data']['base_camera']['rgb'][:-1]
                    try: 
                        hand_img = data['obs']['sensor_data']['hand_camera']['rgb'][:-1]
                        
                    except KeyError:
                        print("Hand camera data not found, using base camera data instead.")
                    img_group.create_dataset('base_camera', data=base_img)
                    camera_names = ['base_camera']
                    if hand_img is not None:
                        img_group.create_dataset('hand_camera', data=hand_img)
                        camera_names.append('hand_camera')
                    out_f.create_dataset('camera_names', data=camera_names)

                    

                    # Define groups to copy with mapping old path -> new top-level group
                    group_mappings = {
                        'obs/sensor_param/base_camera': 'sensor_param/base_camera',
                        'obs/sensor_param/hand_camera': 'sensor_param/hand_camera',
                        # 'env_states/actors': 'env_states/actors',
                        # 'env_states/articulations': 'env_states/articulations'
                    }

                    def ensure_group(out_file, group_path):
                        parts = group_path.split('/')
                        current_group = out_file
                        for part in parts:
                            if part not in current_group:
                                current_group = current_group.create_group(part)
                            else:
                                current_group = current_group[part]
                        return current_group

                    for old_group, new_group in group_mappings.items():
                        if old_group in data:
                            source_group = data[old_group]
                            target_group = ensure_group(out_f, new_group)

                            for name, item in source_group.items():
                                if isinstance(item, h5py.Dataset):
                                    target_group.create_dataset(name, data=item[()])
                                    print(f"✅ Copied dataset: {new_group}/{name}")
                                else:
                                    # If there are nested groups, handle recursively
                                    nested_group = target_group.create_group(name)
                                    for sub_name, sub_item in item.items():
                                        nested_group.create_dataset(sub_name, data=sub_item[()])
                                        print(f"✅ Copied nested dataset: {new_group}/{name}/{sub_name}")
                        else:
                            print(f"⚠️ Group {old_group} not found in input file.")

            print(f"🎉 Data successfully formatted and saved to {output_file}.")
            
            # Process all files: remove break
            # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format HDF5 files from trajectory format to episode format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input HDF5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save formatted HDF5 files")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    format_hdf5(input_dir, output_dir)
