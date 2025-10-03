import argparse
from pathlib import Path

from mani_skill.extend_robotics_datasets.extend_hdf5 import DatasetConverter, str2bool


def main():
    """
    Convert Extend HDF5 dataset to LeRobot format .
    This script processes raw HDF5 files from the Aloha dataset, converts them into a specified format,
    Parameters
    ----------
    --raw-path : Path
        Directory containing the raw HDF5 files.
    --dataset-repo-id : str
        Repository ID where the dataset will be stored.
    --fps : int
        Frames per second for the dataset.
    --robot-type : str, optional
        Type of robot, either "aloha-stationary" or "aloha-mobile". Default is "aloha-stationary".
    --image-compressed : bool, optional
        Set to True if the images are compressed. Default is True.
    --video-encoding : bool, optional
        Set to True to encode images as videos. Default is True.
    --nproc : int, optional
        Number of image writer processes. Default is 10.
    --nthreads : int, optional
        Number of image writer threads. Default is 5.
    """

    parser = argparse.ArgumentParser(description="Convert Extend HDF5 dataset")
    parser.add_argument(
        "--raw-path",
        type=Path,
        required=True,
        help="Directory containing the raw hdf5 files.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Repository ID where the dataset will be stored.",
    )
    parser.add_argument(
        "--fps", type=int, required=True, help="Frames per second for the dataset."
    )

    parser.add_argument(
        "--rgb-mode",
        type=str2bool,
        required=False,
        help="Convert to RGB color if the image is in BGR",
        default=False,
    )

    parser.add_argument(
        "--use-depth-images",
        type=str2bool,
        required=False,
        help="Use depth images if the dataset contains depth image frames",
        default=False,
    )

    parser.add_argument(
        "--description",
        type=str,
        help="Description of the dataset.",
        default="Extend Robotics recorded dataset",
    )

    parser.add_argument(
        "--robot-type",
        type=str,
        choices=["stationary", "mobile"],
        default="stationary",
        help="Type of robot.",
    )
    parser.add_argument(
        "--image-compressed",
        type=str2bool,
        default=False,
        help="Set to True if the images are compressed.",
    )
    parser.add_argument(
        "--video-encoding",
        type=str2bool,
        default=True,
        help="Set to True to encode images as videos.",
    )

    parser.add_argument(
        "--nproc", type=int, default=10, help="Number of image writer processes."
    )
    parser.add_argument(
        "--nthreads", type=int, default=5, help="Number of image writer threads."
    )

    args = parser.parse_args()

    converter = DatasetConverter(
        raw_path=args.raw_path,
        dataset_repo_id=args.dataset_repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        image_compressed=args.image_compressed,
        encode_as_videos=args.video_encoding,
        image_writer_processes=args.nproc,
        image_writer_threads=args.nthreads,
        rgb_mode=args.rgb_mode,
        use_depth_images=args.use_depth_images,
    )
    converter.init_lerobot_dataset()
    converter.extract_episodes(episode_description=args.description)


if __name__ == "__main__":
    main()
