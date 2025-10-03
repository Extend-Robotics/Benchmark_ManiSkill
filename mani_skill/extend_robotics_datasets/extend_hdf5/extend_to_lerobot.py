import torch
import logging
import shutil
import einops
import PIL
import tqdm
import datasets
import numpy as np
from math import ceil
from copy import deepcopy
from pathlib import Path
from PIL import Image
from typing import Callable

from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_VIDEO_PATH,
    DEFAULT_PARQUET_PATH,
    DEFAULT_FEATURES,
    INFO_PATH,
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    write_json,
    serialize_dict,
)
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    LEROBOT_HOME,
    CODEBASE_VERSION,
    get_features_from_robot,
)

from lerobot.common.datasets.image_writer import write_image

DEFAULT_DEPTH_IMAGE_PATH = "images_depth/chunk-{episode_chunk:03d}/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    robot_type: str,
    features: dict,
    use_videos: bool,
    use_depth_images: bool,
) -> dict:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "depth_images_path": DEFAULT_DEPTH_IMAGE_PATH if use_depth_images else None,
        "features": features,
    }


def load_depth_image(depth_image_path: Path | str) -> torch.Tensor:
    depth_image = Image.open(depth_image_path)
    return torch.from_numpy(np.array(depth_image).astype(np.uint16)).unsqueeze(0)


def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video" or ft["dtype"] == "uint16":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        else:
            assert len(ft["shape"]) == 1
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )

    return datasets.Features(hf_features)


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.video_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert (
                c < h and c < w
            ), f"expect channel first images, but instead {batch[key].shape}"

            # sanity check that images are float32 in range [0,1]
            assert (
                batch[key].dtype == torch.float32
            ), f"expect torch.float32, but instead {batch[key].dtype=}"
            assert (
                batch[key].max() <= 1
            ), f"expect pixels lower than 1, but instead {batch[key].max()=}"
            assert (
                batch[key].min() >= 0
            ), f"expect pixels greater than 1, but instead {batch[key].min()=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif key in dataset.meta.depth_image_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert (
                c < h and c < w
            ), f"expect channel first images, but instead {batch[key].shape}"

            assert (
                batch[key].dtype == torch.uint16
            ), f"expect torch.unit16, but instead {batch[key].dtype=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def compute_stats(dataset, batch_size=8, num_workers=8, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm.tqdm(
            dataloader,
            total=ceil(max_num_samples / batch_size),
            desc="Compute mean, min, max",
        )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = (
                mean[key]
                + this_batch_size * (batch_mean - mean[key]) / running_item_count
            )
            max[key] = torch.maximum(
                max[key], einops.reduce(batch[key], pattern, "max")
            )
            min[key] = torch.minimum(
                min[key], einops.reduce(batch[key], pattern, "min")
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm.tqdm(
            dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std"
        )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = (
                std[key] + this_batch_size * (batch_std - std[key]) / running_item_count
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats


class ExtendRoboticsDatasetMetadata(LeRobotDatasetMetadata):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        local_files_only: bool = False,
        use_depth_images: bool = False,
    ):
        super().__init__(repo_id=repo_id, root=root, local_files_only=local_files_only)
        self.use_depth_images = use_depth_images

    def get_depth_image_file_path(
        self, ep_index: int, image_key: str, frame_index: int
    ) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.depth_images_path.format(
            episode_chunk=ep_chunk,
            image_key=image_key,
            episode_index=ep_index,
            frame_index=frame_index,
        )
        return Path(fpath)

    @property
    def depth_images_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["depth_images_path"]

    @property
    def depth_image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "uint16"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [
            key
            for key, ft in self.features.items()
            if ft["dtype"] in ["video", "image", "uint16"]
        ]

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        use_depth_images: bool = False,
    ) -> "ExtendRoboticsDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else LEROBOT_HOME / repo_id
        obj.use_depth_images = use_depth_images
        obj.root.mkdir(parents=True, exist_ok=False)

        if robot is not None:
            features = get_features_from_robot(robot, use_videos)
            robot_type = robot.robot_type
            if not all(cam.fps == fps for cam in robot.cameras.values()):
                logging.warning(
                    f"Some cameras in your {robot.robot_type} robot don't have an fps matching the fps of your dataset."
                    "In this case, frames from lower fps cameras will be repeated to fill in the blanks."
                )
        elif features is None:
            raise ValueError(
                "Dataset features must either come from a Robot or explicitly passed upon creation."
            )
        else:
            # TODO(aliberts, rcadene): implement sanity check for features

            # check if none of the features contains a "/" in their names,
            # as this would break the dict flattening in the stats computation, which uses '/' as separator
            for key in features:
                if "/" in key:
                    raise ValueError(
                        f"Feature names should not contain '/'. Found '/' in feature '{key}'."
                    )

            features = {**features, **DEFAULT_FEATURES}

        obj.tasks, obj.stats, obj.episodes = {}, {}, []
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION, fps, robot_type, features, use_videos, use_depth_images
        )
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.local_files_only = True
        return obj


class ExtendRoboticsDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        use_depth_images: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

        # Load metadata
        self.meta = ExtendRoboticsDatasetMetadata(
            self.repo_id,
            self.root,
            self.local_files_only,
            use_depth_images=use_depth_images,
        )

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_depth_image_file_path(
        self, episode_index: int, image_key: str, frame_index: int
    ) -> Path:
        ep_chunk = self.meta.get_episode_chunk(ep_index=episode_index)
        fpath = DEFAULT_DEPTH_IMAGE_PATH.format(
            episode_chunk=ep_chunk,
            image_key=image_key,
            episode_index=episode_index,
            frame_index=frame_index,
        )
        return self.root / fpath

    def _get_query_depth_image_indices(
        self,
        current_idx: int,
        episode_idx: int,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[int]]:
        depth_image_indices = {}
        for key in self.meta.depth_image_keys:
            if query_indices is not None and key in query_indices:
                depth_image_indices[key] = [
                    idx - self.episode_data_index["from"][episode_idx].item()
                    for idx in query_indices[key]
                ]
            else:
                depth_image_indices[key] = [current_idx]

        return depth_image_indices

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.camera_keys
        }

    def _query_depth_images(
        self, query_depth_image_indices: dict[str, list[int]], ep_idx: int
    ) -> dict:
        item = {}
        for depth_key, query_idx in query_depth_image_indices.items():
            frames = []
            for frame_idx in query_idx:
                depth_image_path = self.root / self.meta.get_depth_image_file_path(
                    ep_index=ep_idx, image_key=depth_key, frame_index=frame_idx
                )
                frames.append(
                    load_depth_image(depth_image_path=depth_image_path).squeeze(0)
                )
            item[depth_key] = torch.stack(frames)

        return item

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            current_ep_idx = (
                self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
            )
            query_indices, padding = self._get_query_indices(idx, current_ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if len(self.meta.depth_image_keys) > 0 and self.meta.use_depth_images:
            current_frame_index = item["frame_index"].item()
            query_depth_image_indices = self._get_query_depth_image_indices(
                current_idx=current_frame_index,
                episode_idx=ep_idx,
                query_indices=query_indices,
            )
            depth_frames = self._query_depth_images(query_depth_image_indices, ep_idx)
            item = {**depth_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                if cam in item:
                    item[cam] = self.image_transforms(item[cam])

        return item

    def _save_depth_image(
        self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path
    ) -> None:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        write_image(image, fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)

            dtype = self.features[key]["dtype"]

            if dtype not in ["image", "video", "uint16"]:
                item = (
                    frame[key].numpy()
                    if isinstance(frame[key], torch.Tensor)
                    else frame[key]
                )
                self.episode_buffer[key].append(item)
            else:
                # Determine the appropriate path and save method for images or depth images
                if dtype in ["image", "video"]:
                    img_path = self._get_image_file_path(
                        episode_index=self.episode_buffer["episode_index"],
                        image_key=key,
                        frame_index=frame_index,
                    )
                    save_method = self._save_image
                elif dtype == "uint16":
                    if self.meta.use_depth_images:
                        img_path = self._get_depth_image_file_path(
                            episode_index=self.episode_buffer["episode_index"],
                            image_key=key,
                            frame_index=frame_index,
                        )
                        save_method = self._save_depth_image
                    else:
                        continue

                # Create directory if it's the first frame
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the image or depth image and update the buffer
                save_method(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))

        self.episode_buffer["size"] += 1

    def save_episode(
        self, task: str, encode_videos: bool = True, episode_data: dict | None = None
    ) -> None:
        """
        This will save to disk the current episode in self.episode_buffer. Note that since it affects files on
        disk, it sets self.consolidated to False to ensure proper consolidation later on before uploading to
        the hub.

        Use 'encode_videos' if you want to encode videos during the saving of this episode. Otherwise,
        you can do it later with dataset.consolidate(). This is to give more flexibility on when to spend
        time for video encoding.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        episode_length = episode_buffer.pop("size")
        episode_index = episode_buffer["episode_index"]
        if episode_index != self.meta.total_episodes:
            # TODO(aliberts): Add option to use existing episode_index
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        if episode_length == 0:
            raise ValueError(
                "You must add one or several frames with `add_frame` before calling `add_episode`."
            )

        task_index = self.meta.get_task_index(task)

        if not set(episode_buffer.keys()) == set(self.features):
            raise ValueError()

        for key, ft in self.features.items():

            if key == "index":
                episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                episode_buffer[key] = np.full((episode_length,), episode_index)
            elif key == "task_index":
                episode_buffer[key] = np.full((episode_length,), task_index)
            elif ft["dtype"] in ["image", "video", "uint16"]:
                continue
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                episode_buffer[key] = np.stack(episode_buffer[key])
            else:
                raise ValueError(key)

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        self.meta.save_episode(episode_index, episode_length, task, task_index)

        if encode_videos and len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

        self.consolidated = False

    def consolidate(
        self, run_compute_stats: bool = True, keep_image_files: bool = False
    ) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        check_timestamps_sync(
            self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s
        )

        if len(self.meta.video_keys) > 0:
            self.encode_videos()
            self.meta.write_video_info()

        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        if run_compute_stats:
            self.stop_image_writer()
            # TODO(aliberts): refactor stats in save_episodes
            self.meta.stats = compute_stats(self)
            serialized_stats = serialize_dict(self.meta.stats)
            write_json(serialized_stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        use_depth_images: bool = False,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "ExtendRoboticsDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = ExtendRoboticsDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=robot,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
            use_depth_images=use_depth_images,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.local_files_only = obj.meta.local_files_only
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        # This bool indicates that the current LeRobotDataset instance is in sync with the files on disk. It
        # is used to know when certain operations are need (for instance, computing dataset statistics). In
        # order to be able to push the dataset to the hub, it needs to be consolidated first by calling
        # self.consolidate().
        obj.consolidated = True

        obj.episodes = None
        obj.hf_dataset = None
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else "pyav"
        return obj
