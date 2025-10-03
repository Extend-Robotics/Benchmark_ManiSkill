import logging
import shutil
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Callable

import datasets
import numpy as np
import packaging.version
import PIL
import torch
from lerobot.datasets.compute_stats import (aggregate_feature_stats,
                                            auto_downsample_height_width,
                                            get_feature_stats, sample_images,
                                            sample_indices, compute_stats)
from lerobot.datasets.lerobot_dataset import (CODEBASE_VERSION,
                                              HF_LEROBOT_HOME, LeRobotDataset,
                                              LeRobotDatasetMetadata)
from lerobot.datasets.utils import (DEFAULT_CHUNK_SIZE, DEFAULT_FEATURES,
                                    DEFAULT_PARQUET_PATH, DEFAULT_VIDEO_PATH,
                                    INFO_PATH, STATS_PATH, _validate_feature_names,
                                    backward_compatible_episodes_stats,
                                    check_delta_timestamps,
                                    check_timestamps_sync,
                                    check_version_compatibility,
                                    get_delta_indices, get_episode_data_index,
                                    get_safe_version, hf_transform_to_torch,
                                    load_episodes, load_episodes_stats,
                                    load_info, load_stats, load_tasks,
                                    validate_episode_buffer, validate_frame,
                                    write_episode, write_episode_stats,
                                    write_info, write_json, serialize_dict, check_timestamps_sync_last)
from lerobot.datasets.video_utils import get_safe_default_codec
from lerobot.robots import Robot
from PIL import Image

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


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError(
                        "Number of dimensions must be at least 1, and is 0 instead."
                    )
                if k == "count" and v.shape != (1,):
                    raise ValueError(
                        f"Shape of 'count' must be (1), but is {v.shape} instead."
                    )
                if "images_depth." in fkey and k != "count" and v.shape != (1, 1, 1):
                    raise ValueError(
                        f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead."
                    )
                elif "images." in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(
                        f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead."
                    )


def aggregate_stats(
    stats_list: list[dict[str, dict]],
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def load_depth_image(depth_image_path: Path | str) -> torch.Tensor:
    depth_image = Image.open(depth_image_path)
    return torch.from_numpy(np.array(depth_image).astype(np.uint16)).unsqueeze(0)


def load_depth_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.uint16
) -> np.ndarray:
    img = Image.open(fpath)
    img_array = np.array(img, dtype=dtype)
    assert img_array.ndim == 2
    img_array = img_array.reshape(-1, *img_array.shape)  # C, H , W ; C = 1
    return img_array


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


def sample_depth_images(depth_image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(depth_image_paths))
    depth_images = None

    for i, idx in enumerate(sampled_indices):
        path = depth_image_paths[idx]
        depth_img = load_depth_image_as_numpy(path, dtype=np.uint16)
        depth_img = auto_downsample_height_width(depth_img)
        if depth_images is None:
            depth_images = np.empty(
                (len(sampled_indices), *depth_img.shape), dtype=np.uint16
            )
        depth_images[i] = depth_img
    return depth_images


def compute_episode_stats(
    episode_data: dict[str, list[str] | np.ndarray], features: dict
) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        elif features[key]["dtype"] == "uint16":
            ep_ft_array = sample_depth_images(data)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(
            ep_ft_array, axis=axes_to_reduce, keepdims=keepdims
        )

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video", "uint16"]:
            value_norm = 1.0 if features[key]["dtype"] == "uint16" else 255.0
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / value_norm, axis=0)
                for k, v in ep_stats[key].items()
            }

    return ep_stats


class ExtendRoboticsDatasetMetadata(LeRobotDatasetMetadata):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        use_depth_images: bool = False,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            revision=revision,
            force_cache_sync=force_cache_sync,
        )
        self.use_depth_images = use_depth_images

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(
                self.stats, self.episodes
            )
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

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

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = (
            aggregate_stats([self.stats, episode_stats])
            if self.stats
            else episode_stats
        )
        write_episode_stats(episode_index, episode_stats, self.root)

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
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        obj.use_depth_images = use_depth_images
        obj.root.mkdir(parents=True, exist_ok=False)

        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION, fps, robot_type, features, use_videos, use_depth_images
        )
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
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
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        use_depth_images: bool = False,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = (
            video_backend if video_backend else get_safe_default_codec()
        )
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = ExtendRoboticsDatasetMetadata(
            self.repo_id,
            self.root,
            self.revision,
            force_cache_sync=force_cache_sync,
            use_depth_images=use_depth_images,
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse(
            "v2.1"
        ):
            episodes_stats = [
                self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes
            ]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            # assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )

        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(
            timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s
        )

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.consolidated = self.meta.stats is not None

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(
            ft_dict, features=features, split="train"
        )

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

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
            query_indices, padding = self._get_query_indices(idx, ep_idx)
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

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item

    def _save_depth_image(
        self, depth_image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path
    ) -> None:
        if isinstance(depth_image, torch.Tensor):
            depth_image = depth_image.cpu().numpy().astype(np.uint16)
        depth_image_pil = Image.fromarray(depth_image)
        depth_image_pil.save(str(fpath), quality=100)

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            dtype = self.features[key]["dtype"]
            if dtype in ["image", "video", "uint16"]:
                if self.features[key]["dtype"] in ["image", "video"]:
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

                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                save_method(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Video encoding is handled automatically based on batch_encoding_size:
        - If batch_encoding_size == 1: Videos are encoded immediately after each episode
        - If batch_encoding_size > 1: Videos are encoded in batches.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
            save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
            None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task) for task in tasks]
        )

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in [
                "image",
                "video",
                "uint16",
            ]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        if has_video_keys and not use_batched_encoding:
            self.encode_episode_videos(episode_index)

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        # Check if we should trigger batch encoding
        if has_video_keys and use_batched_encoding:
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                logging.info(
                    f"Batch encoding {self.batch_encoding_size} videos for episodes {start_ep} to {end_ep - 1}"
                )
                self.batch_encode_videos(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}

        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        # video_files = list(self.root.rglob("*.mp4"))
        # assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()
    
    def consolidate(self, run_compute_stats: bool = True, keep_image_files: bool = False) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        check_timestamps_sync_last(self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s)

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
        batch_encoding_size: int = 1,
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
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = (
            video_backend if video_backend is not None else get_safe_default_codec()
        )
        return obj