from typing import Callable, List, Type
import sys
sys.path.append('/')
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
# from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import torch
from collections import deque
from PIL import Image
import cv2


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PegInsertionSide-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    return parser.parse_args()

import random
import os

# set cuda 
args = parse_args()
# set random seeds
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


env_id = args.env_id

env = gym.make(
    env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode=args.render_mode,
    reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    sim_backend=args.sim_backend
)


MAX_EPISODE_STEPS = 400 
total_episodes = args.num_traj  

import tqdm
for round_idx in range(1):
    success_count = 0 
    total_steps = 0
    folder_path = '/home/kelin/benchmark_maniskill/round'+'_'+str(round_idx)
    os.makedirs(folder_path, exist_ok=True)
    for episode in tqdm.trange(1):
        seed = random.randint(0,20250601)
        obs_window = deque(maxlen=2)
        obs, _ = env.reset(seed=seed)

        base_img = obs['sensor_data']['base_camera']['rgb'].squeeze(0).detach().cpu().numpy()[:, :, [2, 1, 0]]
        wrist_img = obs['sensor_data']['hand_camera']['rgb'].squeeze(0).detach().cpu().numpy()[:, :, [2, 1, 0]]
        obs_window.append(None)
        obs_window.append(np.array(base_img))
        proprio = obs['agent']['qpos'][:, :-1]

        global_steps = 0
        video_frames = []

        success_time = 0
        done = False

        while global_steps < MAX_EPISODE_STEPS and not done:
            image_arrs = []
            for window_img in obs_window:
                image_arrs.append(window_img)
                image_arrs.append(None)
                image_arrs.append(None)
            images = [Image.fromarray(arr) if arr is not None else None
                    for arr in image_arrs]
            
            # img = img[:, :, [2, 1, 0]]
            base_img = torch.from_numpy(base_img/255.0).permute(2, 0, 1)
            base_img = base_img.unsqueeze(0)
            base_img = base_img.to(torch.float32)
            base_img = base_img.to('cuda')
            wrist_img = torch.from_numpy(wrist_img/255.0).permute(2, 0, 1)
            wrist_img = wrist_img.unsqueeze(0)
            wrist_img = wrist_img.to(torch.float32)
            wrist_img = wrist_img.to('cuda')

            observation = {'observation.images.base_camera':base_img,'observation.images.hand_camera':wrist_img,'observation.state':proprio.to('cuda')}

            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            base_img = obs['sensor_data']['base_camera']['rgb'].squeeze(0).detach().cpu().numpy()[:, :, [2, 1, 0]]
            wrist_img = obs['sensor_data']['hand_camera']['rgb'].squeeze(0).detach().cpu().numpy()[:, :, [2, 1, 0]]
            video_frames.append(wrist_img)
            global_steps += 1

        # Save the video
        height, width = 128, 128
        fps = 30  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(folder_path+'/'+str(episode+1)+'_'+str(seed)+'.mp4', fourcc, fps, (width, height))
        for frame in video_frames:
            out.write(frame)

        out.release()
        print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")
        total_steps = total_steps + global_steps

    success_rate = success_count / total_episodes * 100
    print(f"Success rate: {success_rate}%")
    result = [{'total_num': total_episodes, 'success_num': success_count, 'success_rate': success_rate, 'steps': total_steps/25}]
    with open(folder_path+'/result.txt', 'w') as f:
        for item in result:
            f.write(f"{item}\n")