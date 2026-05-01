"""Preprocess raw CALVIN episodes into the per-annotation .dat files the
training data loader consumes.

The original implementation is per-annotation linear: the ABC training set
has ~17 870 annotations and rendering each one through PyBullet end-to-end
takes hours of wall time. The work *between* annotations is independent
(disjoint episode ranges -> disjoint output files), so this script
parallelizes across a multiprocessing.Pool of size --num_workers, paying
the heavy CALVIN env startup once per worker.
"""
from typing import List, Optional, Tuple
from pathlib import Path
import multiprocessing as mp
import os
import pickle
import sys

import tap
import cv2
import numpy as np
import torch
import blosc
import pybullet as pb

from calvin_env.envs.play_table_env import get_env
from lda.utils.calvin import (
    keypoint_discovery,
    deproject,
    get_gripper_camera_view_matrix,
)


class Arguments(tap.Tap):
    traj_len: int = 16
    execute_every: int = 4
    save_path: str = './data/calvin/packaged_ABC_D_full'
    root_dir: str = 'calvin/dataset/task_ABC_D'
    mode: str = 'close_loop'  # [keypose, close_loop]
    tasks: Optional[List[str]] = None
    split: str = 'training'  # [training, validation]
    # Parallelism. 0 = single-process (matches original linear behaviour);
    # >0 = multiprocessing.Pool of that size. Defaults to a moderate fraction
    # of the host's cores so each worker has CPU + RAM headroom for the
    # PyBullet sim it owns.
    num_workers: int = max(1, (os.cpu_count() or 4) // 2)
    # Cap to the first N annotations for smoke runs / debugging. 0 = no cap.
    limit: int = 0


def make_env(dataset_path, split):
    val_folder = Path(dataset_path) / f"{split}"
    return get_env(val_folder, show_gui=False)


def process_datas(datas, mode, traj_len, execute_every, keyframe_inds):
    """Pack the per-step rendered datas into the on-disk state_dict layout.

    Returns the same 8-element list the original implementation produced:
    frame_ids, rgb_pcd tensor, action_tensors, camera_dicts,
    gripper_tensors, trajectories, annotation_ids, all_joint_coords.
    """
    h, w = datas['static_rgb'][0].shape[:2]
    datas['gripper_rgb'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        for m in datas['gripper_rgb']
    ]
    datas['gripper_pcd'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        for m in datas['gripper_pcd']
    ]
    static_rgb = np.stack(datas['static_rgb'], axis=0)
    static_pcd = np.stack(datas['static_pcd'], axis=0)
    gripper_rgb = np.stack(datas['gripper_rgb'], axis=0)
    gripper_pcd = np.stack(datas['gripper_pcd'], axis=0)
    rgb = np.stack([static_rgb, gripper_rgb], axis=1)
    pcd = np.stack([static_pcd, gripper_pcd], axis=1)
    rgb_pcd = np.stack([rgb, pcd], axis=2)
    rgb_pcd = rgb_pcd.transpose(0, 1, 2, 5, 3, 4)
    rgb_pcd = torch.as_tensor(rgb_pcd, dtype=torch.float32)

    keyframe_indices = torch.as_tensor(keyframe_inds)[None, :]
    gripper_indices = torch.arange(len(datas['proprios'])).view(-1, 1)
    action_indices = torch.argmax(
        (gripper_indices < keyframe_indices).float(), dim=1
    ).tolist()
    action_indices[-1] = len(keyframe_inds) - 1
    actions = [datas['proprios'][keyframe_inds[i]] for i in action_indices]
    action_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1) for a in actions
    ]

    camera_dicts = [{'front': (0, 0), 'wrist': (0, 0)}]

    gripper_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
        for a in datas['proprios']
    ]

    all_joint_coords = torch.as_tensor(
        np.stack(datas['all_joint_coords'], axis=0), dtype=torch.float32
    )

    if mode == 'keypose':
        trajectories = []
        for i in range(len(action_indices)):
            target_frame = keyframe_inds[action_indices[i]]
            current_frame = i
            trajectories.append(
                torch.cat(
                    [
                        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                        for a in datas['proprios'][current_frame:target_frame+1]
                    ],
                    dim=0
                )
            )
    else:
        trajectories = []
        for i in range(len(gripper_tensors)):
            traj = datas['proprios'][i:i+traj_len]
            if len(traj) < traj_len:
                traj += [traj[-1]] * (traj_len - len(traj))
            traj = [
                torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                for a in traj
            ]
            traj = torch.cat(traj, dim=0)
            trajectories.append(traj)

    if mode == 'keypose':
        keyframe_inds = [0] + keyframe_inds[:-1].tolist()
        keyframe_indices = torch.as_tensor(keyframe_inds)
        rgb_pcd = torch.index_select(rgb_pcd, 0, keyframe_indices)
        action_tensors = [action_tensors[i] for i in keyframe_inds]
        gripper_tensors = [gripper_tensors[i] for i in keyframe_inds]
        trajectories = [trajectories[i] for i in keyframe_inds]
        all_joint_coords = all_joint_coords[keyframe_indices]
    else:
        rgb_pcd = rgb_pcd[:-1]
        action_tensors = action_tensors[:-1]
        gripper_tensors = gripper_tensors[:-1]
        trajectories = trajectories[:-1]
        all_joint_coords = all_joint_coords[:-1]

        rgb_pcd = rgb_pcd[::execute_every]
        action_tensors = action_tensors[::execute_every]
        gripper_tensors = gripper_tensors[::execute_every]
        trajectories = trajectories[::execute_every]
        all_joint_coords = all_joint_coords[::execute_every]

    frame_ids = list(range(len(rgb_pcd)))

    return [
        frame_ids,
        rgb_pcd,
        action_tensors,
        camera_dicts,
        gripper_tensors,
        trajectories,
        datas['annotation_id'],
        all_joint_coords,
    ]


def load_episode(env, root_dir, split, episode, datas, ann_id):
    data = np.load(f'{root_dir}/{split}/{episode}')

    rgb_static = data['rgb_static']
    rgb_gripper = data['rgb_gripper']
    depth_static = data['depth_static']
    depth_gripper = data['depth_gripper']

    env.reset(robot_obs=data['robot_obs'], scene_obs=data['scene_obs'])
    static_cam = env.cameras[0]
    gripper_cam = env.cameras[1]
    gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)

    static_pcd = deproject(
        static_cam, depth_static,
        homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    static_pcd = np.reshape(
        static_pcd, (depth_static.shape[0], depth_static.shape[1], 3)
    )
    gripper_pcd = deproject(
        gripper_cam, depth_gripper,
        homogeneous=False, sanity_check=False
    ).transpose(1, 0)
    gripper_pcd = np.reshape(
        gripper_pcd, (depth_gripper.shape[0], depth_gripper.shape[1], 3)
    )

    rgb_static = rgb_static / 255. * 2 - 1
    rgb_gripper = rgb_gripper / 255. * 2 - 1

    proprio = np.concatenate([
        data['robot_obs'][:3],
        data['robot_obs'][3:6],
        (data['robot_obs'][[-1]] > 0).astype(np.float32)
    ], axis=-1)

    joint_indices = env.robot.arm_joint_ids
    joint_states = []
    for idx in joint_indices:
        link_state = pb.getLinkState(
            env.robot.robot_uid, idx, computeForwardKinematics=True
        )
        pos = link_state[0]
        orn = link_state[1]
        joint_states.append(np.concatenate([np.array(pos), np.array(orn)], axis=0))

    all_joint_coords = np.stack(joint_states, axis=0)

    datas['static_pcd'].append(static_pcd)
    datas['static_rgb'].append(rgb_static)
    datas['gripper_pcd'].append(gripper_pcd)
    datas['gripper_rgb'].append(rgb_gripper)
    datas['proprios'].append(proprio)
    datas['annotation_id'].append(ann_id)
    datas['all_joint_coords'].append(all_joint_coords)


def init_datas():
    return {
        'static_pcd': [],
        'static_rgb': [],
        'gripper_pcd': [],
        'gripper_rgb': [],
        'proprios': [],
        'annotation_id': [],
        'all_joint_coords': [],
    }


# Per-worker globals. Initialised once when each Pool worker boots so the
# heavy CALVIN sim is paid for once instead of per annotation.
_WORKER_ENV = None
_WORKER_CFG = None  # plain dict — tap.Tap can be awkward to pickle under spawn
_WORKER_SCENE_INFO = None


def _args_to_cfg(args) -> dict:
    """Drop tap.Tap args into a plain dict so spawn workers don't have to
    rehydrate the parser object."""
    return {
        "traj_len": args.traj_len,
        "execute_every": args.execute_every,
        "save_path": args.save_path,
        "root_dir": args.root_dir,
        "mode": args.mode,
        "tasks": list(args.tasks) if args.tasks is not None else None,
        "split": args.split,
    }


def _worker_init(cfg, scene_info):
    """Pool initializer: spin up the worker's own CALVIN env."""
    global _WORKER_ENV, _WORKER_CFG, _WORKER_SCENE_INFO
    _WORKER_CFG = cfg
    _WORKER_SCENE_INFO = scene_info
    _WORKER_ENV = make_env(cfg["root_dir"], cfg["split"])


def _scene_for_start(start_id, split, scene_info):
    if split != 'training' or scene_info is None:
        return 'D'
    if "calvin_scene_B" in scene_info and start_id <= scene_info["calvin_scene_B"][1]:
        return "B"
    if "calvin_scene_C" in scene_info and start_id <= scene_info["calvin_scene_C"][1]:
        return "C"
    if "calvin_scene_A" in scene_info and start_id <= scene_info["calvin_scene_A"][1]:
        return "A"
    return "D"


def _process_one_annotation(payload):
    """Worker entry point: render and save a single annotation's .dat."""
    anno_ind, start_id, end_id = payload
    cfg = _WORKER_CFG
    env = _WORKER_ENV

    datas = init_datas()
    for ep_id in range(start_id, end_id + 1):
        episode = 'episode_{:07d}.npz'.format(ep_id)
        load_episode(env, cfg["root_dir"], cfg["split"], episode, datas, anno_ind)

    _, keyframe_inds = keypoint_discovery(datas['proprios'])
    state_dict = process_datas(
        datas, cfg["mode"], cfg["traj_len"], cfg["execute_every"], keyframe_inds
    )

    scene = _scene_for_start(start_id, cfg["split"], _WORKER_SCENE_INFO)
    ep_save_path = f'{cfg["save_path"]}/{cfg["split"]}/{scene}+0/ann_{anno_ind}.dat'
    os.makedirs(os.path.dirname(ep_save_path), exist_ok=True)
    with open(ep_save_path, "wb") as f:
        f.write(blosc.compress(pickle.dumps(state_dict)))
    return anno_ind


def _enumerate_annotations(args) -> List[Tuple[int, int, int]]:
    """Read auto_lang_ann.npy, filter by --tasks, return (anno_ind, start, end)."""
    annotations = np.load(
        f'{args.root_dir}/{args.split}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True,
    ).item()
    out: List[Tuple[int, int, int]] = []
    for anno_ind, (start_id, end_id) in enumerate(annotations['info']['indx']):
        if args.tasks is not None and annotations['language']['task'][anno_ind] not in args.tasks:
            continue
        out.append((anno_ind, int(start_id), int(end_id)))
    return out


def _load_scene_info(args):
    if args.split != 'training':
        return None
    return np.load(
        f'{args.root_dir}/training/scene_info.npy',
        allow_pickle=True,
    ).item()


def main(args):
    """
    CALVIN contains long videos of "tasks" executed in order with noisy
    transitions between them. The 'annotations' npy lists the indices that
    segment those videos into per-task slices; each slice becomes one
    .dat output here.
    """
    work = _enumerate_annotations(args)
    if args.limit > 0:
        work = work[: args.limit]
    scene_info = _load_scene_info(args)
    cfg = _args_to_cfg(args)
    print(
        f'package_calvin: split={args.split} num_annotations={len(work)} '
        f'num_workers={args.num_workers}'
    )

    if args.num_workers <= 1:
        # Single-process path — keeps the original linear behaviour for hosts
        # where multiprocessing is unwelcome (e.g. sub-1-CPU containers).
        _worker_init(cfg, scene_info)
        for i, payload in enumerate(work):
            _process_one_annotation(payload)
            if (i + 1) % 50 == 0 or i + 1 == len(work):
                print(f'  [{i + 1}/{len(work)}] done')
        return

    ctx = mp.get_context('spawn')  # PyBullet doesn't survive fork-with-state.
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(cfg, scene_info),
    ) as pool:
        completed = 0
        for _ in pool.imap_unordered(_process_one_annotation, work, chunksize=1):
            completed += 1
            if completed % 50 == 0 or completed == len(work):
                print(f'  [{completed}/{len(work)}] done')


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
