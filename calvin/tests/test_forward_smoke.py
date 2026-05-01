"""GPU forward-pass smoke test.

For each (config, ckpt) main+ablation pair, push model+ckpt to CUDA, run
a single synthetic-batch ``forward(..., run_inference=True)`` call, and
sanity-check the output trajectory tensor.

This is a behavioural smoke check, not a correctness check — the inputs
are random tensors, so the trajectory the model returns is meaningless.
What we *are* validating: every conditional branch wired in by
Tasks 17/18/19 reaches the end of ``compute_trajectory`` without
shape/index errors. If any of {GAT block, lie sampling, euclidean DDPM
sampling} were broken by the refactor, this test fails.

Skipped automatically when CUDA is unavailable or ckpts are missing.
"""
import pytest
import torch

from _helpers import (
    CKPT_DIR, CONFIGS_DIR,
    build_namespace_from_config, build_model, load_ckpt_state_dict,
)

PAIRS = [
    ("lda_abc_d.yaml", "abc_gat_score_newloss_300k.pth"),
    ("ablation_no_gat_abc_d.yaml", "abc_nogat_600k.pth"),
    ("ablation_no_lie_abc_d.yaml", "abc_gat_ddpm_300k.pth"),
]


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _make_synthetic_batch(args, device):
    """Build a one-element batch in the same shapes online evaluation yields
    (see lda/eval/evaluate_model.py: rgb/pcd are stacked static+gripper cams,
    fake_trajectory is interpolation_length-1 zeros).
    """
    B = 1
    H, W = (int(x) for x in args.image_size.split(","))
    n_cam = 2
    n_joints = args.num_joints
    n_hist = args.num_history
    traj_len = args.interpolation_length - 1  # matches evaluate_model.py

    rgb = torch.rand(B, n_cam, 3, H, W, device=device)
    pcd = torch.rand(B, n_cam, 3, H, W, device=device)
    instr = torch.randn(B, 53, 512, device=device)
    curr_gripper = torch.zeros(B, n_hist, 8, device=device)
    # gripper pose is xyz + quaternion + gripper-open flag — set a plausible
    # identity quaternion in the configured layout so convert_rot (euclidean
    # path) can find a valid rotation to convert.
    if args.quaternion_format == "wxyz":
        curr_gripper[..., 3] = 1.0
    else:  # xyzw
        curr_gripper[..., 6] = 1.0
    trajectory_mask = torch.zeros(B, traj_len, dtype=torch.bool, device=device)
    # The forward path needs a non-None gt_trajectory for .device access in
    # the GAT block, even at inference. evaluate_model.py uses an all-zeros
    # action_dim=7 placeholder; mirror that.
    fake_trajectory = torch.zeros(B, traj_len, 7, device=device)

    # GAT planner consumes a parallel `sample` dict at inference time.
    sample = {
        "rgbs": rgb,
        "pcds": pcd,
        "instr": instr,
        "joints_coords": torch.zeros(B, n_joints, 7, device=device),
    }
    sample["joints_coords"][..., 3] = 1.0  # identity quat per joint
    return fake_trajectory, rgb, pcd, instr, curr_gripper, trajectory_mask, sample


@pytest.mark.parametrize("cfg_name,ckpt_name", PAIRS)
def test_forward_inference_runs(cfg_name, ckpt_name, device):
    cfg_path = CONFIGS_DIR / cfg_name
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not downloaded: {ckpt_path}")

    args = build_namespace_from_config(cfg_path)
    model = build_model(args).to(device).eval()
    model.load_state_dict(load_ckpt_state_dict(ckpt_path), strict=False)

    fake_traj, rgb, pcd, instr, curr_gripper, traj_mask, sample = (
        _make_synthetic_batch(args, device)
    )

    with torch.no_grad():
        out = model(
            gt_trajectory=fake_traj,
            trajectory_mask=traj_mask,
            rgb_obs=rgb,
            pcd_obs=pcd,
            instruction=instr,
            curr_gripper=curr_gripper,
            run_inference=True,
            sample=sample,
        )

    if isinstance(out, tuple):
        out = out[0]
    # Predicted trajectory is (B, traj_len, ≥7): xyz + quat + (optional gripper).
    assert out.shape[0] == 1
    assert out.shape[1] == args.interpolation_length - 1
    assert out.shape[2] >= 7
    assert torch.isfinite(out).all(), (
        f"non-finite values in trajectory output for {cfg_name}"
    )
