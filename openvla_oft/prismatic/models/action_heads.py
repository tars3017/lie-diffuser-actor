"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


# =============================================================================
# SO(3) helper functions (pure PyTorch, no external Lie group library needed)
# =============================================================================

def so3_exp_map(v: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' formula: axis-angle vector -> rotation matrix.
    v: (..., 3) axis-angle vectors (tangent space of SO(3))
    Returns: (..., 3, 3) rotation matrices
    """
    batch_shape = v.shape[:-1]
    v_flat = v.reshape(-1, 3).float()
    N = v_flat.shape[0]
    device = v_flat.device

    vx, vy, vz = v_flat[:, 0], v_flat[:, 1], v_flat[:, 2]
    zeros = torch.zeros(N, device=device)
    # Skew-symmetric matrix of v
    K = torch.stack([
        zeros, -vz, vy,
        vz, zeros, -vx,
        -vy, vx, zeros
    ], dim=-1).reshape(N, 3, 3)

    theta = v_flat.norm(dim=-1, keepdim=True)  # (N, 1)
    theta3 = theta.reshape(N, 1, 1)
    eps = 1e-8

    # sin(theta)/theta -> 1 as theta->0
    sin_coeff = torch.where(
        theta3 > eps,
        theta3.sin() / theta3.clamp(min=eps),
        torch.ones_like(theta3),
    )
    # (1 - cos(theta))/theta^2 -> 0.5 as theta->0
    cos_coeff = torch.where(
        (theta3 ** 2) > eps ** 2,
        (1 - theta3.cos()) / (theta3 ** 2).clamp(min=eps ** 2),
        0.5 * torch.ones_like(theta3),
    )

    I = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
    R = I + sin_coeff * K + cos_coeff * (K @ K)
    return R.reshape(*batch_shape, 3, 3)


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) log map: rotation matrix -> axis-angle vector (tangent space).
    R: (..., 3, 3) rotation matrices
    Returns: (..., 3) axis-angle vectors
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3).float()
    N = R_flat.shape[0]
    device = R_flat.device

    # theta = arccos((trace(R) - 1) / 2)
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]  # (N,)
    cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = cos_theta.acos()  # (N,)

    # Skew part: (R - R^T) / 2 = sin(theta) * K_hat
    # [K_hat[2,1], K_hat[0,2], K_hat[1,0]] = unit axis components
    skew = (R_flat - R_flat.transpose(-1, -2)) / 2  # (N, 3, 3)
    v_unnorm = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)  # (N, 3)

    # Factor: theta / sin(theta) -> 1 as theta -> 0
    eps = 1e-6
    factor = torch.where(
        theta > eps,
        theta / theta.sin().clamp(min=eps),
        torch.ones_like(theta),
    )  # (N,)
    result = factor.unsqueeze(-1) * v_unnorm  # (N, 3)
    return result.reshape(*batch_shape, 3)


def so3_add_noise(rot_tan: torch.Tensor, z_rot: torch.Tensor, sqrt_alpha_t: torch.Tensor) -> torch.Tensor:
    """
    SO(3) forward diffusion: r_t = log_map(exp_map(r_0) @ exp_map(sqrt_alpha * z)).
    rot_tan: (N, 3) clean rotations as axis-angle (tangent space)
    z_rot: (N, 3) Gaussian noise in tangent space
    sqrt_alpha_t: (N, 1) noise level per sample
    Returns: (N, 3) noisy rotations as axis-angle (tangent space)
    """
    R0 = so3_exp_map(rot_tan)                        # (N, 3, 3)
    R_noise = so3_exp_map(sqrt_alpha_t * z_rot)      # (N, 3, 3)
    Rt = R0 @ R_noise                                # (N, 3, 3) — SO(3) composition
    return so3_log_map(Rt)                           # (N, 3)


def so3_compose_step(rot_tan: torch.Tensor, update_tan: torch.Tensor) -> torch.Tensor:
    """
    Riemannian update step: r_out = log_map(exp_map(r) @ exp_map(update)).
    rot_tan: (N, 3) current rotations in tangent space
    update_tan: (N, 3) update in tangent space
    Returns: (N, 3) updated rotations in tangent space
    """
    Rt = so3_exp_map(rot_tan)         # (N, 3, 3)
    R_upd = so3_exp_map(update_tan)   # (N, 3, 3)
    R_next = Rt @ R_upd               # (N, 3, 3)
    return so3_log_map(R_next)        # (N, 3)


def chordal_distance_tan(pred_tan: torch.Tensor, target_tan: torch.Tensor) -> torch.Tensor:
    """
    Chordal (Frobenius) distance between rotations given as axis-angle tangent vectors.
    pred_tan: (N, 3)
    target_tan: (N, 3)
    Returns: (N,) distances
    """
    m1 = so3_exp_map(pred_tan)    # (N, 3, 3)
    m2 = so3_exp_map(target_tan)  # (N, 3, 3)
    return ((m1 - m2) ** 2).sum(dim=[-1, -2])  # (N,)


def so3_random_uniform(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Sample n rotation matrices uniformly from SO(3) and return as tangent vectors.
    Uses QR decomposition of random Gaussian matrices (Haar measure).
    Returns: (n, 3) axis-angle tangent vectors
    """
    X = torch.randn(n, 3, 3, device=device, dtype=torch.float32)
    Q, R_qr = torch.linalg.qr(X)
    # Ensure det(Q) = +1
    d = torch.diagonal(R_qr, dim1=-2, dim2=-1).sign()
    Q = Q * d.unsqueeze(-2)
    return so3_log_map(Q).to(dtype)  # (n, 3)


# =============================================================================
# SE(3) helper functions (pure PyTorch, no external Lie group library needed)
# =============================================================================

def _build_skew(v: torch.Tensor) -> torch.Tensor:
    """Build skew-symmetric matrices from (N, 3) vectors. Returns (N, 3, 3)."""
    N = v.shape[0]
    device = v.device
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    zeros = torch.zeros(N, device=device)
    K = torch.stack([
        zeros, -vz, vy,
        vz, zeros, -vx,
        -vy, vx, zeros,
    ], dim=-1).reshape(N, 3, 3)
    return K


def se3_exp_map(xi: torch.Tensor):
    """
    SE(3) exponential map: se(3) tangent -> (R, t).
    xi: (..., 6) tangent vectors [v(3) | omega(3)]
      v: linear velocity (translation part)
      omega: angular velocity (rotation part)
    Returns:
      R: (..., 3, 3) rotation matrices
      t: (..., 3) translation vectors
    """
    batch_shape = xi.shape[:-1]
    xi_flat = xi.reshape(-1, 6).float()
    N = xi_flat.shape[0]
    device = xi_flat.device

    v = xi_flat[:, :3]      # (N, 3)
    omega = xi_flat[:, 3:]  # (N, 3)

    K = _build_skew(omega)                       # (N, 3, 3)
    theta = omega.norm(dim=-1, keepdim=True)     # (N, 1)
    theta3 = theta.reshape(N, 1, 1)
    eps = 1e-8

    # sin(θ)/θ  →  1 as θ→0
    sin_coeff = torch.where(
        theta3 > eps,
        theta3.sin() / theta3.clamp(min=eps),
        torch.ones_like(theta3),
    )
    # (1 - cos(θ))/θ²  →  0.5 as θ→0
    cos_coeff = torch.where(
        (theta3 ** 2) > eps ** 2,
        (1 - theta3.cos()) / (theta3 ** 2).clamp(min=eps ** 2),
        0.5 * torch.ones_like(theta3),
    )
    # (θ - sin(θ))/θ³  →  1/6 as θ→0
    cubic_coeff = torch.where(
        (theta3 ** 3) > eps ** 3,
        (theta3 - theta3.sin()) / (theta3 ** 3).clamp(min=eps ** 3),
        (1.0 / 6.0) * torch.ones_like(theta3),
    )

    I = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
    K2 = K @ K

    R = I + sin_coeff * K + cos_coeff * K2                         # (N, 3, 3)
    V = I + cos_coeff * K + cubic_coeff * K2                       # (N, 3, 3) left Jacobian
    t = (V @ v.unsqueeze(-1)).squeeze(-1)                          # (N, 3)

    return R.reshape(*batch_shape, 3, 3), t.reshape(*batch_shape, 3)


def se3_log_map(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    SE(3) logarithmic map: (R, t) -> se(3) tangent vector.
    R: (..., 3, 3) rotation matrices
    t: (..., 3) translation vectors
    Returns: (..., 6) tangent vectors [v(3) | omega(3)]
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3).float()
    t_flat = t.reshape(-1, 3).float()
    N = R_flat.shape[0]
    device = R_flat.device

    omega = so3_log_map(R_flat)      # (N, 3)
    K = _build_skew(omega)           # (N, 3, 3)
    theta = omega.norm(dim=-1)       # (N,)
    theta3 = theta.reshape(N, 1, 1)
    eps = 1e-6

    cos_t = theta3.cos()
    sin_t = theta3.sin()

    # c2 = 1/θ² - (1 + cos θ) / (2 θ sin θ)  →  1/12 as θ→0
    c2 = torch.where(
        theta3 > eps,
        1.0 / (theta3 ** 2).clamp(min=eps ** 2)
        - (1 + cos_t) / (2 * theta3 * sin_t.clamp(min=eps)),
        (1.0 / 12.0) * torch.ones_like(theta3),
    )

    I = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
    V_inv = I - 0.5 * K + c2 * (K @ K)                            # (N, 3, 3)
    v = (V_inv @ t_flat.unsqueeze(-1)).squeeze(-1)                 # (N, 3)

    xi = torch.cat([v, omega], dim=-1)                             # (N, 6)
    return xi.reshape(*batch_shape, 6)


def se3_compose(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor):
    """SE(3) composition: (R1,t1) @ (R2,t2) = (R1@R2, R1@t2 + t1)."""
    R_out = R1 @ R2
    t_out = (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_out, t_out


def se3_add_noise(xi0: torch.Tensor, z: torch.Tensor, sqrt_alpha_t: torch.Tensor) -> torch.Tensor:
    """
    SE(3) forward diffusion: xi_t = log_{SE3}(exp_{SE3}(xi_0) @ exp_{SE3}(sqrt_alpha * z))
    xi0: (N, 6) clean poses as se(3) tangent [v | omega]
    z: (N, 6) Gaussian noise in tangent space
    sqrt_alpha_t: (N, 1) noise level per sample
    Returns: (N, 6) noisy poses as se(3) tangent vectors
    """
    R0, t0 = se3_exp_map(xi0)
    R_noise, t_noise = se3_exp_map(sqrt_alpha_t * z)
    Rt, tt = se3_compose(R0, t0, R_noise, t_noise)
    return se3_log_map(Rt, tt)


def se3_compose_step(xi: torch.Tensor, update_xi: torch.Tensor) -> torch.Tensor:
    """
    Riemannian update on SE(3): xi_out = log_{SE3}(exp_{SE3}(xi) @ exp_{SE3}(update_xi))
    xi: (N, 6), update_xi: (N, 6)
    Returns: (N, 6)
    """
    R, t = se3_exp_map(xi)
    R_upd, t_upd = se3_exp_map(update_xi)
    R_next, t_next = se3_compose(R, t, R_upd, t_upd)
    return se3_log_map(R_next, t_next)


def se3_random_tangent(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Sample n random SE(3) priors as se(3) tangent vectors [v | omega].
    Translation v: N(0, 1); Rotation omega: uniform SO(3) via QR.
    Returns: (n, 6) tangent vectors
    """
    v = torch.randn(n, 3, device=device, dtype=torch.float32)
    omega = so3_random_uniform(n, device=device, dtype=torch.float32)
    return torch.cat([v, omega], dim=-1).to(dtype)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred


# =============================================================================
# SE(3) Score Matching Action Head
# =============================================================================

class SE3PowerNoiseSchedule:
    """
    Power-law variance-exploding noise schedule for SE(3) score matching.
    Adapted from liepose_pytorch (Mohlin et al., CVPR 2024).
    """

    def __init__(self, alpha_start: float = 1e-8, alpha_end: float = 1.0, timesteps: int = 100, power: float = 3.0):
        # Match CoG: create timesteps+1 values; training uses indices 0..timesteps-1
        alphas = np.linspace(alpha_start ** (1 / power), alpha_end ** (1 / power), timesteps + 1) ** power
        self.sqrt_alphas = np.sqrt(alphas).astype(np.float32)  # shape: (T+1,)
        self.timesteps = timesteps
        self.alpha_start = alpha_start


class SE3ScorePredictionModel(nn.Module):
    """
    Score prediction model for SE(3) score matching.
    Takes VLA action token hidden states and predicts scores for all action dims.
    """

    def __init__(self, transformer_hidden_dim: int, hidden_dim: int, action_dim: int = 7):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim + hidden_dim + hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(self, obs, noisy_emb, time_emb):
        # obs: (batch_size, chunk_len, action_dim * hidden_dim)
        # noisy_emb & time_emb: (batch_size, chunk_len, hidden_dim)
        x = torch.cat([obs, noisy_emb, time_emb], dim=-1)
        return self.mlp_resnet(x)


class SE3ScoreMatchingActionHead(nn.Module):
    """
    Score matching action head for robot manipulation, with switchable diffusion geometry.

    Two modes selected by `lie_group`:

      lie_group=True  (default; matches the released openvla-oft-lie-lora-150000 ckpt):
        Dims 0:6 treated as an SE(3) tangent [v(3) | omega(3)] and diffused jointly on
        the SE(3) Riemannian manifold (see Mohlin et al. CVPR 2024, DiffuserActor / Ze
        et al. 2024).
        Forward: xi_t = log_{SE3}(exp_{SE3}(xi_0) @ exp_{SE3}(sqrt_alpha * z))

      lie_group=False (matches the openvla-7b-euclidean-finetuned-libero-10-lora ckpt):
        Dims 0:6 treated as a flat R^6 vector with additive Gaussian noise. This is the
        Euclidean ablation used to isolate the contribution of Lie-space diffusion.
        Forward: xi_t = xi_0 + sqrt_alpha * z

    In both modes:
      - Dim 6 (gripper) uses Euclidean Gaussian diffusion + MSE score loss.
      - Score matching target: ta = -z (negative noise in tangent / R^6 space).
      - Loss: weighted MSE — translation * 20, rotation * 10 (DiffuserActor convention).

    The flag controls runtime math only; class attributes and parameter shapes are
    identical between modes, so a single state_dict can correspond to either training
    recipe (verified by tests/test_lie_group_flag.py).
    """

    is_se3_score_matching = True  # used for detection in predict_action

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        num_steps_train: int = 100,
        epsilon: float = 1e-8,  # Langevin step size coefficient (step = epsilon * (sigma_t/sigma_L)^2)
        lie_group: bool = True,
    ):
        super().__init__()
        self.is_se3_score_matching = True
        self.action_dim = action_dim
        self.noise_schedule = SE3PowerNoiseSchedule(timesteps=num_steps_train)
        self.score_predictor = SE3ScorePredictionModel(
            transformer_hidden_dim=hidden_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )
        self.noisy_action_projector = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        self.epsilon = epsilon
        self.num_inference_steps = num_steps_train  # can be overridden for faster inference
        # Plain Python attribute, NOT a Buffer/Parameter — must not enter state_dict.
        self.lie_group = lie_group

    def sample_noisy_actions(self, ground_truth_actions: torch.Tensor) -> dict:
        """
        Forward diffusion process.

        lie_group=True : SE(3) Riemannian on dims 0:6 (xi_t = log(exp(xi_0) @ exp(sqrt_alpha * z))).
        lie_group=False: Euclidean additive on dims 0:6 (xi_t = xi_0 + sqrt_alpha * z).
        Gripper (dim 6) is always Euclidean.

        ground_truth_actions: (B, chunk_len, action_dim) normalized to [-1, 1]
        Returns dict with: noise_se3, noise_grip, noisy_actions, diffusion_timestep_embeddings
        """
        B, T, _ = ground_truth_actions.shape
        device = ground_truth_actions.device
        dtype = ground_truth_actions.dtype

        xi0 = ground_truth_actions[..., 0:6]   # (B, T, 6) translation + rotation
        grip = ground_truth_actions[..., 6:7]  # (B, T, 1) gripper

        # Sample random timesteps (one per batch element)
        timesteps = torch.randint(0, self.noise_schedule.timesteps, (B,), device=device)

        # Sample Gaussian noise (interpreted as SE(3) tangent if lie_group else as flat R^6)
        z_se3 = torch.randn(B, T, 6, device=device, dtype=dtype)
        z_grip = torch.randn(B, T, 1, device=device, dtype=dtype)

        # sqrt_alpha: (B,) -> (B, 1, 1)
        sqrt_alpha_np = self.noise_schedule.sqrt_alphas[timesteps.cpu().numpy()]
        sqrt_alpha = torch.tensor(sqrt_alpha_np, device=device, dtype=dtype).reshape(B, 1, 1)

        # Forward diffusion for dims 0:6 — branch on geometry.
        if self.lie_group:
            xi0_flat = xi0.reshape(B * T, 6)
            z_se3_flat = z_se3.reshape(B * T, 6)
            sqrt_alpha_flat = sqrt_alpha.expand(B, T, 1).reshape(B * T, 1)
            noisy_xi_flat = se3_add_noise(xi0_flat, z_se3_flat, sqrt_alpha_flat)
            noisy_xi = noisy_xi_flat.to(dtype).reshape(B, T, 6)
        else:
            noisy_xi = xi0 + sqrt_alpha * z_se3

        # Euclidean forward diffusion for gripper (always)
        noisy_grip = grip + sqrt_alpha * z_grip

        # Combine: [noisy 0:6 | noisy_grip]
        noisy_actions = torch.cat([noisy_xi, noisy_grip], dim=-1)

        # Timestep embeddings: (B, hidden_dim) -> (B, 1, hidden_dim)
        diffusion_timestep_embeddings = (
            self.time_encoder(timesteps.float()).to(dtype).to(device).unsqueeze(1)
        )

        return dict(
            noise_se3=z_se3,
            noise_grip=z_grip,
            noisy_actions=noisy_actions,
            timesteps=timesteps,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

    def predict_score(self, actions_hidden_states: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Predict SE(3) scores from VLA action token hidden states, noisy actions, and timesteps.
        """
        B, T, _ = noisy_actions.shape
        rearranged = actions_hidden_states.reshape(B, NUM_ACTIONS_CHUNK, -1)
        
        noisy_emb = self.noisy_action_projector(noisy_actions.to(actions_hidden_states.dtype))
        time_emb = self.time_encoder(timesteps.float()).unsqueeze(1).expand(-1, T, -1).to(actions_hidden_states.dtype)
        
        return self.score_predictor(rearranged, noisy_emb, time_emb)
