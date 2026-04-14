import cv2
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

IMG_H         = 80
IMG_W         = 160
N_FRAMES      = 4
SEG_ROAD      = 1
SEG_ROADLINE  = 6
N_MAJOR       = 6
ROUTE_VEC_DIM = N_MAJOR + 2  # 8: one-hot(6) + cos_bearing + sin_bearing


class ImagePreprocessor:
    def to_gray(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def stack_frames(self, frame_deque) -> np.ndarray:
        return np.stack(list(frame_deque), axis=0)  # (N_FRAMES, H, W) channel-first

    def road_stats(self, seg: np.ndarray):
        h, w          = seg.shape
        roi           = seg[int(h * 0.4):, :]
        roi_h, roi_w  = roi.shape
        total         = roi_h * roi_w
        road_mask_roi = (roi == SEG_ROAD) | (roi == SEG_ROADLINE)
        road_pixels   = int(np.sum(road_mask_roi))
        road_ratio    = road_pixels / max(total, 1)
        if road_pixels == 0:
            offset = 0.5
        else:
            col_indices = np.where(road_mask_roi)
            road_cx     = float(np.mean(col_indices[1]))
            offset      = abs(road_cx - roi_w / 2.0) / roi_w
        full_mask = (seg == SEG_ROAD) | (seg == SEG_ROADLINE)
        return road_ratio, offset, full_mask

    def debug_overlay(self, rgb, seg, speed_kmh, cp_idx, n_cp, runs, next_cp_name=""):
        vis     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = np.zeros_like(vis)
        overlay[seg == SEG_ROAD]     = [0, 180,   0]
        overlay[seg == SEG_ROADLINE] = [0, 220, 220]
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        hud = f"spd={speed_kmh:.0f}km/h  cp={cp_idx}/{n_cp}  next={next_cp_name}  loops={runs}"
        cv2.putText(vis, hud, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        cv2.imshow("AutoCar | Camera RL", vis)
        cv2.waitKey(1)


class CnnFeatureExtractor(BaseFeaturesExtractor):
    """
    Dict obs extractor for SB3 MultiInputPolicy.

      obs["image"]     : (N_FRAMES, IMG_H, IMG_W) uint8  — channel-first
      obs["route_vec"] : (ROUTE_VEC_DIM,) float32

    SB3 with DummyVecEnv delivers image as (B, C, H, W) already in forward().
    obs space is declared channel-first so shape = (N_FRAMES, H, W):
      shape[0] = N_FRAMES = C = 4   <- n_channels for Conv2d
      shape[1] = IMG_H    = 80
      shape[2] = IMG_W    = 160

    Analytical flat size after 3 convs on H=80, W=160:
      H: 80 -> 40 -> 20 -> 10
      W: 160 -> 80 -> 40 -> 20
      64 * 10 * 20 = 12800

    CNN branch   : Conv(4->32)->Conv(32->64)->Conv(64->64)->Linear(256) -> (B,256)
    Route branch : Linear(8->64)->Linear(64->32)                        -> (B, 32)
    Output       : cat                                                   -> (B,288)
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 288):
        super().__init__(observation_space, features_dim)

        img_shape  = observation_space["image"].shape   # (N_FRAMES, H, W)
        n_channels = img_shape[0]                       # 4  <- C, NOT img_shape[2]
        h_in       = img_shape[1]                       # 80
        w_in       = img_shape[2]                       # 160
        route_dim  = observation_space["route_vec"].shape[0]  # 8

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        def _conv_out(s, k, stride, p):
            return (s + 2 * p - k) // stride + 1

        h_out    = _conv_out(_conv_out(_conv_out(h_in, 5, 2, 2), 3, 2, 1), 3, 2, 1)
        w_out    = _conv_out(_conv_out(_conv_out(w_in, 5, 2, 2), 3, 2, 1), 3, 2, 1)
        cnn_flat = 64 * h_out * w_out  # 64 * 10 * 20 = 12800

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_flat, 256),
            nn.ReLU(),
        )

        self.route_mlp = nn.Sequential(
            nn.Linear(route_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        assert 256 + 32 == features_dim, f"features_dim must be 288, got {features_dim}"

    def forward(self, observations: dict) -> torch.Tensor:
        img = observations["image"]

        # Normalise: DummyVecEnv delivers float32 in [0,255]; raw env gives uint8.
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.max() > 1.0:
            img = img / 255.0

        # img is already (B, C, H, W) — no permute needed.
        img_feat   = self.cnn_linear(self.cnn(img))             # (B, 256)
        route_feat = self.route_mlp(observations["route_vec"])  # (B,  32)
        return torch.cat([img_feat, route_feat], dim=1)         # (B, 288)