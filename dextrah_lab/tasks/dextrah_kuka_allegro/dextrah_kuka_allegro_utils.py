# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Imports
import torch

def assert_equals(a, b) -> None:
    # Saves space typing out the full assert and text
    assert a == b, f"{a} != {b}"

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def compute_absolute_action(
    raw_actions: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    N, D = raw_actions.shape
    assert_equals(lower_limits.shape, (D,))
    assert_equals(upper_limits.shape, (D,))

    # Apply actions to hand
    absolute_action = scale(
        x=raw_actions,
        lower=lower_limits,
        upper=upper_limits,
    )
    absolute_action = tensor_clamp(
        t=absolute_action,
        min_t=lower_limits,
        max_t=upper_limits,
    )

    return absolute_action

@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)
