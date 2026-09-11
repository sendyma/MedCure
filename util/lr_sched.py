# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math


def adjust_step_learning_rate(optimizer, step, lr, min_lr, warmup_steps, max_update):
    """Half-cycle cosine decay after a linear warmup, stepped per iteration."""
    if step < warmup_steps:
        lr = lr * step / warmup_steps
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (step - warmup_steps) / (max_update - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
