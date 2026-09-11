# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""All hyper-parameters live here.

`Config` holds the defaults; a named function in `run_configs.py` returns a
`Config` with the overrides for one experiment, and `main.py` selects it by
name (``python -m main --config_name medcure``).
"""

import os
import inspect
import time


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


class Config:
    # ---------------------------------------------------------------- output
    output_root = "runs" + '/{}/'.format(datestr())

    # ------------------------------------------------------------ evaluation
    eval = False
    eval_batch_size = 256
    eval_num_workers = 3      # dataloader workers per eval task
    eval_steps = 100          # evaluate every N optimizer steps
    eval_start_step = 600     # skip evaluation before this step

    # ----------------------------------------------------------- image tower
    visual_name = "dinov2_vitb14"
    image_size = 378

    # ------------------------------------------------------------ text tower
    text_name = "emilyalsentzer/Bio_ClinicalBERT"
    max_bert_length = 256
    text_pooling = "eos"      # 'eos' | 'bos' | 'mean'

    # ------------------------------------------------------- projection head
    projection_head = "linear"  # 'linear' | 'mlp' | 'none'
    proj_dim = 512
    proj_dropout = 0.1

    # ---------------------------------------------------- contrastive target
    temperature = 0.01        # initial value of the learnable logit scale

    # -------------------------------------------------------------- warm-up
    # Before curation starts, train on this fraction of the pool with the plain
    # contrastive loss. Curation ranks pairs by `1 - cos(image, text)`, which
    # carries no signal while the two towers are still unaligned, so the model
    # needs a preliminary alignment first. 0 disables the warm-up.
    #
    # By design, and deliberately so: the slice gets a single pass, it shares
    # `lr` and the one global cosine schedule with the rest of the run, and it
    # is *not* removed from the pool that curation later scores.
    align_warmup_ratio = 0.05

    # ------------------------------------------------------------- curation
    # Fraction of each batch kept by the graph-density curation in the first
    # epoch. That selection is then frozen and reused for every later epoch.
    curation_ratio = 0.25
    n_neighbor = 8            # kNN degree of the batch graph
    gamma_forward = 0.5       # neighbour weight decay when scoring;  larger -> smaller re-weight
    gamma_reverse = 1.0       # neighbour penalty decay when sampling; larger -> smaller re-weight
    # How neighbour contributions are aggregated into the density score
    #   sum           : s_i + sum_j w_ij s_j
    #   mean          : s_i + mean_j(w_ij s_j)
    #   weighted_mean : s_i + sum_j w_ij s_j / sum_j w_ij
    graph_mode = "sum"
    graph_sampling_mode = "weighted"  # 'weighted' | 'absolute'

    # ------------------------------------------------------------- optimizer
    epochs = 10
    max_update = 5000000      # hard cap on optimizer steps
    batch_size = 80           # per GPU
    accum_iter = 1
    lr = 5e-5
    min_lr = 5e-6
    warmup_steps = 0
    weight_decay = 1e-4
    head_weight_decay = 0.0
    # Substrings of parameter names that use `head_weight_decay` instead of
    # `weight_decay`. Only the text projection is listed, and deliberately so:
    # the image tower's head is named `image_projection`, so it keeps the
    # regular `weight_decay`. That asymmetry is what every checkpoint so far was
    # trained with. Adding "image_projection" decays both heads alike, which
    # *changes* training -- don't do it to reproduce an existing run.
    head_weight_decay_modules = ("text_projection",)
    clip_grad = None
    fp16 = True

    # --------------------------------------------------------------- runtime
    nodes = 1
    num_workers = 8
    pin_mem = False
    seed = 0
    resume = None

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if not hasattr(self, "output_dir"):
            self.output_dir = inspect.stack()[1][3]
        self.output_dir = os.path.join(self.output_root, self.output_dir)
        print("config.output_dir =", self.output_dir)

    def add_cmd_args(self, cmd_args):
        for key, value in vars(cmd_args).items():
            if not key.startswith("__") and value is not None:
                setattr(self, key, value)
        return self

    def to_dict(self):
        """Every config value as a plain dict.

        Checkpoints store this rather than the object itself: it captures the
        class-level defaults as well as the overrides, and it carries no
        reference to any class or module, so a checkpoint stays loadable across
        refactors. `val_dataset` is skipped -- it holds live dataset objects.
        """
        skip = {'val_dataset'}
        return {k: getattr(self, k) for k in dir(self)
                if not k.startswith('_') and k not in skip and not callable(getattr(self, k))}

    def __str__(self):
        return "\n".join([f"{k}={v}" for k, v in sorted(self.to_dict().items())])
