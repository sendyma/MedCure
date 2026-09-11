# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import datetime
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import wandb
from PIL import Image

import util.misc as misc
from engine import WARMUP_EPOCH, evaluate, train_one_epoch
from model import CXRClip
from util.funs import cleanup_distributed, seed_everything
from util.misc import NativeScalerWithGradNormCount as NativeScaler


# CLIP's normalisation statistics
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def build_transform(args):
    return transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def build_dataset(args, tokenizer):
    """Build the pre-training set and register every zero-shot eval set on `args`."""
    from evaluation import datasets, metrics

    transform = build_transform(args)
    train_dataset = datasets.MultimodalPretrainingDataset(
        split='train', transform=transform, max_bert_length=args.max_bert_length, tokenizer=tokenizer)

    catalog, all_labels = metrics.load_metadata()
    args.val_dataset = {
        d: datasets.get_downstream_dataset(catalog, d, is_train=False, transform=transform)
        for d in catalog
    }
    return train_dataset


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # snapshot the code next to the checkpoints so a run stays reproducible
    source_files = [
        'configs.py', 'constants.py', 'engine.py', 'main.py',
        'model.py', 'projection.py', 'run_configs.py',
        'evaluation/datasets.py', 'evaluation/metrics.py',
        'evaluation/dataset_catalog.json', 'evaluation/labels.json',
    ]
    for src_file in source_files:
        if os.path.exists(src_file):
            shutil.copy(src_file, args.output_root)

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        args.distributed = True
        args.world_size = ngpus_per_node * args.nodes
        args.dist_url = 'tcp://127.0.0.1:' + str(12000 + np.random.randint(0, 1000))
        print(f"Starting distributed training on {ngpus_per_node} GPUs ({args.dist_url})")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        print("Starting single GPU training")
        main_worker(0, args)


def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu if not hasattr(args, 'rank') else args.rank
    torch.cuda.set_device(gpu)

    if args.distributed:
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        dist.barrier()
        print(f"Process {args.rank} initialized")

    seed_everything(args.seed)

    if args.rank == 0:
        wandb.init(project='MedCure')  # mode='disabled'

    model = CXRClip.from_config(args)
    model.to(torch.device(f'cuda:{gpu}'))
    tokenizer = model.text_encoder.tokenizer

    dataset_train = build_dataset(args, tokenizer=tokenizer)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff_batch_size = args.batch_size * args.accum_iter * args.world_size
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('image size: %d, max_bert_length: %d' % (args.image_size, args.max_bert_length))
    print('lr: %.2e, accum_iter: %d, effective batch size: %d' % (args.lr, args.accum_iter, eff_batch_size))
    print('len(dataset):', len(dataset_train))

    if args.distributed:
        # find_unused_parameters is required: BERT's pooler runs in the forward
        # pass but its output is discarded, so those weights get no gradient.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 1-d parameters (norms, biases, logit_scale) never get weight decay; every
    # other parameter does. Note that DINOv2's `pos_embed` and `cls_token` are
    # 3-d, so they are decayed like ordinary weights -- that is what the
    # existing checkpoints were trained with.
    p_wd, p_no_wd, p_head_wd = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim == 1:
            p_no_wd.append(p)
        elif any(part in n for part in args.head_weight_decay_modules):
            p_head_wd.append(p)
        else:
            p_wd.append(p)

    param_groups = [{"params": p_wd, "weight_decay": args.weight_decay},
                    {"params": p_no_wd, "weight_decay": 0.}]

    if p_head_wd:
        param_groups.append({"params": p_head_wd, "weight_decay": args.head_weight_decay})

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8)

    loss_scaler = NativeScaler(args.fp16)

    start_epoch, best_acc, step, resumed_subset = 0, [0.], [0], set()
    if args.resume:
        if args.resume.startswith("checkpoint"):
            args.resume = os.path.join(args.output_dir, args.resume)
        start_epoch, best_acc, step, resumed_subset = misc.load_model(
            args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        best_acc, step = [best_acc], [step if step is not None else 0]
        print("resuming", args.resume, "from step", step[0], "with best_acc", best_acc[0])

    if args.eval:
        metric = evaluate(args, model_without_ddp, tokenizer)
        json_str = json.dumps({"step": step[0], "acc": metric, "seen": eff_batch_size * step[0]})
        print(json_str)
        exit(0)

    args.full_dataset_size = len(dataset_train)
    curation_ratio = args.curation_ratio

    def make_loader(dataset, shuffle=True):
        sampler = None
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=shuffle)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=args.num_workers,
            collate_fn=getattr(dataset_train, "collate_fn", None),
            pin_memory=args.pin_mem,
            sampler=sampler,
            drop_last=True  # keep the batch size identical on every rank
        )
        return loader, sampler

    # Stage 0: warm up cross-modal alignment on a small slice of the pool.
    # One pass, same lr as the rest of the run, and the slice stays in the pool
    # that stage 1 curates over -- see `align_warmup_ratio` in configs.py.
    steps_warmup, data_loader_warmup, sampler_warmup = 0, None, None
    if args.align_warmup_ratio and start_epoch == 0:
        n_warmup = int(args.align_warmup_ratio * len(dataset_train))
        # A CPU generator with a fixed seed gives every rank the same slice.
        order = torch.randperm(len(dataset_train), generator=torch.Generator().manual_seed(args.seed))
        warmup_ids = sorted(order[:n_warmup].tolist())
        data_loader_warmup, sampler_warmup = make_loader(torch.utils.data.Subset(dataset_train, warmup_ids))
        steps_warmup = len(data_loader_warmup)

    # Stage 1: curate + pre-train over the full pool.
    data_loader_full, sampler_full = make_loader(dataset_train)

    global_example_ids = set()
    if resumed_subset and start_epoch >= 1:  # resuming past stage 1: the subset is already fixed
        global_example_ids.update(resumed_subset)

    def freeze_subset():
        """Freeze the curated samples into a Subset, identical on every rank."""
        ids = sorted(global_example_ids)
        if args.distributed:
            payload = [ids]
            dist.broadcast_object_list(payload, src=0)
            ids = payload[0]
            global_example_ids.clear()
            global_example_ids.update(ids)
        loader, sampler = make_loader(torch.utils.data.Subset(dataset_train, ids))
        if args.rank == 0:
            print('subset fixed: %d / %d = %.4f, %d steps/epoch' % (
                len(ids), args.full_dataset_size, len(ids) / args.full_dataset_size, len(loader)))
        return loader, sampler

    data_loader_subset, sampler_subset = (freeze_subset() if global_example_ids else (None, None))

    # Horizon for the LR cosine: stage 1 walks the full pool, later epochs the
    # frozen subset. Curation keeps exactly int(ratio * batch * world_size) pairs
    # per step with no repeats, so the subset size is known up front.
    steps_curate = len(data_loader_full)
    global_batch = args.batch_size * args.world_size
    steps_subset = (steps_curate * int(curation_ratio * global_batch)) // global_batch
    total_steps = steps_warmup + steps_curate + (args.epochs - 1) * steps_subset
    if args.rank == 0:
        print('planned steps: warm-up %d + curation epoch %d + %d x subset epoch %d = %d' % (
            steps_warmup, steps_curate, args.epochs - 1, steps_subset, total_steps))

    start_time = time.time()

    if data_loader_warmup is not None:
        if args.distributed:
            sampler_warmup.set_epoch(0)
        train_one_epoch(
            model, model_without_ddp, tokenizer, data_loader_warmup, best_acc,
            optimizer, torch.device(f'cuda:{gpu}'), WARMUP_EPOCH, step, loss_scaler, eff_batch_size,
            args.clip_grad, global_example_ids,
            curation_ratio=None,
            total_steps=total_steps,
            args=args
        )

    for epoch in range(start_epoch, args.epochs):

        if step[0] >= args.max_update:
            if args.rank == 0:
                print(f"Reached max steps ({args.max_update}), terminating training")
            break

        # Curate while no subset exists yet (stage 1); afterwards train on it.
        curating = data_loader_subset is None
        data_loader_epoch = data_loader_full if curating else data_loader_subset
        sampler_epoch = sampler_full if curating else sampler_subset
        if args.distributed:
            sampler_epoch.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, model_without_ddp, tokenizer, data_loader_epoch, best_acc,
            optimizer, torch.device(f'cuda:{gpu}'), epoch, step, loss_scaler, eff_batch_size,
            args.clip_grad, global_example_ids,
            curation_ratio=(curation_ratio if curating else None),
            total_steps=total_steps,
            args=args
        )

        if curating:  # stage 1 done -- freeze the subset
            data_loader_subset, sampler_subset = freeze_subset()

        if args.rank == 0:
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                global_example_ids=global_example_ids, loss_scaler=loss_scaler,
                epoch=epoch, epoch_name="last", best_acc=best_acc[0], step=step[0])

    metric = evaluate(args, model_without_ddp, tokenizer)
    if args.rank == 0:
        with open(os.path.join(args.output_dir, 'subset.json'), 'w') as f_subset:
            for item in global_example_ids:
                f_subset.write(json.dumps(item) + '\n')
        print('final subset ratio:', len(global_example_ids) / len(dataset_train))

        json_str = json.dumps({"step": step[0], "acc": metric, "seen": eff_batch_size * step[0]})
        print(json_str)

        print('Training time {}'.format(datetime.timedelta(seconds=int(time.time() - start_time))))

    cleanup_distributed()


def parse_args():
    '''All hyper-parameters come from a named config; only these few are CLI flags.'''
    parser = argparse.ArgumentParser(description='MedCure', add_help=False)
    parser.add_argument('--config_name', default='medcure', help='name of a function in run_configs.py')
    parser.add_argument('--resume', default=None, type=str, help='checkpoint to resume from')
    parser.add_argument('--eval', default=None, action='store_true', help='evaluate only, then exit')
    cmd_args = parser.parse_args()

    import run_configs
    return getattr(run_configs, cmd_args.config_name)().add_cmd_args(cmd_args)


if __name__ == '__main__':
    args = parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
