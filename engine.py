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

import json
import math
import sys
import time
from typing import Iterable

import numpy as np
import torch
import wandb

import util.lr_sched as lr_sched
import util.misc as misc


#: `epoch` value used for the alignment warm-up, which runs before epoch 0.
WARMUP_EPOCH = -1


def to_device(samples, device):
    for key in samples:
        if key in ["pixel_values", "image_ids"]:
            samples[key] = samples[key].to(device, non_blocking=True)
        elif key == "text_tokens":
            for k in samples[key]:
                samples[key][k] = samples[key][k].to(device, non_blocking=True)
    return samples


@torch.no_grad()
def evaluate(args, model, tokenizer):
    """Zero-shot / retrieval evaluation over every task in evaluation/dataset_catalog.json."""
    model.eval()

    from evaluation import metrics as eval_metrics

    catalog, all_labels = eval_metrics.load_metadata()
    metrics = {}
    start_time = time.time()

    for d in catalog:
        val_dataset = args.val_dataset[d]
        labels = all_labels[d]

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.eval_num_workers,
            pin_memory=False,
            drop_last=False,
        )

        metric = eval_metrics.evaluate(args, d, val_loader, labels, model, tokenizer, args.max_bert_length)

        metrics[d] = metric
        if args.eval:
            misc.print_json(args.output_dir, json.dumps({"task": d, "acc": metric}))

    print("evaluation time: %.2fs" % (time.time() - start_time))
    model.train()
    return metrics


def train_one_epoch(model: torch.nn.Module, model_without_ddp, tokenizer,
                    data_loader: Iterable, best_acc, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, step, loss_scaler, eff_batch_size, max_norm: float = 0,
                    global_example_ids: set = (),
                    curation_ratio=None,
                    total_steps=None,
                    args=None):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    if epoch == WARMUP_EPOCH:
        stage = "warm-up: pretrain for cross-modal alignment"
    elif curation_ratio is not None:
        stage = "curation+pretrain (curation_ratio=%.3f)" % curation_ratio
    else:
        stage = "pretrain on fixed subset"
    if args.rank == 0:
        print("epoch %d: %s, %d steps" % (epoch, stage, len(data_loader)))

    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header, args.max_update)):
        if step[0] > args.max_update:
            break

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_step_learning_rate(optimizer, step[0], args.lr, args.min_lr, args.warmup_steps, total_steps)

        inputs = to_device(samples, device)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            image_embeds, text_embeds, image_ids, loss_dict = model(
                inputs['pixel_values'], inputs['text_tokens'], inputs['image_ids'], curation_ratio)
            loss_total = loss_dict['contrastive_loss']

        loss_value = loss_total.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_total /= accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(loss_total, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=update_grad)

        logits = image_embeds.detach() @ text_embeds.detach().t()
        off_diagonal = ~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        cosine_score = logits.diagonal().mean()
        non_cosine_score = logits[off_diagonal].mean()

        if curation_ratio is not None:  # stage 2 must not grow the frozen subset
            global_example_ids.update(image_ids[loss_dict['keep mask'].cpu()].cpu().numpy().tolist())

        args.subset_ratio = len(global_example_ids) / args.full_dataset_size

        if update_grad:
            step[0] += 1
            optimizer.zero_grad()

        if data_iter_step % 3 == 0 and args.rank == 0:
            wandb.log({
                'Epoch': epoch,
                'Global step': step[0],
                'Local step': data_iter_step,
                "Optimizer LR": optimizer.param_groups[0]['lr'],
                "Temperature": model_without_ddp.logit_scale.exp().item(),
                "Actual batch": loss_dict['keep mask'].sum().item(),
                'Loss_ctr': loss_dict['contrastive_loss'].item(),
                'Loss_total': loss_total.item(),
                'cosine_score': cosine_score.item(),
                'non cosine_score': non_cosine_score.item(),
                'subset ratio': args.subset_ratio,
                'batch ratio': loss_dict['keep mask'].sum() / len(loss_dict['keep mask']),
            })

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=max(g["lr"] for g in optimizer.param_groups))

        # The warm-up neither evaluates nor checkpoints: a "best" saved there
        # would record epoch -1 and make a resume re-run the warm-up.
        if (epoch != WARMUP_EPOCH and step[0]
                and step[0] % args.eval_steps == 0 and step[0] >= args.eval_start_step):
            metric = evaluate(args, model_without_ddp, tokenizer)
            if args.rank == 0:
                misc.print_json(args.output_dir, json.dumps(
                    {"step": step[0], "acc": metric, "seen": eff_batch_size * step[0]}))
                wandb.log({f"Eval/{k}": v['mean'] for k, v in metric.items()})

                mean_acc = float(np.mean([v['mean'] for v in metric.values()]))
                if mean_acc > best_acc[0]:
                    best_acc[0] = mean_acc
                    misc.save_model(args=args, model_without_ddp=model_without_ddp,
                                    optimizer=optimizer, global_example_ids=global_example_ids,
                                    loss_scaler=loss_scaler, epoch=epoch, epoch_name="best",
                                    best_acc=best_acc[0], step=step[0])
            model.train(True)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
