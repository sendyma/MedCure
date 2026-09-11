# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import json
import os

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from constants import DATA_ROOT


def gather(local, world_size):
    """Collect a per-rank python object into one flat list. No-op when not distributed."""
    if world_size == 1 or not dist.is_initialized():
        return list(local)
    parts = [None for _ in range(world_size)]
    dist.all_gather_object(parts, local)
    return [x for part in parts for x in part]


def load_metadata(metadir="evaluation"):
    """Read the eval-task catalog and label sets.

    Catalog paths are stored relative to the corpus root so the file is portable;
    they are resolved against `constants.DATA_ROOT` (`$MEDCURE_DATA_ROOT`) here.
    """
    with open(os.path.join(metadir, 'dataset_catalog.json')) as f:
        catalog = json.load(f)
    for entry in catalog.values():
        entry['path'] = str(DATA_ROOT / entry['path'])
    with open(os.path.join(metadir, 'labels.json')) as f:
        all_labels = json.load(f)
    return catalog, all_labels


# Tasks scored by image-text retrieval; everything else is zero-shot classification.
RETRIEVAL_TASKS = frozenset({'chexpert_5x200_retrieve', 'mimic_retrieve'})


def evaluate(args, d, val_loader, labels, model, tokenizer, max_bert_length):
    if args.rank == 0:
        print('Evaluating: {}, Number of samples: {}'.format(d, len(val_loader.dataset)))

    if d in RETRIEVAL_TASKS:
        return retrieval_image_text(args, model, val_loader, tokenizer, max_bert_length)
    return zeroshot_binary_prompt(args, model, val_loader, labels, tokenizer, max_bert_length)


@torch.no_grad()
def zeroshot_binary_prompt(args, model, dataloader, cxr_labels, tokenizer, max_bert_length=256):
    """Per-finding AUC by scoring "{finding}" against "no {finding}"."""
    device = next(model.parameters()).device

    zeroshot_weights = []
    for classname in cxr_labels:
        texts = tokenizer([t.format(classname) for t in ('{}', 'no {}')], return_tensors='pt',
                          padding="longest", truncation=True, max_length=max_bert_length)
        texts = {k: v.to(device, non_blocking=True) for k, v in texts.items()}
        zeroshot_weights.append(model.encode_text(texts))  # already L2-normalised
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)  # [2, n_classes, dim]

    local_predictions, local_targets = [], []
    for batch in dataloader:  # datasets yield (image, target, path)
        images, target = batch[0], batch[1]
        image_features = model.encode_image(images.to(device, non_blocking=True))
        # [B, n_classes, 2]: positive vs negated prompt
        logits = torch.stack([image_features @ zeroshot_weights[i].t() for i in (0, 1)], dim=2)
        local_predictions.append(logits.softmax(dim=2).cpu().numpy())
        local_targets.append(target.cpu().numpy())

    predictions = np.concatenate(gather(local_predictions, args.world_size), axis=0)
    targets = np.concatenate(gather(local_targets, args.world_size), axis=0)

    all_auc = []
    for c in range(len(cxr_labels)):
        keep = targets[:, c] != -1  # -1 marks MIMIC's uncertain labels
        y_true, y_score = targets[keep, c], predictions[keep, c, 0]
        if len(np.unique(y_true)) > 1:  # AUC is undefined for a single-class column
            all_auc.append(roc_auc_score(y_true, y_score))
    all_auc = np.asarray(all_auc)
    return {"mean": all_auc.mean(), "individual": all_auc.tolist()}

@torch.no_grad()
def retrieval_image_text(args, model, dataloader, tokenizer, max_bert_length=256):
    """Image -> text retrieval over the set of distinct reports."""
    device = next(model.parameters()).device

    local_image_embeddings, local_text_embeddings, local_texts = [], [], []
    for batch in dataloader:  # datasets yield (image, report, path)
        images, text = batch[0], batch[1]
        tokens = tokenizer(text, return_tensors='pt', padding="longest",
                           truncation=True, max_length=max_bert_length)
        tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}
        local_image_embeddings.append(model.encode_image(images.to(device, non_blocking=True)).cpu().numpy())
        local_text_embeddings.append(model.encode_text(tokens).cpu().numpy())
        local_texts.extend(text)

    image_embeddings = np.concatenate(gather(local_image_embeddings, args.world_size), axis=0)
    text_embeddings = np.concatenate(gather(local_text_embeddings, args.world_size), axis=0)
    text_list = gather(local_texts, args.world_size)

    # Collapse duplicate reports: each distinct text becomes one retrieval candidate.
    text_to_label, candidate_rows, labels = {}, [], []
    for i, text in enumerate(text_list):
        if text not in text_to_label:
            text_to_label[text] = len(candidate_rows)
            candidate_rows.append(i)
        labels.append(text_to_label[text])
    candidates = text_embeddings[np.asarray(candidate_rows)]
    n_text = len(candidate_rows)

    similarities = cosine_similarity(image_embeddings, candidates)
    # rank 1 == the paired report is the most similar candidate
    positions = similarities.argsort(axis=1).argsort(axis=1)[np.arange(len(labels)), labels]
    ranks = n_text - positions

    result = {f"Recall@{k}": float((ranks <= k).mean()) for k in (1, 5, 10)}
    result["MeanRank"] = float(ranks.mean())
    return {"mean": result["Recall@1"], "individual": list(result.values())}
