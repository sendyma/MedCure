# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from dinov2.models.vision_transformer import vit_base
from projection import load_projection_head

log = logging.getLogger(__name__)


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[
                ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)
            ],
            None,
        )


URL_DICT = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
}


class TextEncoder(nn.Module):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.cls_token_id
        self.out_dim = self.model.config.hidden_size

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs["last_hidden_state"]  # (batch, seq_len, hidden_size)


class ImageEncoder(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', image_size=224):
        super().__init__()
        self.model = vit_base(patch_size=14, img_size=image_size, init_values=1.0, block_chunks=0)
        state_dict = torch.hub.load_state_dict_from_url(URL_DICT[model_name], map_location="cpu")

        # The released weights are for a 224px grid; resize the positional
        # embedding bicubically when training at another resolution.
        if self.model.pos_embed.shape[1] != state_dict['pos_embed'].shape[1]:
            cls_pos_embed = state_dict['pos_embed'][:, 0:1, :]
            patch_pos_embed = state_dict['pos_embed'][:, 1:, :]
            orig_size = int(patch_pos_embed.shape[1] ** 0.5)
            new_size = image_size // self.model.patch_size
            patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_size, new_size),
                                            mode='bicubic', align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
            state_dict['pos_embed'] = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

        # strict=False: the *_reg4 checkpoints carry 4 register tokens that this
        # backbone is built without, so those weights are dropped.
        print('load dinov2 pretrained model:', self.model.load_state_dict(state_dict, strict=False))
        self.out_dim = self.model.embed_dim

    def forward(self, x):
        return self.model(x)


class CXRClip(nn.Module):
    #: constructor arguments that are read straight off a `configs.Config`
    CONFIG_KEYS = (
        "visual_name", "text_name", "image_size", "temperature", "text_pooling",
        "projection_head", "proj_dim", "proj_dropout",
        "graph_mode", "graph_sampling_mode", "gamma_forward", "gamma_reverse", "n_neighbor",
    )

    @classmethod
    def from_config(cls, args):
        """Build the model from a `configs.Config`. See that file for the defaults."""
        return cls(**{k: getattr(args, k) for k in cls.CONFIG_KEYS})

    def __init__(
        self,
        visual_name="dinov2_vitb14",
        text_name="emilyalsentzer/Bio_ClinicalBERT",
        image_size=224,
        temperature=0.01,
        text_pooling="eos",
        projection_head="linear",
        proj_dim=512,
        proj_dropout=0.1,
        graph_mode="sum",
        graph_sampling_mode="weighted",
        gamma_forward=0.5,
        gamma_reverse=1.0,
        n_neighbor=8,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=visual_name, image_size=image_size)
        self.text_encoder = TextEncoder(model_name=text_name)

        self.text_pooling = text_pooling

        self.projection = projection_head != "none"
        if self.projection:
            head_cfg = {"name": projection_head, "dropout": proj_dropout, "proj_dim": proj_dim}
            self.image_projection = load_projection_head(
                embedding_dim=self.image_encoder.out_dim, config_projection_head=head_cfg
            )
            self.text_projection = load_projection_head(
                embedding_dim=self.text_encoder.out_dim, config_projection_head=head_cfg
            )
        else:
            assert (
                self.image_encoder.out_dim == self.text_encoder.out_dim
            ), "Without 'projection_head', embedding_dim of the image and text encoder must be the same."

        self.temperature = temperature
        if self.temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        else:
            self.logit_scale = torch.tensor(1, dtype=torch.float32)
            log.warning("[CXRCLIP] missing temperature scaling factor")

        # Batch-graph curation; see `configs.Config` for what each knob means.
        self.graph_mode = graph_mode
        self.graph_sampling_mode = graph_sampling_mode
        self.gamma_forward = gamma_forward
        self.gamma_reverse = gamma_reverse
        self.n_neighbor = n_neighbor

    def encode_image(self, image):
        image_features = self.image_encoder(image)
        image_embeddings = self.image_projection(image_features) if self.projection else image_features
        return F.normalize(image_embeddings, p=2, dim=1)

    def encode_text(self, text_tokens):
        text_features = self.text_encoder(text_tokens)

        if self.text_pooling == "eos":
            # take features from the eot embedding (eos_token is the highest number in each sequence)
            eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
            text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        elif self.text_pooling == "bos":   # [CLS] token
            text_features = text_features[:, 0]
        elif self.text_pooling == "mean":
            input_mask_expanded = text_tokens["attention_mask"].unsqueeze(axis=-1).expand(text_features.size()).float()
            text_features = torch.sum(text_features * input_mask_expanded, axis=1) / torch.clamp(input_mask_expanded.sum(axis=1), min=1e-9)
        else:
            raise NotImplementedError("Not supported pooling method : %s", self.text_pooling)

        text_embeddings = self.text_projection(text_features) if self.projection else text_features
        return F.normalize(text_embeddings, p=2, dim=1)

    def forward(self, images, text_tokens, image_ids=None, curation_ratio=None):
        """One training step's worth of embeddings, ids and losses.

        Args:
            curation_ratio: `None` trains on the whole batch (fixed-subset
                stage); a ratio selects that fraction by graph density first
                (curation stage).

        Returns:
            (image_embeds, text_embeds, image_ids, loss_dict), all gathered
            across ranks so the contrastive batch spans every GPU.
        """
        # encode_* already returns L2-normalised embeddings.
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(text_tokens)

        if dist.is_initialized() and dist.get_world_size() > 1:
            all_image_embeds = AllGather.apply(image_embeddings)
            all_text_embeds = AllGather.apply(text_embeddings)
            all_image_ids = AllGather.apply(image_ids) if image_ids is not None else None
        else:
            all_image_embeds = image_embeddings
            all_text_embeds = text_embeddings
            all_image_ids = image_ids

        loss_dict = self._compute_losses(all_image_embeds, all_text_embeds, curation_ratio)
        return all_image_embeds, all_text_embeds, all_image_ids, loss_dict

    def _compute_losses(self, all_image_embeds, all_text_embeds, curation_ratio=None):
        """`curation_ratio=None` skips selection and uses the whole batch."""
        device = all_image_embeds.device
        batch_size_all = all_image_embeds.size(0)
        keep_all = torch.ones(batch_size_all, dtype=torch.bool, device=device)

        if curation_ratio is None:
            selected_indices, mask = keep_all, None
        else:
            selected_indices = self._curate_batch(all_image_embeds, all_text_embeds, curation_ratio)
            if selected_indices.sum() == 0:  # ratio too small to keep anything
                selected_indices, mask = keep_all, None
            else:
                mask = selected_indices

        return {
            'contrastive_loss': self.compute_contrastive_loss(all_image_embeds, all_text_embeds, mask),
            'keep mask': selected_indices,
        }

    @torch.no_grad()
    def _curate_batch(self, all_image_embeds, all_text_embeds, curation_ratio):
        """Pick `curation_ratio` of the batch by graph density.

        A pair is interesting when the two modalities disagree, and more so when
        its neighbours disagree too; greedy selection then damps the neighbours
        of everything it picks so the batch stays diverse.
        """
        device = all_image_embeds.device
        batch_size_all = all_image_embeds.size(0)

        cos_sim = torch.mm(all_image_embeds, all_text_embeds.t()).diagonal()
        importance_scores = 1 - cos_sim

        concat_features = F.normalize(torch.cat([all_image_embeds, all_text_embeds], dim=-1), dim=-1, p=2)
        dists = torch.cdist(concat_features, concat_features, p=2)  # [B, B], within [0, 2]
        sorted_dists, sorted_indices = torch.sort(dists, dim=1)
        # column 0 is the sample itself, so the neighbours start at 1
        neighbor_distances = sorted_dists[:, 1:self.n_neighbor + 1]
        neighbors = sorted_indices[:, 1:self.n_neighbor + 1]

        edge_weights = torch.exp(-neighbor_distances * self.gamma_forward)
        connect = edge_weights * importance_scores[neighbors]
        neighbor_sum = torch.sum(connect, dim=-1)
        if self.graph_mode == 'sum':
            graph_density = importance_scores + neighbor_sum
        elif self.graph_mode == 'mean':
            graph_density = importance_scores + torch.mean(connect, dim=-1)
        elif self.graph_mode == 'weighted_mean':
            graph_density = importance_scores + neighbor_sum / (torch.sum(edge_weights, dim=-1) + 1e-12)
        else:
            raise ValueError('unknown graph_mode: %s' % self.graph_mode)

        selected = self.select_batch_(int(curation_ratio * batch_size_all),
                                      graph_density, neighbors, neighbor_distances)
        keep = torch.zeros(batch_size_all, dtype=torch.bool, device=device)
        keep[torch.as_tensor(selected, dtype=torch.long, device=device)] = True
        return keep

    def select_batch_(self, N, graph_density, neighbors, distances):
        """Greedily take the densest sample, then damp its neighbours.

        `graph_density` is modified in place.
        """
        batch = set()
        while len(batch) < N:
            selected = torch.argmax(graph_density).item()
            if self.graph_sampling_mode == 'absolute':
                penalty = graph_density[selected]
            elif self.graph_sampling_mode == 'weighted':
                # the closer a neighbour is, the more of the pick's score it loses
                penalty = torch.exp(-distances[selected] * self.gamma_reverse) * graph_density[selected]
            else:
                raise ValueError('unknown graph_sampling_mode: %s' % self.graph_sampling_mode)
            graph_density[neighbors[selected]] -= penalty
            batch.add(selected)
            # push everything already picked below the current floor
            graph_density[list(batch)] = torch.min(graph_density) - 1
        return list(batch)

    def compute_contrastive_loss(self, image_embeds, text_embeds, mask=None):
        if mask is not None:
            image_embeds = image_embeds[mask]
            text_embeds = text_embeds[mask]

        # cosine similarity as logits
        logits_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_per_text  = self.logit_scale.exp() * text_embeds @ image_embeds.t()
        labels = torch.arange(logits_per_image.size(0), device=image_embeds.device)

        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2.
        return loss
