import math
import copy
import os
import warnings
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Tuple
import matplotlib.pyplot as plt

from torch import Tensor, device


class ResNet101(nn.Module):
    def __init__(self, args: dict):
        super(ResNet101, self).__init__()
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        model = models.resnet101()
        if len(args['resnet101_path']) != 0:
            cur_state_dict = model.state_dict()
            pre_trained_state = torch.load(args['resnet101_path'])
            valid_state_dict = {k: v for k, v in pre_trained_state.items() if k in cur_state_dict}
            cur_state_dict.update(valid_state_dict)
            model.load_state_dict(cur_state_dict)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        patch_feats = self.model(images)
        # avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  # avg_pool
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # return patch_feats, avg_feats  # (batch, 49, 2048)
        return patch_feats  # (batch, 49, 2048)

    def get_global_embedding(self, images):
        # images is tensor (batch, 49, 2048)
        batch_size, _, feat_size = images.size()
        images = images.permute(0, 2, 1)
        images = images.reshape(batch_size, feat_size, 7, 7)
        avg_feats = self.avg_fnt(images).squeeze().reshape(-1, images.size(1))  # avg_pool
        return avg_feats


class ResNet50(nn.Module):
    def __init__(self, args: dict):
        super(ResNet50, self).__init__()
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        model = models.resnet50()
        checkpoint = os.path.join(args['cxr_bert_path'], 'biovil_image_resnet50_proj_size_128.pt')
        cur_state_dict = model.state_dict()
        pretrain_state = torch.load(checkpoint)
        valid_state = {k.split('encoder.encoder.')[1]: v for k, v in pretrain_state.items()
                       if 'encoder' in k and k.split('encoder.encoder.')[1] in cur_state_dict}
        # valid_state = {}
        # for k, v in pretrain_state.items():
        #     if 'encoder' in k:
        #         cur_k = k.split('encoder.encoder.')[1]
        #         if cur_k in cur_state:
        #             valid_state[cur_k] = v
        cur_state_dict.update(valid_state)
        model.load_state_dict(cur_state_dict, strict=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        patch_feats = self.model(images)
        # avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  # avg_pool
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # return patch_feats, avg_feats  # (batch, 49, 2048)
        return patch_feats  # (batch, 49, 2048)

    def get_global_embedding(self, images):
        # images is tensor (batch, 49, 2048)
        batch_size, _, feat_size = images.size()
        images = images.permute(0, 2, 1)
        images = images.reshape(batch_size, feat_size, 7, 7)
        avg_feats = self.avg_fnt(images).squeeze().reshape(-1, images.size(1))  # avg_pool
        return avg_feats


class ResNet101V724(nn.Module):
    def __init__(self, args: dict):
        super(ResNet101V724, self).__init__()
        # self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        model = models.resnet101()
        if len(args['resnet101_path']) != 0:
            cur_state_dict = model.state_dict()
            pre_trained_state = torch.load(args['resnet101_path'])
            valid_state_dict = {k: v for k, v in pre_trained_state.items() if k in cur_state_dict}
            cur_state_dict.update(valid_state_dict)
            model.load_state_dict(cur_state_dict)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        patch_feats = self.model(images)
        # avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  # avg_pool
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # return patch_feats, avg_feats  # (batch, 49, 2048)
        avg_feats = torch.mean(patch_feats, dim=1)
        return patch_feats, avg_feats  # (batch, 49, 2048)

    # def get_global_embedding(self, images):
    #     # images is tensor (batch, *, 2048)
    #     # batch_size, _, feat_size = images.size()
    #     # images = images.permute(0, 2, 1)
    #     # images = images.reshape(batch_size, feat_size, 7, 7)
    #     # avg_feats = self.avg_fnt(images).squeeze().reshape(-1, images.size(1))  # avg_pool
    #     avg_feats = torch.mean(images, dim=1)
    #     return avg_feats


class ResNet50V724(nn.Module):
    def __init__(self, args: dict):
        super(ResNet50V724, self).__init__()
        model = models.resnet50()
        checkpoint = os.path.join(args['cxr_bert_path'], 'biovil_image_resnet50_proj_size_128.pt')
        cur_state_dict = model.state_dict()
        pretrain_state = torch.load(checkpoint)
        valid_state = {k.split('encoder.encoder.')[1]: v for k, v in pretrain_state.items()
                       if 'encoder' in k and k.split('encoder.encoder.')[1] in cur_state_dict}
        cur_state_dict.update(valid_state)
        model.load_state_dict(cur_state_dict, strict=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        patch_feats = self.model(images)  # (batch, 2048, 12, 12)
        # avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  # avg_pool
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # avg_feats = torch.mean(patch_feats, dim=1)
        return patch_feats  # (batch, 49, 2048)

    # def get_global_embedding(self, images):
    #     # images is tensor (batch, *, 2048)
    #     # batch_size, _, feat_size = images.size()
    #     # images = images.permute(0, 2, 1)
    #     # images = images.reshape(batch_size, feat_size, 7, 7)
    #     # avg_feats = self.avg_fnt(images).squeeze().reshape(-1, images.size(1))  # avg_pool
    #     avg_feats = torch.mean(images, dim=1)
    #     return avg_feats


class ResNetTemp(nn.Module):
    def __init__(self, args: dict):
        super(ResNetTemp, self).__init__()
        model = models.resnet101()
        if len(args['resnet_checkpoint']) != 0:
            model.load_state_dict(torch.load(args['resnet_checkpoint']))
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        # self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  # avg_pool
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class PermuteProjectionHead(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()

        self.proj = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.proj(x)
        return x.permute(0, 2, 1).contiguous()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ScaledDotProductAttention2D(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention2D, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        queries, keys, values = queries.unsqueeze(0), keys.unsqueeze(0), values.unsqueeze(0)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        queries, keys, values = self.ln_1(queries), self.ln_2(keys), self.ln_2(values)

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out.squeeze(0)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ScaledDotProductAttention2D(dim, dim_head, dim_head, heads, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, queries, keys, values):
        for attn, ff in self.layers:
            queries = attn(queries, keys, values) + queries
            queries = ff(queries) + queries

        return self.norm(queries)


class EncoderPermuteProject(nn.Module):
    """
    Permutes the last hidden state of the encoder so that the spatial position is represented by axis -2 and the encoded
    representation for a spatial position is represented by axis -1. Following this, the encoded representation for each
    spatial position is projected to the size of the decoder's hidden state.
    """
    def __init__(
        self,
        permute_encoder_last_hidden_state: Union[List, bool],
        encoder_last_hidden_state_size: int,
        decoder_hidden_state_size: int,
        **kwargs,
    ):
        """
        Argument/s:
            permute_encoder_last_hidden_state - permutation of the last hidden state of the encoder.
            encoder_last_hidden_state_size - the size of the encoder's last hidden state for each spatial position,
                or axis -1.
            decoder_hidden_state_size - the hidden state size of the decoder.
            kwargs - keyword arguments.
        """
        super(EncoderPermuteProject, self).__init__()

        self.permute_encoder_last_hidden_state = permute_encoder_last_hidden_state
        self.projection = nn.Linear(
            in_features=encoder_last_hidden_state_size,
            out_features=decoder_hidden_state_size,
            bias=False,
        )

    def forward(self, encoder_last_hidden_state: torch.FloatTensor):
        """
        Forward propagation.

        Argument/s:
            images - a batch of images.

        Returns
            a Tensor.
        """
        if self.permute_encoder_last_hidden_state:
            encoder_last_hidden_state = encoder_last_hidden_state.permute(self.permute_encoder_last_hidden_state)
        return self.projection(encoder_last_hidden_state)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()  # h is the number of heads
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:  # query is visual/textual features, key and value is memory matrix
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:  # each variable is transformed into the same dimension with MLP
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]  # reshape the Q, K, and V, (b, 8, 49, 64)
        # note that it includes multi-heads
        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)  # responding values is transformed by using MLP


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)  # scores is each sample, head, patch feature
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # distance
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)  # find the most similar memory vectors
    # 16, 8, 49, 32, 64
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)  # importance weight (16, 8, 49, 32)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


def get_extended_attention_mask(
        attention_mask: Tensor, input_shape: Tuple[int], device: device = None, dtype: torch.float = None,
        is_decoder: bool = False
) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
        device
        dtype
        is_decoder
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = torch.float

    if device is None:
        device = attention_mask.device

    if not (attention_mask.dim() == 2 and is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            extended_attention_mask = create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
    if device is not None:
        warnings.warn(
            "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
        )
    else:
        device = attention_mask.device
    batch_size, seq_length = input_shape
    seq_ids = torch.arange(seq_length, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    # causal and attention masks must have same type with pytorch version < 1.3
    causal_mask = causal_mask.to(attention_mask.dtype)

    if causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        causal_mask = torch.cat(
            [
                torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                causal_mask,
            ],
            axis=-1,
        )

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    return extended_attention_mask

