# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp


from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_causal

torch.autograd.set_detect_anomaly(True)


def count_nan(x):
    return torch.isnan(x).sum().detach().cpu().item()


def check_attn_holes(x):
    """Checks how many rows of the attention mask have all 0s."""
    out = torch.sum(x, dim=3)
    return len(out[out == 0])


class Attention(nn.Module):
    """
    A simple modification of timm's Attention class for (masked) self/cross attention.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cross_attn=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.cross_attn = cross_attn

        if cross_attn:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None, attn_mask=None):
        if context is not None:
            assert self.cross_attn, "cross_attn should be set to True if using context"
        B, N, C = x.shape

        if self.cross_attn:
            q = (
                self.q(x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            kv = (
                self.kv(context)
                .reshape(B, N, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = (kv[0], kv[1])
        else:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            # q, k, v: B x num_heads x N x C // num_heads
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Use custom large neg vals over float("-inf") so softmax gives uniform on empty rows
        if attn_mask is not None:
            attn.masked_fill_(attn_mask == 0, -65500)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Simple modification of timm's Block to support both self + cross attention.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cross_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cross_attn=cross_attn,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, context=None, attn_mask=None):
        if context is not None:
            context = self.norm1(context)
        x = x + self.drop_path(self.attn(self.norm1(x), context, attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TwoStreamBlock(nn.Module):
    """ TwoStream Block for XLNet style architecture."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.content_blk = Block(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            norm_layer=norm_layer,
            cross_attn=False,
        )
        self.query_blk = Block(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            norm_layer=norm_layer,
            cross_attn=True,
        )

    def forward(self, x, query, attn_mask, attn_mask_query):
        """
            x: normal content input for self-attention or key/value for query
            query: for query attention stream 
            attn_mask: attn mask for content stream
            attn_mask_query: attn mask for query stream, 0s on diag. 

            Return: content_out, query_out
        """
        # query needs position-only as input, content as key/value
        query = self.query_blk(query, context=x, attn_mask=attn_mask_query)
        # content takes everything, goes after so x used above is previous x
        x = self.content_blk(x, attn_mask=attn_mask)

        return x, query


class XLViT(nn.Module):
    """XLViT, i.e. using permutations with XLNet-inspire Two Stream architecture."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        skip_first=5
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        # Query token embedding for first layer
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # causal attention mask for permutation indexing
        # should be B x num_heads x N x N
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(num_patches + 1, num_patches + 1)).view(
                1, 1, num_patches + 1, num_patches + 1
            ),
        )

        self.register_buffer(
            "mask_q",
            torch.tril(torch.ones(num_patches + 1, num_patches + 1), diagonal=-1).view(
                1, 1, num_patches + 1, num_patches + 1
            ),
        )

        # blocks are (content, query) i.e. (h, g)
        # content is the typical attention mechanism, query is cross attn
        self.blocks = nn.ModuleList(
            [
                TwoStreamBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.pred = nn.Linear(
            embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.skip_first = skip_first
        print(f"Skip first set to {skip_first}")

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        # adjust for causal encoding
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def create_permutation_mask(self, x):
        N, L, D = x.shape  # batch, length, dim

        # generate random permutation for each sample in batch
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)

        # index in for modified attention mask
        idx = ids_shuffle.view(N, 1, 1, L).repeat(1, 1, L, 1)  # N, 1, L, L
        idx_r = ids_shuffle.view(N, 1, L, 1).repeat(1, 1, 1, L)  # N, 1, L, L

        # gather across columns, then across rows of the attn mask
        attn_mask = torch.gather(
            self.mask[:, :, :L, :L].repeat(N, 1, 1, 1), dim=3, index=idx
        )
        attn_mask = torch.gather(attn_mask, dim=2, index=idx_r)

        # repeat but with diagonal removed
        attn_mask_query = torch.gather(
            self.mask_q[:, :, :L, :L].repeat(N, 1, 1, 1), dim=3, index=idx
        )
        attn_mask_query = torch.gather(attn_mask_query, dim=2, index=idx_r)
        # attn_mask_query = attn_mask.detach().clone()
        # attn_mask_query.diagonal(dim1=-2, dim2=-1).fill_(0.0)

        return attn_mask, attn_mask_query, ids_shuffle

    def forward_encoder(self, x):
        """
        x: [N, 3, H, W].
        after patch embed: [N, L, D], sequence
        """
        # embed patches
        x = self.patch_embed(x)
        N, L, D = x.shape

        # create query input from trainable token
        x_query = self.query_token.expand(x.shape[0], L, -1)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        x_query = x_query + self.pos_embed[:, 1:, :]

        # get permutations of x and the attn mask
        attn_mask, attn_mask_query, ids_shuffle = self.create_permutation_mask(
            x)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_query = torch.cat((cls_tokens, x_query), dim=1)

        # exten attn for cls; use cls as dummy token that everything attends to but only attends to itself
        attn_mask = F.pad(attn_mask, (1, 0, 1, 0), value=0.0)
        attn_mask_query = F.pad(attn_mask_query, (1, 0, 1, 0), value=0.0)
        attn_mask[:, :, :, 0] = 1.0
        attn_mask_query[:, :, :, 0] = 1.0

        # apply Transformer blocks, being careful to use the right x blocks
        for i, blk in enumerate(self.blocks):
            x, x_query = blk(x, x_query, attn_mask, attn_mask_query)

        # predict from query sequence
        x_query = self.norm(x_query)

        # predictor projection
        x_query = self.pred(x_query)

        # remove cls token
        x_query = x_query[:, 1:, :]

        return x_query, ids_shuffle

    def forward_loss(self, imgs, pred, ids_shuffle):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        ids_shuffle: [N, L]
        """
        target = self.patchify(imgs)

        # no need to permute target, we permuted in attention to target directly
        # no need to remove first token for causal either, we predict it based on query

        # use ids_restore to make mask to skip loss on the first skip_first tokens
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = ids_restore >= self.skip_first

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # first avg out patches
        loss = (loss * mask).sum() / mask.sum()  # then mask out

        return loss

    def forward(self, imgs):
        pred, ids_shuffle = self.forward_encoder(imgs)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, ids_shuffle)
        return loss, pred

# * Modify num_heads 3 -> 12
def xl_vit_tiny_patch16(**kwargs):
    model = XLViT(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def xl_vit_small_patch16(**kwargs):
    model = XLViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def xl_vit_base_patch16(**kwargs):
    model = XLViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def xl_vit_large_patch16(**kwargs):
    model = XLViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def xl_vit_huge_patch14(**kwargs):
    model = XLViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# set recommended archs
# xl_vit_tiny_patch16 = xl_vit_tiny_patch16  # decoder: 512 dim, 8 blocks
# xl_vit_small_patch16 = xl_vit_small_patch16  # decoder: 512 dim, 8 blocks
# xl_vit_base_patch16 = xl_vit_base_patch16  # decoder: 512 dim, 8 blocks
# xl_vit_large_patch16 = xl_vit_large_patch16  # decoder: 512 dim, 8 blocks
# xl_vit_huge_patch14 = xl_vit_huge_patch14  # decoder: 512 dim, 8 blocks
