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

import timm.models.vision_transformer

from attacker import PGDAttacker, NoOpAttacker, WTAttacker, EnsembleAttacker, Gaussian_Attacker

from pytorch_wavelets import DWTForward,DWTInverse

def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status

to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')

class MixLayerNorm(nn.Module):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].

    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.

    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.

    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''
    def __init__(self, d_model, eps=1e-06):
        super(MixLayerNorm, self).__init__()
        self.main_bn = nn.LayerNorm(d_model, eps=eps)
        self.aux_bn = nn.LayerNorm(d_model, eps=eps)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = self.main_bn(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            input0 = self.main_bn(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class AdVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(AdVisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool

        self.attacker = NoOpAttacker()
        self.mixbn = True
        self.attack_type = "pgd"
        self.criterion = nn.CrossEntropyLoss
        
        self.xfm=DWTForward(J=6,mode='symmetric',wave='sym8')
        self.ifm=DWTInverse(mode='symmetric',wave='sym8')

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_mixbn(self, mixbn):
        self.mixbn = mixbn

    def set_criterion(self, criterion):
        self.criterion = criterion

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def _forward_impl(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def forward(self, x, labels):
        training = self.training
        # input_len = len(x)
        # only during training do we need to attack, and cat the clean and auxiliary pics
        if training:
            self.eval()
            self.apply(to_adv_status)
            if isinstance(self.attacker, NoOpAttacker):
                images = x
                targets = labels 
            else:
                if isinstance(self.attacker, PGDAttacker) or isinstance(self.attacker, WTAttacker) or isinstance(self.attacker, EnsembleAttacker):
                    # wt
                    # aux_images, _, _ = self.attacker.attack(x, labels, self._forward_impl, self.xfm, self.ifm)

                    # pgd
                    aux_images, _, _ = self.attacker.attack(x, labels, self._forward_impl, self.xfm, self.ifm, self.criterion)

                    # ensemble
                    # aux_images, _, _ = self.attacker.attack(x, labels, self._forward_impl, self.xfm, self.ifm)
                elif isinstance(self.attacker, Gaussian_Attacker):
                    aux_images = self.attacker.attack(x, mean=0.0, var=0.001)
                # else:
                    # mix
                    # aux_images1, _, loss1 = self.attacker[0].attack(x, labels, self._forward_impl, xfm=self.xfm, ifm=self.ifm)
                    # aux_images2, _, loss2 = self.attacker[1].attack(x, labels, self._forward_impl, xfm=None, ifm=None)
                    # aux_images = torch.where(loss1.view(loss1.shape[0],1,1,1) <= loss2.view(loss2.shape[0],1,1,1), aux_images1, aux_images2)

                    # random
                    # prob = self.prob[id]
                    # if prob > 0.5:
                    #     aux_images, _, _ = self.attacker[0].attack(x, labels, self._forward_impl, xfm=self.xfm, ifm=self.ifm, criterion=self.criterion)
                    # else:
                    #     aux_images, _, _ = self.attacker[1].attack(x, labels, self._forward_impl, xfm=None, ifm=None, criterion=self.criterion)

                images = torch.cat([x, aux_images], dim=0)
                targets = torch.cat([labels, labels], dim=0)
            self.train()
            if self.mixbn:
                # the DataParallel usually cat the outputs along the first dimension simply,
                # so if we don't change the dimensions, the outputs will be something like
                # [clean_batches_gpu1, adv_batches_gpu1, clean_batches_gpu2, adv_batches_gpu2...]
                # Then it will be hard to distinguish clean batches and adversarial batches.
                self.apply(to_mix_status)
                return self._forward_impl(images), targets
            else:
                self.apply(to_clean_status)
                return self._forward_impl(images), targets
        else:
            images = x
            targets = labels
            return self._forward_impl(images), targets

    

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def advit_base_patch16(**kwargs):
    model = AdVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(MixLayerNorm, eps=1e-6), **kwargs)
    return model


def advit_large_patch16(**kwargs):
    model = AdVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(MixLayerNorm, eps=1e-6), **kwargs)
    return model


def advit_huge_patch14(**kwargs):
    model = AdVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(MixLayerNorm, eps=1e-6), **kwargs)
    return model