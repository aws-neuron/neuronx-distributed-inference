# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
S3Diff one-step 4x super-resolution on AWS Neuron.

Model: Yukang/S3Diff (weights from zhangap/S3Diff on HuggingFace)
Paper: "Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors" (ECCV 2024)

Architecture: SD-Turbo UNet with dynamic LoRA modulation. A DEResNet encoder
estimates input degradation, and per-layer LoRA scaling factors are computed
from these scores to condition the UNet on the specific degradation pattern.

This module provides the full pipeline: load, compile, and run inference.
Uses torch_neuronx.trace() since the model is small (~2 GB) and does not
benefit from tensor parallelism.
"""

import math
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Scale factor and padding constants
SF = 4
PAD_H = 512
PAD_W = 512

# ---------------------------------------------------------------------------
# DEResNet -- degradation estimator
# ---------------------------------------------------------------------------


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class DEResNet(nn.Module):
    """Degradation Estimation ResNet. Outputs per-degradation scores in [0, 1]."""

    def __init__(
        self,
        num_in_ch=3,
        num_degradation=2,
        num_feats=[64, 64, 64, 128],
        num_blocks=[2, 2, 2, 2],
        downscales=[1, 1, 2, 1],
    ):
        super().__init__()
        num_stage = len(num_feats)
        self.conv_first = nn.ModuleList()
        for _ in range(num_degradation):
            self.conv_first.append(nn.Conv2d(num_in_ch, num_feats[0], 3, 1, 1))
        self.body = nn.ModuleList()
        for _ in range(num_degradation):
            body = []
            for stage in range(num_stage):
                for _ in range(num_blocks[stage]):
                    body.append(ResidualBlockNoBN(num_feats[stage]))
                if downscales[stage] == 1:
                    if (
                        stage < num_stage - 1
                        and num_feats[stage] != num_feats[stage + 1]
                    ):
                        body.append(
                            nn.Conv2d(num_feats[stage], num_feats[stage + 1], 3, 1, 1)
                        )
                elif downscales[stage] == 2:
                    body.append(
                        nn.Conv2d(
                            num_feats[stage],
                            num_feats[min(stage + 1, num_stage - 1)],
                            3,
                            2,
                            1,
                        )
                    )
            self.body.append(nn.Sequential(*body))
        self.num_degradation = num_degradation
        self.fc_degree = nn.ModuleList()
        for _ in range(num_degradation):
            self.fc_degree.append(
                nn.Sequential(
                    nn.Linear(num_feats[-1], 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 1),
                    nn.Sigmoid(),
                )
            )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        degrees = []
        for i in range(self.num_degradation):
            x_out = self.conv_first[i](x)
            feat = self.body[i](x_out)
            feat = self.avg_pool(feat).squeeze(-1).squeeze(-1)
            degrees.append(self.fc_degree[i](feat).squeeze(-1))
        return torch.stack(degrees, dim=1)


# ---------------------------------------------------------------------------
# Custom LoRA forward with degradation modulation
# ---------------------------------------------------------------------------


def my_lora_fwd(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """LoRA forward that injects degradation modulation between lora_A and lora_B.

    For Conv2d LoRA layers: einsum('...khw,...kr->...rhw', lora_A(x), de_mod)
    For Linear LoRA layers: einsum('...lk,...kr->...lr', lora_A(x), de_mod)

    The de_mod tensor is a [B, rank, rank] matrix set per-layer by the wrapper
    modules before the forward pass.
    """
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(
            x, *args, adapter_names=adapter_names, **kwargs
        )
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            if not self.use_dora[active_adapter]:
                _tmp = lora_A(dropout(x))
                if isinstance(lora_A, torch.nn.Conv2d):
                    _tmp = torch.einsum("...khw,...kr->...rhw", _tmp, self.de_mod)
                elif isinstance(lora_A, torch.nn.Linear):
                    _tmp = torch.einsum("...lk,...kr->...lr", _tmp, self.de_mod)
                else:
                    raise NotImplementedError
                result = result + lora_B(_tmp) * scaling
            else:
                x = dropout(x)
                result = result + self._apply_dora(
                    x, lora_A, lora_B, scaling, active_adapter
                )
        result = result.to(torch_result_dtype)
    return result


# ---------------------------------------------------------------------------
# Neuron tracing wrappers
# ---------------------------------------------------------------------------


class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        return self.text_encoder(input_ids)[0]


class VAEEncoderWrapper(nn.Module):
    """Wraps VAE encoder to accept de_mod_all [B, 6, rank, rank] as explicit input."""

    def __init__(self, vae, vae_lora_layers, lora_rank_vae):
        super().__init__()
        self.vae = vae
        self.vae_lora_layers = vae_lora_layers
        self.lora_rank_vae = lora_rank_vae
        self.layer_block_map = {}
        for layer_name in vae_lora_layers:
            split_name = layer_name.split(".")
            if split_name[1] == "down_blocks":
                self.layer_block_map[layer_name] = int(split_name[2])
            elif split_name[1] == "mid_block":
                self.layer_block_map[layer_name] = 4
            else:
                self.layer_block_map[layer_name] = 5

    def forward(self, pixel_values, de_mod_all):
        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                block_idx = self.layer_block_map[layer_name]
                module.de_mod = de_mod_all[:, block_idx]
        latent = (
            self.vae.encode(pixel_values).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
        return latent


class UNetWrapper(nn.Module):
    """Wraps UNet to accept de_mod_all [B, 10, rank, rank] as explicit input."""

    def __init__(self, unet, unet_lora_layers, lora_rank_unet):
        super().__init__()
        self.unet = unet
        self.unet_lora_layers = unet_lora_layers
        self.lora_rank_unet = lora_rank_unet
        self.layer_block_map = {}
        for layer_name in unet_lora_layers:
            split_name = layer_name.split(".")
            if split_name[0] == "down_blocks":
                self.layer_block_map[layer_name] = int(split_name[1])
            elif split_name[0] == "mid_block":
                self.layer_block_map[layer_name] = 4
            elif split_name[0] == "up_blocks":
                self.layer_block_map[layer_name] = int(split_name[1]) + 5
            else:
                self.layer_block_map[layer_name] = 9

    def forward(self, latent, timestep, encoder_hidden_states, de_mod_all):
        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                block_idx = self.layer_block_map[layer_name]
                module.de_mod = de_mod_all[:, block_idx]
        return self.unet(
            latent, timestep, encoder_hidden_states=encoder_hidden_states
        ).sample


class VAEDecoderWrapper(nn.Module):
    """Simple wrapper for VAE decoder (no LoRA)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample


# ---------------------------------------------------------------------------
# S3Diff Neuron Pipeline
# ---------------------------------------------------------------------------


class S3DiffNeuronPipeline:
    """End-to-end S3Diff super-resolution pipeline on Neuron.

    Handles model loading, compilation, and inference.

    Args:
        sd_turbo_path: Path to SD-Turbo checkpoint (stabilityai/sd-turbo)
        s3diff_weights_path: Path to s3diff.pkl weights
        de_net_path: Path to de_net.pth weights
        compile_dir: Directory for compiled model cache
        lr_size: Input low-resolution size (default: 128)
    """

    def __init__(
        self,
        sd_turbo_path: str,
        s3diff_weights_path: str,
        de_net_path: str,
        compile_dir: str = "/tmp/s3diff/compiled/",
        lr_size: int = 128,
    ):
        self.sd_turbo_path = sd_turbo_path
        self.s3diff_weights_path = s3diff_weights_path
        self.de_net_path = de_net_path
        self.compile_dir = compile_dir
        self.lr_size = lr_size
        self.hr_size = lr_size * SF

        # Will be set during load()
        self.de_net_neuron = None
        self.text_enc_neuron = None
        self.vae_enc_neuron = None
        self.unet_neuron = None
        self.vae_dec_neuron = None
        self.tokenizer = None
        self.sched = None
        self.compute_modulation = None

    def load(self):
        """Load all model components and build the modulation network."""
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig
        from transformers import AutoTokenizer, CLIPTextModel

        print("Loading SD-Turbo + S3Diff LoRA weights...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sd_turbo_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.sd_turbo_path, subfolder="text_encoder"
        ).eval()
        vae = AutoencoderKL.from_pretrained(self.sd_turbo_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(
            self.sd_turbo_path, subfolder="unet"
        )

        sd = torch.load(self.s3diff_weights_path, map_location="cpu")
        self.lora_rank_unet = sd["rank_unet"]
        self.lora_rank_vae = sd["rank_vae"]
        print(f"LoRA ranks: unet={self.lora_rank_unet}, vae={self.lora_rank_vae}")

        # Add LoRA adapters and load trained weights
        vae_lora_config = LoraConfig(
            r=self.lora_rank_vae,
            init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"],
        )
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        _sd_vae = vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        vae.load_state_dict(_sd_vae)

        unet_lora_config = LoraConfig(
            r=self.lora_rank_unet,
            init_lora_weights="gaussian",
            target_modules=sd["unet_lora_target_modules"],
        )
        unet.add_adapter(unet_lora_config)
        _sd_unet = unet.state_dict()
        for k in sd["state_dict_unet"]:
            _sd_unet[k] = sd["state_dict_unet"][k]
        unet.load_state_dict(_sd_unet)

        # Monkey-patch LoRA forward methods
        vae_lora_layers = []
        for name, module in vae.named_modules():
            if "base_layer" in name:
                vae_lora_layers.append(name[: -len(".base_layer")])
        for name, module in vae.named_modules():
            if name in vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        unet_lora_layers = []
        for name, module in unet.named_modules():
            if "base_layer" in name:
                unet_lora_layers.append(name[: -len(".base_layer")])
        for name, module in unet.named_modules():
            if name in unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        vae.eval()
        unet.eval()

        # Modulation MLPs (tiny, run on CPU)
        num_embeddings = 64
        block_embedding_dim = 64
        W = nn.Parameter(sd["w"], requires_grad=False)

        vae_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
        unet_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
        vae_block_mlp = nn.Sequential(nn.Linear(block_embedding_dim, 64), nn.ReLU(True))
        unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64), nn.ReLU(True)
        )
        vae_fuse_mlp = nn.Linear(256 + 64, self.lora_rank_vae**2)
        unet_fuse_mlp = nn.Linear(256 + 64, self.lora_rank_unet**2)
        vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        for name, module in [
            ("vae_de_mlp", vae_de_mlp),
            ("unet_de_mlp", unet_de_mlp),
            ("vae_block_mlp", vae_block_mlp),
            ("unet_block_mlp", unet_block_mlp),
            ("vae_fuse_mlp", vae_fuse_mlp),
            ("unet_fuse_mlp", unet_fuse_mlp),
        ]:
            _ssd = module.state_dict()
            for k in sd[f"state_dict_{name}"]:
                _ssd[k] = sd[f"state_dict_{name}"][k]
            module.load_state_dict(_ssd)
        vae_block_embeddings.load_state_dict(
            sd["state_embeddings"]["state_dict_vae_block"]
        )
        unet_block_embeddings.load_state_dict(
            sd["state_embeddings"]["state_dict_unet_block"]
        )

        for m in [
            vae_de_mlp,
            unet_de_mlp,
            vae_block_mlp,
            unet_block_mlp,
            vae_fuse_mlp,
            unet_fuse_mlp,
        ]:
            m.eval()

        # DEResNet
        de_net = DEResNet(num_in_ch=3, num_degradation=2)
        de_net.load_state_dict(torch.load(self.de_net_path, map_location="cpu"))
        de_net.eval()

        # Scheduler
        self.sched = DDPMScheduler.from_pretrained(
            self.sd_turbo_path, subfolder="scheduler"
        )
        self.sched.set_timesteps(1, device="cpu")

        # Build wrappers
        self._text_enc_wrapper = TextEncoderWrapper(text_encoder)
        self._vae_enc_wrapper = VAEEncoderWrapper(
            vae, vae_lora_layers, self.lora_rank_vae
        )
        self._unet_wrapper = UNetWrapper(unet, unet_lora_layers, self.lora_rank_unet)
        self._vae_dec_wrapper = VAEDecoderWrapper(vae)
        self._de_net = de_net

        # Build modulation closure
        lora_rank_vae = self.lora_rank_vae
        lora_rank_unet = self.lora_rank_unet

        def compute_modulation(deg_score):
            deg_proj = deg_score[..., None] * W[None, None, :] * 2 * np.pi
            deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
            deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

            vae_de_c_embed = vae_de_mlp(deg_proj)
            unet_de_c_embed = unet_de_mlp(deg_proj)

            vae_block_c_embeds = vae_block_mlp(vae_block_embeddings.weight)
            unet_block_c_embeds = unet_block_mlp(unet_block_embeddings.weight)

            B = deg_score.shape[0]
            vae_embeds = vae_fuse_mlp(
                torch.cat(
                    [
                        vae_de_c_embed.unsqueeze(1).repeat(
                            1, vae_block_c_embeds.shape[0], 1
                        ),
                        vae_block_c_embeds.unsqueeze(0).repeat(B, 1, 1),
                    ],
                    -1,
                )
            )
            unet_embeds = unet_fuse_mlp(
                torch.cat(
                    [
                        unet_de_c_embed.unsqueeze(1).repeat(
                            1, unet_block_c_embeds.shape[0], 1
                        ),
                        unet_block_c_embeds.unsqueeze(0).repeat(B, 1, 1),
                    ],
                    -1,
                )
            )

            return (
                vae_embeds.reshape(B, 6, lora_rank_vae, lora_rank_vae),
                unet_embeds.reshape(B, 10, lora_rank_unet, lora_rank_unet),
            )

        self.compute_modulation = compute_modulation

        print(f"VAE LoRA layers: {len(vae_lora_layers)}")
        print(f"UNet LoRA layers: {len(unet_lora_layers)}")
        print("Model loaded successfully.")

    def compile(self):
        """Compile all components with torch_neuronx.trace()."""
        import torch_neuronx

        os.makedirs(self.compile_dir, exist_ok=True)
        lr_h, lr_w = self.lr_size, self.lr_size
        hr_h, hr_w = self.hr_size, self.hr_size
        lat_h, lat_w = hr_h // 8, hr_w // 8

        # DEResNet
        path = os.path.join(self.compile_dir, "de_net.pt")
        if os.path.exists(path):
            print("DEResNet: loading cached...")
            self.de_net_neuron = torch.jit.load(path)
        else:
            print("DEResNet: compiling...")
            t0 = time.time()
            self.de_net_neuron = torch_neuronx.trace(
                self._de_net,
                torch.randn(1, 3, lr_h, lr_w),
                compiler_args=["--auto-cast", "matmult", "-O1"],
            )
            torch.jit.save(self.de_net_neuron, path)
            print(f"  Done in {time.time() - t0:.1f}s")

        # Text encoder
        path = os.path.join(self.compile_dir, "text_encoder.pt")
        if os.path.exists(path):
            print("Text encoder: loading cached...")
            self.text_enc_neuron = torch.jit.load(path)
        else:
            print("Text encoder: compiling...")
            t0 = time.time()
            self.text_enc_neuron = torch_neuronx.trace(
                self._text_enc_wrapper,
                torch.zeros(1, 77, dtype=torch.long),
                compiler_args=["--auto-cast", "matmult", "-O1"],
            )
            torch.jit.save(self.text_enc_neuron, path)
            print(f"  Done in {time.time() - t0:.1f}s")

        # VAE encoder (with LoRA)
        path = os.path.join(self.compile_dir, "vae_encoder.pt")
        if os.path.exists(path):
            print("VAE encoder: loading cached...")
            self.vae_enc_neuron = torch.jit.load(path)
        else:
            print("VAE encoder: compiling...")
            t0 = time.time()
            self.vae_enc_neuron = torch_neuronx.trace(
                self._vae_enc_wrapper,
                (
                    torch.randn(1, 3, hr_h, hr_w),
                    torch.randn(1, 6, self.lora_rank_vae, self.lora_rank_vae),
                ),
                compiler_args=["--model-type=unet-inference", "-O1"],
            )
            torch.jit.save(self.vae_enc_neuron, path)
            print(f"  Done in {time.time() - t0:.1f}s")

        # UNet (with LoRA)
        path = os.path.join(self.compile_dir, "unet.pt")
        if os.path.exists(path):
            print("UNet: loading cached...")
            self.unet_neuron = torch.jit.load(path)
        else:
            print("UNet: compiling...")
            t0 = time.time()
            self.unet_neuron = torch_neuronx.trace(
                self._unet_wrapper,
                (
                    torch.randn(1, 4, lat_h, lat_w),
                    torch.tensor([999], dtype=torch.long),
                    torch.randn(1, 77, 1024),
                    torch.randn(1, 10, self.lora_rank_unet, self.lora_rank_unet),
                ),
                compiler_args=["--model-type=unet-inference", "-O1"],
            )
            torch.jit.save(self.unet_neuron, path)
            print(f"  Done in {time.time() - t0:.1f}s")

        # VAE decoder (no LoRA)
        path = os.path.join(self.compile_dir, "vae_decoder.pt")
        if os.path.exists(path):
            print("VAE decoder: loading cached...")
            self.vae_dec_neuron = torch.jit.load(path)
        else:
            print("VAE decoder: compiling...")
            t0 = time.time()
            self.vae_dec_neuron = torch_neuronx.trace(
                self._vae_dec_wrapper,
                torch.randn(1, 4, lat_h, lat_w),
                compiler_args=["--model-type=unet-inference", "-O1"],
            )
            torch.jit.save(self.vae_dec_neuron, path)
            print(f"  Done in {time.time() - t0:.1f}s")

        print("All components compiled.")

    @torch.no_grad()
    def __call__(
        self,
        lr_image: Image.Image,
        pos_prompt: str = "high quality, highly detailed, clean",
        neg_prompt: str = "blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed",
        cfg_scale: float = 1.07,
    ) -> Image.Image:
        """Run 4x super-resolution on a low-resolution PIL image.

        Args:
            lr_image: Input PIL image (should be lr_size x lr_size)
            pos_prompt: Positive text prompt
            neg_prompt: Negative text prompt
            cfg_scale: Classifier-free guidance scale

        Returns:
            Super-resolved PIL image (4x the input resolution)
        """
        to_tensor = transforms.ToTensor()
        im_lr = to_tensor(lr_image).unsqueeze(0)

        # Preprocess: resize 4x + normalize + pad
        ori_h, ori_w = im_lr.shape[2:]
        im_lr_resize = F.interpolate(
            im_lr,
            size=(ori_h * SF, ori_w * SF),
            mode="bilinear",
            align_corners=False,
        )
        im_lr_resize_norm = (im_lr_resize * 2 - 1.0).clamp(-1, 1)
        resize_h, resize_w = im_lr_resize_norm.shape[2:]
        im_lr_resize_norm = F.pad(
            im_lr_resize_norm,
            pad=(0, PAD_W - resize_w, 0, PAD_H - resize_h),
            mode="reflect",
        )

        # 1. DEResNet -> degradation scores
        deg_score = self.de_net_neuron(im_lr)

        # 2. Compute modulation on CPU
        vae_de_mod_all, unet_de_mod_all = self.compute_modulation(deg_score)

        # 3. Text encoding
        pos_tokens = self.tokenizer(
            pos_prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        neg_tokens = self.tokenizer(
            neg_prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        pos_enc = self.text_enc_neuron(pos_tokens)
        neg_enc = self.text_enc_neuron(neg_tokens)

        # 4. VAE Encode (with de_mod)
        latent = self.vae_enc_neuron(im_lr_resize_norm, vae_de_mod_all)

        # 5. UNet x2 for CFG (with de_mod)
        timestep = torch.tensor([999], dtype=torch.long)
        pos_pred = self.unet_neuron(latent, timestep, pos_enc, unet_de_mod_all)
        neg_pred = self.unet_neuron(latent, timestep, neg_enc, unet_de_mod_all)
        model_pred = neg_pred + cfg_scale * (pos_pred - neg_pred)

        # 6. Scheduler step (CPU)
        x_denoised = self.sched.step(
            model_pred.cpu(), torch.tensor([999]), latent.cpu(), return_dict=True
        ).prev_sample

        # 7. VAE Decode
        output = self.vae_dec_neuron(x_denoised).clamp(-1, 1)
        output = output[:, :, :resize_h, :resize_w]
        return transforms.ToPILImage()((output[0] * 0.5 + 0.5).cpu().clamp(0, 1))
