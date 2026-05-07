# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# This implementation is derived from the Diffusers library.
# The original codebase has been optimized and modified to achieve optimal performance
# characteristics when executed on Amazon Neuron devices.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline, FluxFillPipeline, FluxControlPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.getLogger(__name__)


class NeuronFluxPipeline(FluxPipeline):
    """
    Neuron-optimized Flux pipeline with parallel CFG support.

    This pipeline overrides __call__ to implement batched CFG inference,
    reducing transformer calls from 2 to 1 per timestep when using CFG.

    When do_true_cfg is True (true_cfg_scale > 1 and negative_prompt provided):
    - Positive and negative prompts are batched together
    - Single transformer forward pass processes both
    - Outputs are split and CFG formula is applied
    - Halves the number of transformer calls during denoising
    """

    @functools.wraps(FluxPipeline.encode_prompt)
    def encode_prompt(self, *args, **kwargs):
        assert kwargs.get("lora_scale") is None, "NeuronFluxPipeline does not support LoRA."
        return super().encode_prompt(*args, **kwargs)

    @functools.wraps(FluxPipeline.__call__)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[Any] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Any] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        Override of FluxPipeline.__call__ with parallel CFG batching.

        When CFG is enabled (true_cfg_scale > 1 and negative_prompt provided),
        this implementation batches positive and negative inputs for parallel inference.
        """
        # Validate Neuron-specific constraints
        assert ip_adapter_image is None, "NeuronFluxPipeline does not support ip_adapter_image input."
        assert ip_adapter_image_embeds is None, "NeuronFluxPipeline does not support ip_adapter_image_embeds input."
        assert negative_ip_adapter_image is None, "NeuronFluxPipeline does not support negative_ip_adapter_image input."
        assert negative_ip_adapter_image_embeds is None, "NeuronFluxPipeline does not support negative_ip_adapter_image_embeds input."

        # Check if CFG is enabled and if parallel CFG is configured
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        cfg_parallel_enabled = getattr(self.transformer.config, 'cfg_parallel_enabled', False)

        # Only use parallel CFG if both CFG is enabled AND cfg_parallel_enabled is configured
        use_parallel_cfg = do_true_cfg and cfg_parallel_enabled

        if not use_parallel_cfg:
            # No CFG - use standard FluxPipeline implementation
            with self.transformer.image_rotary_emb_cache_context():
                return super().__call__(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt_2,
                    true_cfg_scale=true_cfg_scale,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    sigmas=sigmas,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    ip_adapter_image=ip_adapter_image,
                    ip_adapter_image_embeds=ip_adapter_image_embeds,
                    negative_ip_adapter_image=negative_ip_adapter_image,
                    negative_ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    output_type=output_type,
                    return_dict=return_dict,
                    joint_attention_kwargs=joint_attention_kwargs,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    max_sequence_length=max_sequence_length,
                )

        # CFG is enabled - implement parallel batched inference
        with self.transformer.image_rotary_emb_cache_context():
            return self._call_with_parallel_cfg(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                true_cfg_scale=true_cfg_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                joint_attention_kwargs=joint_attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

    def _call_with_parallel_cfg(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        Parallel CFG implementation - batches positive and negative prompts.

        This method replicates FluxPipeline.__call__ but modifies the denoising
        loop to use batched transformer calls.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode positive prompt
        lora_scale = self.joint_attention_kwargs.get("scale", None)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Encode negative prompt (for CFG)
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            _,
        ) = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 7. Prepare guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 8. Batch embeddings for parallel CFG inference
        # Concatenate: [negative, positive]
        batched_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        batched_pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # Batch guidance if present
        if guidance is not None:
            batched_guidance = torch.cat([guidance, guidance], dim=0)
        else:
            batched_guidance = None

        # 9. Denoising loop with parallel CFG
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Batch latents for this timestep: [negative, positive]
                batched_latents = torch.cat([latents, latents], dim=0)

                # Broadcast timestep to batch dimension
                timestep = t.expand(batched_latents.shape[0]).to(batched_latents.dtype)

                # Single batched transformer call
                transformer_output = self.transformer(
                    hidden_states=batched_latents,
                    timestep=timestep / 1000,
                    guidance=batched_guidance,
                    pooled_projections=batched_pooled_embeds,
                    encoder_hidden_states=batched_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )
                batched_noise_pred = transformer_output[0] if isinstance(transformer_output, (tuple, list)) else transformer_output

                # Split batched output and apply CFG formula
                # Debug: check shapes before split
                chunk_size = batched_noise_pred.shape[0] // 2
                neg_noise_pred = batched_noise_pred[:chunk_size]
                pos_noise_pred = batched_noise_pred[chunk_size:]
                noise_pred = neg_noise_pred + true_cfg_scale * (pos_noise_pred - neg_noise_pred)

                # Compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 10. Decode latents
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


class NeuronFluxFillPipeline(FluxFillPipeline):
    @functools.wraps(FluxFillPipeline.encode_prompt)
    def encode_prompt(self, *args, **kwargs):
        assert (
            kwargs.get("lora_scale") is None
        ), "NeuronFluxFillPipeline does not support LoRA."
        return super().encode_prompt(*args, **kwargs)

    @functools.wraps(FluxFillPipeline.__call__)
    def __call__(self, *args, **kwargs):
        assert (
            kwargs.get("ip_adapter_image") is None
        ), "NeuronFluxFillPipeline does not support ip_adapter_image input."
        assert (
            kwargs.get("ip_adapter_image_embeds") is None
        ), "NeuronFluxFillPipeline does not support ip_adapter_image_embeds input."
        assert (
            kwargs.get("negative_ip_adapter_image") is None
        ), "NeuronFluxFillPipeline does not support negative_ip_adapter_image input."
        assert (
            kwargs.get("negative_ip_adapter_image_embeds") is None
        ), "NeuronFluxFillPipeline does not support negative_ip_adapter_image_embeds input."

        with self.transformer.image_rotary_emb_cache_context():
            return super().__call__(*args, **kwargs)


class NeuronFluxControlPipeline(FluxControlPipeline):
    @functools.wraps(FluxControlPipeline.encode_prompt)
    def encode_prompt(self, *args, **kwargs):
        assert (
            kwargs.get("lora_scale") is None
        ), "NeuronFluxControlPipeline does not support LoRA."
        return super().encode_prompt(*args, **kwargs)

    @functools.wraps(FluxControlPipeline.__call__)
    def __call__(self, *args, **kwargs):
        with self.transformer.image_rotary_emb_cache_context():
            return super().__call__(*args, **kwargs)
