import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# 启用内存优化
torch.cuda.empty_cache()

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

# 使用 CPU offload 来节省显存
pipeline.enable_model_cpu_offload()
pipeline.set_progress_bar_config(disable=None)

image1 = Image.open("image1.png").convert("RGB")
image2 = Image.open("image2.png").convert("RGB")
prompt = "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。"

print("\n" + "="*80)
print("Pipeline Shape Analysis for QwenImageEditPlusPipeline")
print("="*80)

# ===== Step 1: Input images =====
print("\n[Step 1] Input Images")
print(f"  image1.size (W, H): {image1.size}")
print(f"  image2.size (W, H): {image2.size}")

# ===== Step 2: Calculate dimensions =====
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE, VAE_IMAGE_SIZE

image_size = image2.size  # last image determines output size
calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
vae_scale_factor = pipeline.vae_scale_factor
multiple_of = vae_scale_factor * 2
width = calculated_width // multiple_of * multiple_of
height = calculated_height // multiple_of * multiple_of

print("\n[Step 2] Dimension Calculation")
print(f"  vae_scale_factor: {vae_scale_factor}")
print(f"  Target output size (W, H): ({width}, {height})")

# ===== Step 3: Preprocess images =====
print("\n[Step 3] Image Preprocessing")
images = [image1, image2]
condition_images = []
vae_images = []
vae_image_sizes = []

for i, img in enumerate(images):
    image_width, image_height = img.size
    condition_width, condition_height = calculate_dimensions(CONDITION_IMAGE_SIZE, image_width / image_height)
    vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

    condition_img = pipeline.image_processor.resize(img, condition_height, condition_width)
    vae_img = pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2)

    condition_images.append(condition_img)
    vae_images.append(vae_img)
    vae_image_sizes.append((vae_width, vae_height))

    print(f"  Image {i+1}:")
    print(f"    Original size: {img.size}")
    print(f"    Condition image size (for text encoder): ({condition_width}, {condition_height})")
    print(f"    VAE input size: ({vae_width}, {vae_height})")
    print(f"    VAE preprocessed tensor shape: {vae_img.shape}  # (B, C, T, H, W)")

# ===== Step 4: Text Encoding =====
print("\n[Step 4] Text Encoding (Qwen2.5-VL)")
device = pipeline._execution_device

with torch.inference_mode():
    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
        image=condition_images,
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    print(f"  prompt_embeds shape: {prompt_embeds.shape}  # (B, seq_len, hidden_dim)")
    print(f"  prompt_embeds_mask shape: {prompt_embeds_mask.shape}  # (B, seq_len)")

    # Negative prompt
    negative_prompt = " "
    negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
        image=condition_images,
        prompt=negative_prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    print(f"  negative_prompt_embeds shape: {negative_prompt_embeds.shape}")
    print(f"  negative_prompt_embeds_mask shape: {negative_prompt_embeds_mask.shape}")

torch.cuda.empty_cache()

# ===== Step 5: VAE Encoding =====
print("\n[Step 5] VAE Encoding")
num_channels_latents = pipeline.transformer.config.in_channels // 4
print(f"  num_channels_latents: {num_channels_latents}")

with torch.inference_mode():
    generator = torch.manual_seed(0)
    latents, image_latents = pipeline.prepare_latents(
        vae_images,
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    print(f"  Noise latents shape (packed): {latents.shape}  # (B, H/2*W/2, C*4)")
    print(f"  Image latents shape (packed): {image_latents.shape}  # (B, num_imgs*h/2*w/2, C*4)")

    # Calculate unpacked shapes
    latent_height = 2 * (int(height) // (vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (vae_scale_factor * 2))
    print(f"  Unpacked latent spatial size: ({latent_height}, {latent_width})")

    # ===== Step 6: Transformer Input Preparation =====
    print("\n[Step 6] Transformer Input")
    latent_model_input = torch.cat([latents, image_latents], dim=1)
    print(f"  Combined latent input shape: {latent_model_input.shape}  # (B, noise_seq + img_seq, C*4)")

    img_shapes = [
        [
            (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
            *[
                (1, vae_height // vae_scale_factor // 2, vae_width // vae_scale_factor // 2)
                for vae_width, vae_height in vae_image_sizes
            ],
        ]
    ]
    print(f"  img_shapes: {img_shapes}")

    # ===== Step 7: Transformer Forward =====
    print("\n[Step 7] Transformer Forward (theoretical shapes)")
    print(f"  Input hidden_states shape: {latent_model_input.shape}")
    print(f"  Input encoder_hidden_states shape: {prompt_embeds.shape}")
    print(f"  Input encoder_hidden_states_mask shape: {prompt_embeds_mask.shape}")
    print(f"  Expected output shape: {latent_model_input.shape}  # same as input")
    print(f"  Noise prediction (cropped) shape: ({latents.shape[0]}, {latents.shape[1]}, {latents.shape[2]})  # only noise part")

    # ===== Step 8: VAE Decoding =====
    print("\n[Step 8] VAE Decoding (shape info)")
    # Unpack latents for VAE
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline as Pipeline
    unpacked_latents = Pipeline._unpack_latents(latents, height, width, vae_scale_factor)
    print(f"  Unpacked latents shape: {unpacked_latents.shape}  # (B, C, T, H, W)")

    # Denormalize
    latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, pipeline.vae.config.z_dim, 1, 1, 1).to(device, unpacked_latents.dtype)
    latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(1, pipeline.vae.config.z_dim, 1, 1, 1).to(device, unpacked_latents.dtype)
    denorm_latents = unpacked_latents / latents_std + latents_mean
    print(f"  Denormalized latents shape: {denorm_latents.shape}")

    decoded = pipeline.vae.decode(denorm_latents.to(pipeline.vae.dtype), return_dict=False)[0]
    print(f"  VAE decoded output shape: {decoded.shape}  # (B, C, T, H, W)")
    print(f"  Final image tensor shape: {decoded[:, :, 0].shape}  # (B, C, H, W)")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY: Pipeline Flow")
print("="*80)
print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QwenImageEditPlusPipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT IMAGES                                                               │
│  ├── image1: PIL.Image (W1, H1)                                             │
│  └── image2: PIL.Image (W2, H2)                                             │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ IMAGE PREPROCESSING                                                  │   │
│  │ ├── Condition images (for text encoder): resize to ~384x384 area    │   │
│  │ └── VAE images: resize to ~1024x1024 area, shape (B, 3, 1, H, W)    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ├──────────────────────┬──────────────────────────────────────    │
│           ▼                      ▼                                          │
│  ┌──────────────────────┐  ┌──────────────────────────────────────────┐    │
│  │ TEXT ENCODER         │  │ VAE ENCODER                              │    │
│  │ (Qwen2.5-VL-7B)      │  │ (AutoencoderKLQwenImage)                 │    │
│  │                      │  │                                          │    │
│  │ Input:               │  │ Input: (B, 3, 1, H, W)                   │    │
│  │  - prompt + images   │  │ Output: (B, 16, 1, H/8, W/8)             │    │
│  │                      │  │                                          │    │
│  │ Output:              │  │ Then PACK:                               │    │
│  │  - prompt_embeds     │  │ (B, H/16*W/16, 64)                       │    │
│  │    (B, seq, 3584)    │  │                                          │    │
│  └──────────────────────┘  └──────────────────────────────────────────┘    │
│           │                      │                                          │
│           │                      │  + Random noise latents (packed)         │
│           │                      │    (B, H/16*W/16, 64)                    │
│           │                      │                                          │
│           │                      ▼                                          │
│           │              ┌──────────────────────────────────────────┐       │
│           │              │ CONCAT: [noise, img1_lat, img2_lat]      │       │
│           │              │ Shape: (B, total_seq, 64)                │       │
│           │              └──────────────────────────────────────────┘       │
│           │                      │                                          │
│           └──────────────────────┼──────────────────────────────────────    │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TRANSFORMER (QwenImageTransformer2DModel) - Denoising Loop x N     │   │
│  │                                                                     │   │
│  │ Input:                                                              │   │
│  │  - hidden_states: (B, total_seq, 64)                               │   │
│  │  - encoder_hidden_states: (B, text_seq, 3584)                      │   │
│  │  - timestep, guidance, img_shapes                                  │   │
│  │                                                                     │   │
│  │ Output:                                                             │   │
│  │  - noise_pred: (B, total_seq, 64) → crop to (B, noise_seq, 64)     │   │
│  │                                                                     │   │
│  │ Scheduler step: latents = scheduler.step(noise_pred, t, latents)   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ UNPACK & VAE DECODE                                                 │   │
│  │                                                                     │   │
│  │ Unpack: (B, H/16*W/16, 64) → (B, 16, 1, H/8, W/8)                  │   │
│  │ VAE decode: (B, 16, 1, H/8, W/8) → (B, 3, 1, H, W)                 │   │
│  │ Extract frame: (B, 3, H, W)                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│                         OUTPUT: PIL.Image (W, H)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "="*80)
print("Key Dimensions Reference")
print("="*80)
print(f"""
VAE:
  - vae_scale_factor: {vae_scale_factor}
  - latent_channels (z_dim): {pipeline.vae.config.z_dim}
  - Compression: Image (H, W) → Latent (H/{vae_scale_factor}, W/{vae_scale_factor})

Transformer:
  - in_channels: {pipeline.transformer.config.in_channels}
  - Packed channels: {pipeline.transformer.config.in_channels // 4} * 4 = {pipeline.transformer.config.in_channels}
  - Patch size: 2x2 (latents packed into 2x2 patches)

Text Encoder (Qwen2.5-VL-7B):
  - Hidden size: 3584
  - Max sequence length: 1024
""")

print("\nDebug script completed!")
