import os
import time
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# ===== 配置 =====
NUM_WARMUP = 1      # warmup 次数
NUM_RUNS = 3        # 正式计时运行次数
NUM_INFERENCE_STEPS = 40

# ===== 加载模型 =====
load_start = time.perf_counter()
pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
load_end = time.perf_counter()
print(f"Pipeline loaded in {load_end - load_start:.2f}s")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=True)  # 禁用进度条以获得更准确的计时

# ===== 加载图片 =====
image1 = Image.open("image1.png").convert("RGB")
image2 = Image.open("image2.png").convert("RGB")
prompt = "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而隆重。"

def run_inference(seed=0):
    """运行一次推理"""
    inputs = {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
    return output.images[0]

# ===== Warmup =====
print(f"\n{'='*60}")
print(f"Warmup ({NUM_WARMUP} run(s))...")
print('='*60)

for i in range(NUM_WARMUP):
    warmup_start = time.perf_counter()
    torch.cuda.synchronize()
    _ = run_inference(seed=i)
    torch.cuda.synchronize()
    warmup_end = time.perf_counter()
    print(f"  Warmup {i+1}: {warmup_end - warmup_start:.2f}s")

# 清理 GPU 缓存
torch.cuda.empty_cache()

# ===== 正式计时运行 =====
print(f"\n{'='*60}")
print(f"Timed runs ({NUM_RUNS} run(s))...")
print('='*60)

times = []
for i in range(NUM_RUNS):
    torch.cuda.synchronize()
    start = time.perf_counter()

    output_image = run_inference(seed=100 + i)

    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}s")

# ===== 统计结果 =====
print(f"\n{'='*60}")
print("Timing Summary")
print('='*60)
print(f"  Inference steps: {NUM_INFERENCE_STEPS}")
print(f"  Total runs: {NUM_RUNS}")
print(f"  Mean time: {sum(times)/len(times):.2f}s")
print(f"  Min time: {min(times):.2f}s")
print(f"  Max time: {max(times):.2f}s")
if len(times) > 1:
    import statistics
    print(f"  Std dev: {statistics.stdev(times):.2f}s")
print(f"  Throughput: {NUM_INFERENCE_STEPS / (sum(times)/len(times)):.2f} steps/s")

# ===== 保存最后一张图片 =====
output_image.save("output_image_edit_plus.png")
print(f"\nImage saved at {os.path.abspath('output_image_edit_plus.png')}")

# ===== GPU 内存使用 =====
print(f"\n{'='*60}")
print("GPU Memory Usage")
print('='*60)
print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
