"""
V-JEPA 2 video classification demo.

Classifies a video using the finetuned V-JEPA 2 ViT-L model
on Something-Something v2 (174 action classes).

Runs on CPU. No Neuron hardware needed.

Usage:
    pip install transformers accelerate torchvision decord
    python demo_classify.py                          # sample bowling video
    python demo_classify.py path/to/video.mp4        # your own video
    python demo_classify.py photo.jpg                # static image (repeated as frames)
"""

import sys
import os
import urllib.request
import numpy as np
import torch
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

model_id = "facebook/vjepa2-vitl-fpc16-256-ssv2"


def load_video_frames(source):
    if source.endswith((".jpg", ".jpeg", ".png")):
        from PIL import Image
        img = np.array(Image.open(source).convert("RGB"))
        return np.stack([img] * 16)

    from decord import VideoReader
    vr = VideoReader(source)
    total = len(vr)
    indices = np.linspace(0, total - 1, 16, dtype=int)
    return vr.get_batch(indices).asnumpy()  # (T, H, W, C)


def main():
    # Get video source
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        # Big Buck Bunny — CC-BY-3.0, Blender Foundation
        source = "/tmp/bigbuckbunny_10s.mp4"
        if not os.path.exists(source):
            url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
            print("Downloading Big Buck Bunny sample clip (CC-BY-3.0, Blender Foundation)...")
            urllib.request.urlretrieve(url, source)

    print(f"Loading model {model_id}...")
    model = VJEPA2ForVideoClassification.from_pretrained(model_id)
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model.eval()

    print(f"Loading video: {source}")
    video = load_video_frames(source)
    print(f"Video shape: {video.shape}")

    inputs = processor(video, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    print("\nTop 10 predictions (Something-Something v2 action classes):")
    probs = torch.softmax(logits, dim=-1)
    top10 = probs.topk(10)
    for i, (idx, prob) in enumerate(zip(top10.indices[0], top10.values[0])):
        label = model.config.id2label[idx.item()]
        print(f"  {i+1:2d}. {prob:.1%}  {label}")


if __name__ == "__main__":
    main()
