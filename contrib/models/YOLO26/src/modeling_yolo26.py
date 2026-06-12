"""YOLO26 Object Detection on AWS Neuron (Trainium2 / Inferentia2).

Compiles Ultralytics YOLO26 detection models for inference on NeuronCores using
``torch_neuronx.trace()``. Supports all 5 detection variants (n/s/m/l/x) plus
segmentation, pose, and OBB task heads.

Architecture
------------
YOLO26 is a 24-layer CNN (Conv2d + BatchNorm + SiLU backbone, FPN/PAN neck,
Detect head) with a small C2PSA self-attention block. The DFL layer is
``nn.Identity()`` (reg_max=1). Models range from 2.4M to 58.9M parameters.

Neuron Strategy
---------------
- ``torch_neuronx.trace()`` with fixed input shape (configurable resolution)
- ``end2end=False`` before ``fuse()`` — ``topk``/``sort`` not supported on trn2
- FP32 for n/s variants; BF16 required for m/l/x (FP32 exceeds SB allocation)
- No ``--auto-cast`` flags — ``matmult`` produces NaN for Conv2d-dominant models
- Data Parallelism across NeuronCores for throughput scaling
- ``--lnc 1`` compiler flag required when running on LNC=1 mode
- Batch sizes > 1 supported (C2PSA .split() bug fixed via .chunk() workaround)
- Layer 0 input channel padding (3→16) for 22% throughput improvement

Layer 0 Padding Optimization
-----------------------------
The first conv layer (Conv2d(3→64, 3×3, stride=2)) wastes 97% of TensorE's
P_DIM=128 tile because C_in=3 gives K_dim=27 (far below 128). The Neuron
compiler falls back to a spill-heavy depthwise path (37% DMA, 7.5MB spill).

Padding the input to 16 channels (K_dim=144≥128) triggers the clean im2col→
matmul path: zero spill, 2 DMA transfers, 22% full-model speedup. This is
enabled by default via ``pad_channels=16``. The padding adds 13 zero channels
to the input and expands Layer 0's weight tensor accordingly — the output is
mathematically identical.

Key Results (trn2.3xlarge, LNC=1, DP=8)
----------------------------------------
- YOLO26s: 1,523 img/s (1.43x vs A10G compiled)
- YOLO26m: 1,267 img/s (2.67x vs A10G compiled)
- YOLO26l: 1,093 img/s (2.95x vs A10G compiled)
- YOLO26x:   876 img/s (4.49x vs A10G compiled)

inf2.xlarge Results (DP=2, 384×640, 16ch pad)
----------------------------------------------
- YOLO26l: ~192 img/s per core, ~405 img/s DP=2 (22% improvement over no-pad)
"""

import os
import time

import torch
import torch.nn as nn

try:
    import torch_neuronx
except ImportError:
    torch_neuronx = None

from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f

try:
    from ultralytics.nn.modules.block import C2PSA as UltralyticsC2PSA
except ImportError:
    UltralyticsC2PSA = None

try:
    from ultralytics.nn.modules.block import Attention as UltralyticsAttention
except ImportError:
    UltralyticsAttention = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_DTYPES = {
    "n": torch.float32,
    "s": torch.float32,
    "m": torch.bfloat16,
    "l": torch.bfloat16,
    "x": torch.bfloat16,
}

INPUT_SHAPE = (3, 640, 640)  # C, H, W

# Optimal input channel padding for Layer 0 on NeuronCores.
# The compiler cannot form efficient TensorE matmuls with C_in=3 (wastes 97%
# of P_DIM=128 tile). Padding to 16 channels triggers the clean im2col->matmul
# path: zero spill, 2 DMA transfers, 1.71x faster Layer 0, 22% faster full model.
# Values 4-12 are WORSE than 3 (trigger pathological compiler code paths).
# Value 32 works but is suboptimal (extra DMA for larger input).
LAYER0_PAD_CHANNELS = 16

DEFAULT_COMPILER_ARGS = []  # No autocast for YOLO26

COSINE_SIM_THRESHOLDS = {
    "n": 0.99,
    "s": 0.99,
    "m": 0.98,
    "l": 0.99,
    "x": 0.99,
}


# ---------------------------------------------------------------------------
# Model Preparation
# ---------------------------------------------------------------------------


def _c2f_forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
    """C2f forward using .chunk() instead of .split().

    The neuronx-cc compiler has a bug where .split((c, c), dim=1) causes
    compilation failure (exit code 70) at batch_size=4 with small spatial
    dimensions (H*W < ~264). Using .chunk(2, 1) — which is semantically
    identical — compiles correctly.

    See: https://github.com/aws-neuron/aws-neuron-sdk/issues/1323
    """
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def _c2psa_forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
    """C2PSA forward using .chunk() instead of .split().

    The neuronx-cc compiler has a bug where .split((c, c), dim=1) combined
    with downstream attention produces incorrect output for odd-indexed batch
    elements (CosSim ~0.08-0.23 vs CPU reference). This is the same .split()
    bug as in C2f but manifests differently: instead of a compilation failure,
    it produces silently wrong results at batch_size >= 2.

    Workaround: use .chunk(2, 1) which is semantically identical to
    .split((c, c), dim=1) but compiles correctly.

    See: https://github.com/aws-neuron/aws-neuron-sdk/issues/1323
    """
    a, b = self.cv1(x).chunk(2, 1)
    b = self.m(b)
    return self.cv2(torch.cat((a, b), 1))


def _attention_forward_fixed(self, x: torch.Tensor) -> torch.Tensor:
    """Fixed Attention.forward that uses slicing instead of .split().

    The Neuron compiler (neuronx-cc) has a bug lowering torch.Tensor.split()
    with unequal split sizes on dim=2 of a 4D tensor after a view/reshape.
    This produces numerically incorrect output (CosSim ~0.45 vs CPU reference).

    Workaround: replace .split([key_dim, key_dim, head_dim], dim=2) with
    explicit tensor slicing, which compiles correctly.

    See: https://github.com/aws-neuron/aws-neuron-sdk/issues/1323
    """
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    qkv_reshaped = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)

    # Use slicing instead of .split() — works around neuronx-cc bug
    q = qkv_reshaped[:, :, : self.key_dim, :]
    k = qkv_reshaped[:, :, self.key_dim : self.key_dim * 2, :]
    v = qkv_reshaped[:, :, self.key_dim * 2 :, :]

    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = attn.softmax(dim=-1)
    x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
    x = self.proj(x)
    return x


def prepare_yolo26(weight_path: str, dtype: torch.dtype = torch.float32) -> nn.Module:
    """Load and prepare a YOLO26 model for Neuron tracing.

    The preparation recipe:
    1. Set ``end2end=False`` before ``fuse()`` (topk unsupported on Neuron)
    2. ``fuse()`` merges BatchNorm into Conv2d, removes training branches
    3. Set ``export=True``, ``dynamic=False`` for clean single-tensor output
    4. Replace ``C2f.forward`` with chunk-based forward (works around
       neuronx-cc compiler bug where ``.split()`` fails at bs=4)
    5. Replace ``C2PSA.forward`` with chunk-based forward (works around
       neuronx-cc bug where ``.split()`` corrupts odd batch elements at bs>=2)
    6. Patch ``Attention.forward`` to use slicing instead of ``.split()``
       (works around neuronx-cc compiler bug with unequal split sizes)
    7. Convert to target dtype if not FP32

    Parameters
    ----------
    weight_path : str
        Path to the ``.pt`` weight file (e.g., ``"yolo26s.pt"``).
    dtype : torch.dtype
        Target dtype. Use ``torch.float32`` for n/s, ``torch.bfloat16`` for m/l/x.

    Returns
    -------
    nn.Module
        Prepared model in eval mode, ready for ``torch_neuronx.trace()``.
    """
    model = YOLO(weight_path)
    pytorch_model = model.model.eval()

    # Disable end2end BEFORE fuse (topk not supported on Neuron)
    detect = pytorch_model.model[-1]
    detect.end2end = False

    pytorch_model = pytorch_model.fuse(verbose=False)

    for m in pytorch_model.modules():
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "dynamic"):
            m.dynamic = False
        if hasattr(m, "format"):
            m.format = "torchscript"
        if hasattr(m, "shape"):
            m.shape = None
        # Fix C2f: use chunk-based forward instead of forward_split to work
        # around neuronx-cc compiler bug where .split() causes compilation
        # failure at bs=4 with small spatial dimensions (exit code 70).
        # chunk(2, 1) is semantically identical to split((c, c), 1).
        if isinstance(m, C2f):
            m.forward = _c2f_forward_chunk.__get__(m, type(m))
        # Fix C2PSA: use chunk-based forward instead of .split() to work
        # around neuronx-cc bug where .split((c,c), 1) combined with
        # downstream attention corrupts odd-indexed batch elements at bs>=2.
        if UltralyticsC2PSA is not None and isinstance(m, UltralyticsC2PSA):
            m.forward = _c2psa_forward_chunk.__get__(m, type(m))
        # Fix C2PSA Attention: replace .split() with slicing to work around
        # neuronx-cc compiler bug with unequal split sizes on dim=2
        if UltralyticsAttention is not None and isinstance(m, UltralyticsAttention):
            m.forward = _attention_forward_fixed.__get__(m, type(m))

    if dtype != torch.float32:
        pytorch_model = pytorch_model.to(dtype)

    return pytorch_model


class _PaddedLayer0Model(nn.Module):
    """Wrapper that pads Layer 0's input channels for better NeuronCore efficiency.

    The Neuron compiler cannot form efficient TensorE matmuls when C_in < 16
    (K_dim = C_in * K * K < P_DIM=128). For C_in=3 (standard RGB), the compiler
    falls back to a spill-heavy depthwise path that wastes 37% of execution time
    on DMA. Padding to 16 channels (K_dim=144 >= 128) enables the clean matmul
    path: zero spill, 22% faster full-model inference.

    This wrapper:
    1. Creates a Conv2d(16, C_out, 3x3) with zero-padded weights
    2. Accepts [B, 16, H, W] input (caller pads RGB to 16 channels)
    3. Runs the padded conv + SiLU as Layer 0
    4. Feeds the output through the remaining layers unchanged
    """

    def __init__(
        self, full_model: nn.Module, target_channels: int = LAYER0_PAD_CHANNELS
    ):
        super().__init__()
        self.full_model = full_model
        self.target_channels = target_channels

        # Extract original Layer 0 (Conv + SiLU)
        layer0 = full_model.model[0]
        orig_conv = layer0.conv
        orig_weight = orig_conv.weight.data  # [C_out, 3, K, K]
        C_out, C_in, K, K2 = orig_weight.shape

        # Create padded conv
        self.padded_conv = nn.Conv2d(
            target_channels,
            C_out,
            kernel_size=K,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )

        # Zero-pad weights: [C_out, 3, K, K] -> [C_out, target_channels, K, K]
        new_weight = torch.zeros(C_out, target_channels, K, K2, dtype=orig_weight.dtype)
        new_weight[:, :C_in, :, :] = orig_weight
        self.padded_conv.weight.data = new_weight

        if orig_conv.bias is not None:
            self.padded_conv.bias.data = orig_conv.bias.data.clone()

        self.act = nn.SiLU()

        # Precompute which layers need their outputs saved for skip connections
        self._save_set = set(full_model.save)
        for i, m in enumerate(full_model.model):
            if isinstance(m.f, list):
                for j in m.f:
                    if j >= 0:
                        self._save_set.add(j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with padded Layer 0.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [B, target_channels, H, W] (zero-padded RGB).
        """
        # Run padded Layer 0
        l0_out = self.act(self.padded_conv(x))

        # Run remaining layers (1 through N)
        model = self.full_model
        y = [None] * len(model.model)
        y[0] = l0_out
        x_cur = l0_out

        for i in range(1, len(model.model)):
            m = model.model[i]
            if m.f != -1:
                if isinstance(m.f, int):
                    x_cur = y[m.f] if m.f >= 0 else x_cur
                else:
                    x_cur = [(y[j] if j >= 0 else x_cur) for j in m.f]
            x_cur = m(x_cur)
            y[i] = x_cur if i in self._save_set else None

        return x_cur


def get_variant_dtype(variant: str) -> torch.dtype:
    """Return the recommended dtype for a YOLO26 variant.

    Parameters
    ----------
    variant : str
        One of ``"n"``, ``"s"``, ``"m"``, ``"l"``, ``"x"``.

    Returns
    -------
    torch.dtype
        ``torch.float32`` for n/s, ``torch.bfloat16`` for m/l/x.
    """
    return VARIANT_DTYPES[variant]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_yolo26(
    weight_path: str,
    batch_size: int = 1,
    dtype: torch.dtype | None = None,
    save_path: str | None = None,
    lnc: int | None = None,
    compiler_args: list[str] | None = None,
    input_shape: tuple[int, int] = (640, 640),
    pad_channels: int | None = LAYER0_PAD_CHANNELS,
) -> torch.jit.ScriptModule:
    """Compile a YOLO26 model for Neuron inference.

    Parameters
    ----------
    weight_path : str
        Path to the ``.pt`` weight file.
    batch_size : int
        Batch size per NeuronCore.
    dtype : torch.dtype, optional
        Override dtype. If ``None``, uses the recommended dtype for the variant.
    save_path : str, optional
        If provided, saves the compiled model to this path.
    lnc : int, optional
        Logical NeuronCore config (1 or 2). Adds ``--lnc`` compiler flag if set.
    compiler_args : list[str], optional
        Additional compiler arguments. Defaults to no autocast flags.
    input_shape : tuple[int, int]
        (H, W) input resolution. Default (640, 640).
    pad_channels : int or None
        Pad Layer 0 input channels for better NeuronCore TensorE utilization.
        Default 16 (22% speedup). Set to ``None`` to disable padding (uses
        standard 3-channel input). Values 4-12 are pathological -- do NOT use.

    Returns
    -------
    torch.jit.ScriptModule
        The traced Neuron model. If ``pad_channels`` is set, the model expects
        input of shape ``[B, pad_channels, H, W]`` with RGB in channels 0-2
        and zeros in channels 3+.
    """
    if torch_neuronx is None:
        raise RuntimeError("torch_neuronx is not installed. Run on a Neuron instance.")

    # Infer variant from weight path
    variant = _infer_variant(weight_path)
    if dtype is None:
        dtype = get_variant_dtype(variant) if variant else torch.float32

    model = prepare_yolo26(weight_path, dtype=dtype)

    H, W = input_shape
    input_channels = pad_channels if pad_channels else 3

    if pad_channels and pad_channels > 3:
        # Wrap model with padded Layer 0 for better TensorE utilization
        model = _PaddedLayer0Model(model, target_channels=pad_channels)

    dummy = torch.randn(batch_size, input_channels, H, W, dtype=dtype)
    with torch.no_grad():
        _ = model(dummy)  # dry run to populate anchors

    args = list(compiler_args) if compiler_args else list(DEFAULT_COMPILER_ARGS)
    if lnc is not None:
        args.extend(["--lnc", str(lnc)])

    traced = torch_neuronx.trace(model, dummy, compiler_args=args)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.jit.save(traced, save_path)

    return traced


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class YOLO26NeuronModel:
    """High-level wrapper for YOLO26 inference on Neuron.

    Handles compilation, caching, data-parallel scaling, and accuracy
    validation in a single class.

    Parameters
    ----------
    variant : str
        One of ``"n"``, ``"s"``, ``"m"``, ``"l"``, ``"x"``.
    batch_size : int
        Batch size per NeuronCore.
    cache_dir : str
        Directory for cached compiled models.
    lnc : int, optional
        Logical NeuronCore config (1 or 2). Detected from environment if not set.
    num_cores : int, optional
        Number of NeuronCores for data parallelism. If > 1, wraps in DataParallel.
    input_shape : tuple[int, int], optional
        (H, W) input resolution. Default (640, 640).
    pad_channels : int or None, optional
        Pad Layer 0 input channels for 22% speedup. Default 16. Set to None to
        disable. The model automatically handles padding in ``__call__``.

    Examples
    --------
    >>> model = YOLO26NeuronModel("l", batch_size=1, num_cores=2, input_shape=(384, 640))
    >>> output = model(torch.randn(2, 3, 384, 640))  # 3ch input — padding is automatic
    >>> output.shape
    torch.Size([2, 84, 5040])
    """

    def __init__(
        self,
        variant: str,
        batch_size: int = 1,
        cache_dir: str = "compiled",
        lnc: int | None = None,
        num_cores: int = 1,
        input_shape: tuple[int, int] = (640, 640),
        pad_channels: int | None = LAYER0_PAD_CHANNELS,
    ):
        self.variant = variant
        self.batch_size = batch_size
        self.dtype = get_variant_dtype(variant)
        self.dtype_name = "bf16" if self.dtype == torch.bfloat16 else "fp32"
        self.num_cores = num_cores
        self.input_shape = input_shape
        self.pad_channels = pad_channels

        if lnc is None:
            lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
        self.lnc = lnc

        weight_path = f"yolo26{variant}.pt"
        H, W = input_shape
        pad_str = f"_pad{pad_channels}" if pad_channels else ""
        neff_name = (
            f"yolo26{variant}_{self.dtype_name}_bs{batch_size}"
            f"_{H}x{W}{pad_str}_lnc{lnc}.pt"
        )
        save_path = os.path.join(cache_dir, neff_name)

        if os.path.exists(save_path):
            self._model = torch.jit.load(save_path)
        else:
            self._model = compile_yolo26(
                weight_path,
                batch_size=batch_size,
                dtype=self.dtype,
                save_path=save_path,
                lnc=lnc,
                input_shape=input_shape,
                pad_channels=pad_channels,
            )

        if num_cores > 1 and torch_neuronx is not None:
            self._model = torch_neuronx.DataParallel(
                self._model,
                device_ids=list(range(num_cores)),
                dim=0,
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference.

        Accepts standard 3-channel RGB input. If ``pad_channels`` is set,
        padding is applied automatically before inference.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Raw detection output ``[B, 84, N]`` where N depends on resolution.
        """
        x = x.to(self.dtype)

        if self.pad_channels and self.pad_channels > 3:
            # Pad input channels: [B, 3, H, W] -> [B, pad_channels, H, W]
            B, C, H, W = x.shape
            padded = torch.zeros(
                B, self.pad_channels, H, W, dtype=x.dtype, device=x.device
            )
            padded[:, :C, :, :] = x
            x = padded

        with torch.no_grad():
            return self._model(x)

    def benchmark(self, warmup: int = 10, iterations: int = 50) -> dict:
        """Measure throughput and latency.

        Returns
        -------
        dict
            Keys: ``p50_ms``, ``p95_ms``, ``p99_ms``, ``throughput_img_s``.
        """
        import numpy as np

        total_bs = self.batch_size * self.num_cores
        H, W = self.input_shape
        dummy = torch.randn(total_bs, 3, H, W, dtype=self.dtype)

        for _ in range(warmup):
            self(dummy)

        latencies = []
        for _ in range(iterations):
            t0 = time.time()
            self(dummy)
            latencies.append((time.time() - t0) * 1000)

        lat = np.array(sorted(latencies))
        p50 = float(np.percentile(lat, 50))
        return {
            "p50_ms": round(p50, 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "p99_ms": round(float(np.percentile(lat, 99)), 2),
            "throughput_img_s": round(total_bs / (p50 / 1000), 1),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_accuracy(
    weight_path: str,
    neuron_model: torch.jit.ScriptModule | YOLO26NeuronModel,
    dtype: torch.dtype | None = None,
    input_shape: tuple[int, int] = (640, 640),
    seed: int = 42,
) -> dict:
    """Compare Neuron output against CPU reference.

    Parameters
    ----------
    weight_path : str
        Path to the ``.pt`` weight file.
    neuron_model : ScriptModule or YOLO26NeuronModel
        The compiled Neuron model.
    dtype : torch.dtype, optional
        Input dtype. Inferred from variant if ``None``.
    input_shape : tuple[int, int]
        (H, W) input resolution. Default (640, 640).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``cosine_similarity``, ``max_error``, ``mean_error``, ``has_nan``.
    """
    variant = _infer_variant(weight_path)
    if dtype is None:
        dtype = get_variant_dtype(variant) if variant else torch.float32

    H, W = input_shape
    torch.manual_seed(seed)
    dummy = torch.randn(1, 3, H, W, dtype=dtype)

    cpu_model = prepare_yolo26(weight_path, dtype=dtype)
    with torch.no_grad():
        cpu_out = cpu_model(dummy)

    if isinstance(neuron_model, YOLO26NeuronModel):
        nrn_out = neuron_model(dummy)
    else:
        with torch.no_grad():
            nrn_out = neuron_model(dummy)

    cpu_flat = cpu_out.flatten().float()
    nrn_flat = nrn_out.flatten().float()

    cossim = torch.nn.functional.cosine_similarity(
        cpu_flat.unsqueeze(0), nrn_flat.unsqueeze(0)
    ).item()

    diff = (cpu_flat - nrn_flat).abs()

    return {
        "cosine_similarity": round(cossim, 6),
        "max_error": round(diff.max().item(), 6),
        "mean_error": round(diff.mean().item(), 6),
        "has_nan": bool(torch.isnan(nrn_out).any().item()),
    }


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def postprocess_detections(
    raw_output: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
) -> list[dict]:
    """Decode raw YOLO26 output into detection results with NMS.

    Since ``topk``/``sort`` are not supported on Neuron, postprocessing runs
    on CPU. This adds ~0.1ms overhead per image at typical detection counts.

    Parameters
    ----------
    raw_output : torch.Tensor
        Raw model output of shape ``[B, 84, N]`` where N is the number of
        anchors (8400 for 640x640 input). First 4 rows are bbox coordinates
        (cx, cy, w, h in pixel space), remaining 80 are COCO class scores.
    conf_threshold : float
        Minimum class confidence to keep a detection.
    iou_threshold : float
        IoU threshold for non-maximum suppression.
    max_detections : int
        Maximum number of detections to return per image.

    Returns
    -------
    list[dict]
        One dict per image in the batch, each containing:
        - ``boxes``: ``[N, 4]`` tensor of (x1, y1, x2, y2) boxes
        - ``scores``: ``[N]`` tensor of confidence scores
        - ``classes``: ``[N]`` tensor of class indices (0-79)
    """
    raw_output = raw_output.float().cpu()
    batch_size = raw_output.shape[0]
    results = []

    for i in range(batch_size):
        pred = raw_output[i]  # [84, N]

        # Split bbox and class scores
        boxes_cxcywh = pred[:4, :]  # [4, N]
        class_scores = pred[4:, :]  # [80, N]

        # Get max class score and class index per anchor
        max_scores, class_ids = class_scores.max(dim=0)  # [N]

        # Filter by confidence
        mask = max_scores >= conf_threshold
        if mask.sum() == 0:
            results.append(
                {
                    "boxes": torch.zeros(0, 4),
                    "scores": torch.zeros(0),
                    "classes": torch.zeros(0, dtype=torch.long),
                }
            )
            continue

        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]
        filtered_boxes = boxes_cxcywh[:, mask]  # [4, M]

        # Convert cx, cy, w, h -> x1, y1, x2, y2
        cx, cy, w, h = (
            filtered_boxes[0],
            filtered_boxes[1],
            filtered_boxes[2],
            filtered_boxes[3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # [M, 4]

        # Apply NMS per class
        keep_indices = _batched_nms(
            boxes_xyxy, filtered_scores, filtered_classes, iou_threshold
        )

        # Limit detections
        if len(keep_indices) > max_detections:
            keep_indices = keep_indices[:max_detections]

        results.append(
            {
                "boxes": boxes_xyxy[keep_indices],
                "scores": filtered_scores[keep_indices],
                "classes": filtered_classes[keep_indices],
            }
        )

    return results


def _batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Class-aware NMS: apply NMS independently per class.

    Parameters
    ----------
    boxes : torch.Tensor
        ``[N, 4]`` boxes in (x1, y1, x2, y2) format.
    scores : torch.Tensor
        ``[N]`` confidence scores.
    classes : torch.Tensor
        ``[N]`` class indices.
    iou_threshold : float
        IoU threshold for suppression.

    Returns
    -------
    torch.Tensor
        Indices of kept detections, sorted by score (descending).
    """
    # Offset boxes by class to prevent cross-class suppression
    max_coord = boxes.max().item() + 1.0
    offsets = classes.float() * max_coord
    offset_boxes = boxes + offsets[:, None]

    # Sort by score descending
    order = scores.argsort(descending=True)
    offset_boxes = offset_boxes[order]
    scores_sorted = scores[order]

    keep = []
    while len(order) > 0:
        idx = order[0]
        keep.append(idx.item())

        if len(order) == 1:
            break

        # Compute IoU of first box with rest
        ious = _compute_iou(offset_boxes[0:1], offset_boxes[1:]).squeeze(0)
        remaining = ious < iou_threshold
        order = order[1:][remaining]
        offset_boxes = offset_boxes[1:][remaining]

    return torch.tensor(keep, dtype=torch.long)


def _compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Parameters
    ----------
    boxes1 : torch.Tensor
        ``[M, 4]`` boxes in (x1, y1, x2, y2) format.
    boxes2 : torch.Tensor
        ``[N, 4]`` boxes in (x1, y1, x2, y2) format.

    Returns
    -------
    torch.Tensor
        ``[M, N]`` IoU matrix.
    """
    x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3:4].T)

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - intersection
    return intersection / (union + 1e-7)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_neuron_core_count() -> int:
    """Detect the number of available NeuronCores."""
    import subprocess
    import json

    try:
        result = subprocess.run(
            ["neuron-ls", "--json-output"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        total = sum(dev.get("nc_count", 0) for dev in data)
        return total
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        OSError,
    ):
        return 0


def _infer_variant(weight_path: str) -> str | None:
    """Infer the YOLO26 variant letter from a weight file path."""
    base = os.path.basename(weight_path).lower()
    for v in ("n", "s", "m", "l", "x"):
        if f"yolo26{v}" in base:
            return v
    return None
