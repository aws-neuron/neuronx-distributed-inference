#!/usr/bin/env python3
import os
import re 
import math
import random
from typing import Dict, Tuple, Optional, List
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


pio.renderers.default = "browser"

# ---- config ----
MODULE_NAME_EQUIV = {
    "mlp.gate": "mlp.router.linear_router"
}

# ---- filename parsing ----
# Examples:
# captured_tensors_cte_step_1_module_layers.0.mlp.router.linear_router_output.pt
# captured_tensors_ctx_step_1_module_layers.3.mlp.gate_output.pt
# captured_tensors_tkg_step_9_module_layers.3.mlp.router.linear_router_output.pt
_FNAME_RE = re.compile(
    r"""^captured_tensors_
        (?P<phase>cte|ctx|tkg)
        _step_(?P<step>\d+)
        _module_(?P<fullkey>.+?)
        (?:_(?:output|inputs?)|\.outputs?)?   # tolerate _output/.outputs, etc.
        \.pt$
    """, re.X
)

def _parse(fn: str) -> Optional[Tuple[str, int, str]]:
    m = _FNAME_RE.match(fn)
    if not m:
        return None
    phase = m.group("phase")
    phase = "ctx" if phase in ("ctx", "cte") else "tkg"  # normalize cte→ctx
    step  = int(m.group("step"))
    full  = m.group("fullkey")  # e.g., layers.0.mlp.gate or layers.0.mlp.router.linear_router
    return phase, step, full

def _map_cpu_key(fullkey: str) -> str:
    """
    Map CPU suffix to Neuron suffix. Input is 'layers.N.<suffix...>'.
    """
    # split off 'layers.N.'
    if not fullkey.startswith("layers."):
        return fullkey
    parts = fullkey.split(".", 2)  # ["layers", "<N>", "<rest>"]
    if len(parts) < 3:
        return fullkey
    head = ".".join(parts[:2])     # "layers.N"
    rest = parts[2]                # "<rest>"
    for src, dst in MODULE_NAME_EQUIV.items():
        if rest == src or rest.startswith(src + "."):
            rest = rest.replace(src, dst, 1)
            break
    return f"{head}.{rest}"

# ---- indexing ----
def index_dir(root: str, is_cpu: bool) -> Dict[Tuple[str, int, str], str]:
    """
    Map (phase, step, canon_key) -> filepath
    canon_key is the full 'layers.N.<suffix...>' form (already mapped for CPU).
    """
    root = os.path.expanduser(root)
    out = {}
    for fn in os.listdir(root):
        if not fn.endswith(".pt"): 
            continue
        parsed = _parse(fn)
        if not parsed:
            continue
        phase, step, full = parsed
        if is_cpu:
            full = _map_cpu_key(full)
        out[(phase, step, full)] = os.path.join(root, fn)
    return out

# ---- I/O helpers ----
def _load(path: str) -> Optional[torch.Tensor]:
    try:
        t = torch.load(path, map_location="cpu")
        return t if isinstance(t, torch.Tensor) else None
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None

def _flatten(t: torch.Tensor, max_elems: int = 4096) -> torch.Tensor:
    if t.is_sparse:
        t = t.to_dense()
    t = t.reshape(-1).to(torch.float32).cpu()
    return t[:max_elems] if t.numel() > max_elems else t

# ---- shape alignment rules ----
def align_ctx(cpu: torch.Tensor, neu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU: [T,H] -> [1,T,H]
    NEU: [1,512,H] -> [1,T,H] (truncate T)
    """
    if cpu.dim() != 2:
        raise ValueError(f"Expected CPU CTX rank 2 [T,H], got {cpu.shape}")
    if neu.dim() != 3:
        raise ValueError(f"Expected NEU CTX rank 3 [1,*,H], got {neu.shape}")
    T, H = cpu.shape
    cpu_aligned = cpu.unsqueeze(0)            # [1,T,H]
    neu_aligned = neu[:, :T, :]               # truncate to T  -> [1,T,H]
    return cpu_aligned, neu_aligned

def align_tkg(cpu: torch.Tensor, neu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU: [1,H] -> [1,1,H]
    NEU: [1,1,H] (as-is)
    """
    if cpu.dim() == 1:
        cpu = cpu.unsqueeze(0)                # [H] -> [1,H]
    if cpu.dim() != 2:
        raise ValueError(f"Expected CPU TKG rank 2 [1,H], got {cpu.shape}")
    if neu.dim() != 3:
        raise ValueError(f"Expected NEU TKG rank 3 [1,1,H], got {neu.shape}")
    cpu_aligned = cpu.unsqueeze(1)            # [1,1,H]
    return cpu_aligned, neu

# ---- diff metrics ----
def diff_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    d = (a - b).to(torch.float32).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt((d**2).mean()).item()),
    }

# ---- plotting ----
def _grid(nplots: int) -> Tuple[int, int]:
    if nplots <= 6:   
        ncols = 2
    elif nplots <= 12:
        ncols = 3
    elif nplots <= 20:
        ncols = 4
    else:             
        ncols = 5
    nrows = (nplots + ncols - 1) // ncols
    return nrows, ncols


def _scatter_tkg_by_step(tkg_ready, title: str, max_points: int = 4096):
    """
    tkg_ready: list of (phase, step, name, cpu_aligned, neu_aligned)
    One row per step; columns = #modules for that step.
    """
    if not tkg_ready:
        return None

    # group by step
    by_step: Dict[int, List[Tuple[str,int,str,torch.Tensor,torch.Tensor]]] = {}
    for rec in tkg_ready:
        _, step, _, _, _ = rec
        by_step.setdefault(step, []).append(rec)

    steps = sorted(by_step.keys())
    # sort modules within each step by name for a stable layout
    for s in steps:
        by_step[s].sort(key=lambda x: x[2])  # name

    # columns = max modules in any step
    nrows = len(steps)
    ncols = max(len(by_step[s]) for s in steps)

    # spacing that won’t violate Plotly’s constraints
    vspace = 0.06 if nrows <= 1 else min(0.04, (1.0/(nrows-1))-1e-6)
    hspace = 0.08 if ncols <= 1 else min(0.08, (1.0/(ncols-1))-1e-6)

    # Subplot titles: module names only (no step) to avoid overlap
    subplot_titles = []
    for s in steps:
        names = [rec[2] for rec in by_step[s]]
        # pad with blanks if this row has fewer modules than max
        names += [""] * (ncols - len(names))
        subplot_titles.extend(names)

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=vspace, horizontal_spacing=hspace
    )

    # draw points
    idx = 0
    for r, s in enumerate(steps, start=1):
        row_items = by_step[s]
        for c in range(1, ncols+1):
            if c <= len(row_items):
                phase, step, name, a3, b3 = row_items[c-1]
                a = _flatten(a3, max_points)
                b = _flatten(b3, max_points)
                m = min(a.numel(), b.numel())
                if m == 0:
                    continue
                a = a[:m] 
                b = b[:m]
                ms = 6
                if m <= 64:  
                    ms = max(ms, 8)
                if m <= 8:   
                    ms = max(ms, 12)
                fig.add_trace(
                    go.Scattergl(
                        x=a.numpy(), y=b.numpy(),
                        mode='markers',
                        marker=dict(size=ms, opacity=0.6),
                        name=f"step {step} · {name}",
                        showlegend=False,
                    ),
                    row=r, col=c
                )
                lo = float(min(a.min().item(), b.min().item()))
                hi = float(max(a.max().item(), b.max().item()))
                if lo == hi:
                    lo -= 1.0
                    hi += 1.0
                fig.add_shape(
                    type='line', x0=lo, y0=lo, x1=hi, y1=hi,
                    line=dict(color='red', dash='dash'),
                    row=r, col=c
                )
            idx += 1

        # add a clear row label “step N” on the left margin
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.0, y=1 - (r-0.5)/max(1, nrows),  # left side, centered on the row
            xanchor="right", yanchor="middle",
            text=f"<b>step {s}</b>",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0.03)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            borderpad=3,
        )

    # shrink subplot title font a bit so they don’t collide
    fig.update_layout(
        height=max(300*nrows, 350), width=1300,
        title=title, showlegend=False,
        margin=dict(t=60, l=90, r=30, b=40),  # extra left margin for row labels
    )
    for ann in fig.layout.annotations:
        if ann.text and "layers." in ann.text:  # subplot titles are module names
            ann.font.size = 10

    return fig


def _scatter_fig(pairs: List[Tuple[str,int,str,torch.Tensor,torch.Tensor]],
                 title: str, max_points: int = 4096):
    if not pairs:
        return None
    nrows, ncols = _grid(len(pairs))
    vspace = 0.06 if nrows <= 1 else min(0.04, (1.0/(nrows-1))-1e-6)
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f"{p[0]} s={p[1]} · {p[2]}" for p in pairs],
        vertical_spacing=vspace, horizontal_spacing=0.08
    )
    for i, (phase, step, name, a3, b3) in enumerate(pairs):
        row = i // ncols + 1
        col = i % ncols + 1
        a = _flatten(a3, max_points)
        b = _flatten(b3, max_points)
        m = min(a.numel(), b.numel())
        if m == 0: 
            continue
        a = a[:m]
        b = b[:m]
        ms = 2
        if phase == "tkg":
            ms = max(ms, 6)
            if m <= 64:  
                ms = max(ms, 8)
            if m <= 8:   
                ms = max(ms, 12)
        fig.add_trace(go.Scattergl(
            x=a.numpy(), y=b.numpy(),
            mode='markers',
            marker=dict(size=ms, opacity=0.6),
            name=f"{phase} s{step} {name}",
            showlegend=False
        ), row=row, col=col)
        lo = float(min(a.min().item(), b.min().item()))
        hi = float(max(a.max().item(), b.max().item()))
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(color='red', dash='dash'),
                      row=row, col=col)
    fig.update_layout(height=350*nrows, width=1300, title=title, showlegend=False,
                      margin=dict(t=60, l=40, r=30, b=40))
    return fig

# ---- main compare ----
def compare_and_plot(cpu_dir: str,
                     neuron_dir: str,
                     tkg_steps_to_plot: Optional[List[int]] = None,
                     tkg_max_random: int = 12,
                     max_points: int = 4096,
                     rtol: float = 1e-3,
                     atol: float = 1e-3):
    cpu_idx = index_dir(cpu_dir, is_cpu=True)
    neu_idx = index_dir(neuron_dir, is_cpu=False)

    keys = sorted(set(cpu_idx.keys()) | set(neu_idx.keys()),
                  key=lambda k: (k[0], k[1], k[2]))

    # report missing
    missing = [k for k in keys if (k in cpu_idx) ^ (k in neu_idx)]
    for phase, step, name in missing:
        side = "CPU" if (phase,step,name) not in cpu_idx else "Neuron"
        print(f"[MISSING] {phase} step={step} {name} — missing on {side}")

    # separate CTX and TKG pairs we actually have on both sides
    pairs = [k for k in keys if k in cpu_idx and k in neu_idx]
    ctx_pairs = [k for k in pairs if k[0] == "ctx"]
    tkg_pairs = [k for k in pairs if k[0] == "tkg"]

    # optionally downselect TKG steps for plotting
    if tkg_steps_to_plot is None:
        # pick up to tkg_max_random random steps that exist
        steps = sorted({s for (_, s, _) in tkg_pairs})
        if len(steps) > tkg_max_random:
            steps = sorted(random.sample(steps, tkg_max_random))
    else:
        steps = sorted(tkg_steps_to_plot)

    tkg_pairs = [k for k in tkg_pairs if k[1] in steps]

    # Load, align, diff
    ctx_ready = []
    tkg_ready = []
    n_ok = n_shape = n_dtype = n_value = 0

    for phase, step, name in ctx_pairs:
        a = _load(cpu_idx[(phase, step, name)])   # CPU
        b = _load(neu_idx[(phase, step, name)])   # NEU
        if a is None or b is None:
            print(f"[SKIP] load failure: {phase} {step} {name}")
            continue
        a_al, b_al = align_ctx(a, b)
        if a_al.dtype != b_al.dtype:
            print(f"[DTYPE] ctx step={step} {name}: CPU {a_al.dtype} -> cast {b_al.dtype}")
            a_al = a_al.to(b_al.dtype)
            n_dtype += 1
        if tuple(a_al.shape) != tuple(b_al.shape):
            print(f"[SHAPE] ctx step={step} {name}: CPU{tuple(a.shape)}→{tuple(a_al.shape)} != NEU{tuple(b_al.shape)}")
            n_shape += 1
        else:
            if not torch.allclose(a_al, b_al, rtol=rtol, atol=atol):
                dm = diff_metrics(a_al, b_al)
                print(f"[VALUE] ctx step={step} {name} not close "
                      f"(rtol={rtol}, atol={atol}) max|Δ|={dm['max_abs']:.3e} "
                      f"mean|Δ|={dm['mean_abs']:.3e} rmse={dm['rmse']:.3e}")
                n_value += 1
            else:
                n_ok += 1
        ctx_ready.append((phase, step, name, a_al, b_al))

    for phase, step, name in tkg_pairs:
        a = _load(cpu_idx[(phase, step, name)])
        b = _load(neu_idx[(phase, step, name)])
        if a is None or b is None:
            print(f"[SKIP] load failure: {phase} {step} {name}")
            continue
        a_al, b_al = align_tkg(a, b)
        if a_al.dtype != b_al.dtype:
            print(f"[DTYPE] tkg step={step} {name}: CPU {a_al.dtype} -> cast {b_al.dtype}")
            a_al = a_al.to(b_al.dtype)
            n_dtype += 1
        if tuple(a_al.shape) != tuple(b_al.shape):
            print(f"[SHAPE] tkg step={step} {name}: CPU{tuple(a.shape)}→{tuple(a_al.shape)} != NEU{tuple(b_al.shape)}")
            n_shape += 1
        else:
            if not torch.allclose(a_al, b_al, rtol=rtol, atol=atol):
                dm = diff_metrics(a_al, b_al)
                print(f"[VALUE] tkg step={step} {name} not close "
                      f"(rtol={rtol}, atol={atol}) max|Δ|={dm['max_abs']:.3e} "
                      f"mean|Δ|={dm['mean_abs']:.3e} rmse={dm['rmse']:.3e}")
                n_value += 1
            else:
                n_ok += 1
        tkg_ready.append((phase, step, name, a_al, b_al))

    print("\n=== Summary ===")
    print(f"Paired tensors      : {len(ctx_ready) + len(tkg_ready)} "
          f"(ctx={len(ctx_ready)}, tkg={len(tkg_ready)})")
    print(f"OK                  : {n_ok}")
    print(f"Shape mismatches    : {n_shape}")
    print(f"Dtype casts         : {n_dtype}")
    print(f"Value mismatches    : {n_value}")
    print(f"Missing (unpaired)  : {len(missing)}")

    # Plot CTX (one figure)
    if ctx_ready:
        fig_ctx = _scatter_fig(ctx_ready, "CTX — CPU vs Neuron", max_points)
        if fig_ctx: 
            fig_ctx.show()

    # Plot chosen TKG steps (one figure)
    if tkg_ready:
        # Sort by (step, name) for a nicer layout
        tkg_ready.sort(key=lambda p: (p[1], p[2]))
        fig_tkg = _scatter_tkg_by_step(tkg_ready, "TKG (subset) — CPU vs Neuron", max_points)
        if fig_tkg: 
            fig_tkg.show()


if __name__ == "__main__":
    cpu_dir = "~/qwen3-09-11/tensor_capture/cpu/config_4layer-bfloat16/c257fee5d2120a35a99729ded40e39820b02a60121be818e8f2639ceaf6a8917/"
    neuron_dir = "~/qwen3-09-11/accuracy/neuron/config_4layer-bfloat16/c257fee5d2120a35a99729ded40e39820b02a60121be818e8f2639ceaf6a8917"

    # choose a few TKG steps to visualize; if None, the script samples up to 12 randomly
    steps_to_plot = [4, 25, 40, 100, 254, 409]
    compare_and_plot(cpu_dir, neuron_dir, tkg_steps_to_plot=steps_to_plot, max_points=4096)
