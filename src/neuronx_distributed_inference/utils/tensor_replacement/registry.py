import os
import re
from neuronx_distributed_inference.modules.attention.utils import order_strided_tensor, stride_tensor
import torch
from typing import Dict, List, Optional, Any
from collections import defaultdict


# ------------------------------------------------------------------------------------
# File-name contract (tensor capture artifacts)
# ------------------------------------------------------------------------------------
# We discover tensors by scanning directories that contain capture outputs. Filenames
# encode phase (cte/ctx/tkg), a step number, and the module path used by the model.
#
# Examples that match:
#   captured_tensors_cte_step_1_module_layers.0.mlp.router.linear_router_output.pt
#   captured_tensors_tkg_step_2_module_layers.7.attn.qkv.inputs.pt
#
# Regex fields:
#  - group(1) == step number (int)
#  - group(2) == module name (e.g., "layers.0.mlp.router.linear_router")
#
# The trailing "._(inputs|outputs).pt" suffix is optional and ignored for identity.
#
_FN = re.compile(
    r"^captured_tensors_(?:cte|ctx|tkg)_step_(\d+)_module_([\w\.]+?)"
    r"(?:[._]outputs?)\.pt$"
)


# ------------------------------------------------------------------------------------
# Shape normalization utilities
# ------------------------------------------------------------------------------------
def _overlap_slices(src_shape, dst_shape):
    """Tuple of slices covering the per-dim overlap between src and dst."""
    return tuple(slice(0, min(s, d)) for s, d in zip(src_shape, dst_shape))


def _ensure_rank_and_pad_ref_to_target(
    t: torch.Tensor,
    neu_t: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize CPU tensor `t` to match a 3D Neuron tensor `neu_t`'s shape [B_neu, S_neu, D_neu].

    Accepted ref shapes:
      • [S, D]                 (single prompt)
      • [S*B, D]               (flattened multi-prompt)
      • [B, S, D]              (already batched)

    Behavior:
      • We reshape 2D refs to 3D using S_neu/B candidates; we never reshape `neu_t`.
      • Overlay rules:
          - If B_ref == B_neu → write overlap for all batches.
          - If B_ref == 1 and B_neu > 1 → write only batch 0.
      • No zero padding — any non-overlap remains from `neu_t`.
    """
    if neu_t.dim() != 3:
        raise ValueError(f"Expected neu_t to be 3D, got dim={neu_t.dim()} with shape {tuple(neu_t.shape)}")

    B_neu, S_neu, D_neu = neu_t.shape
    out = neu_t.to(dtype=t.dtype)

    # Convert t -> 3D if needed
    if t.dim() == 2:
        S_flat, D_ref = t.shape
        if D_ref != D_neu:
            raise ValueError(f"Expected ref {D_ref} and neuron tensors {D_neu} to have same hidden states")
    else:
        raise ValueError(f"Expected ref shape {t.shape} to be 2D")

    if B_neu == 1:
        # Rule 1: just unsqueeze to [1, S, D]
        t = t.unsqueeze(0)
    else:
        # Rule 2: unpack S into B,S using B_neu (source of truth)
        if S_flat % B_neu != 0:
            raise ValueError(
                f"Cannot unpack 2D ref {tuple(t.shape)} into B={B_neu}: "
                f"S_flat={S_flat} is not divisible by B_neu={B_neu}."
            )
        S_ref = S_flat // B_neu
        t = t.reshape(B_neu, S_ref, D_ref)

    if t.shape[0] == B_neu:
        sl = _overlap_slices(t.shape, out.shape)
        out[sl] = t[sl].to(device=out.device, dtype=out.dtype)
        return out
    else:
        raise ValueError(f"Ref {t.shape} and neuron {B_neu} batch sizes do not match: ")


# ------------------------------------------------------------------------------------
# Name reconciliation (reference -> neuron canonical)
# ------------------------------------------------------------------------------------
def _apply_ref_equiv(name: str, equiv: Dict[str, str]) -> str:
    """
    Apply a user-provided "reference -> Neuron" substring mapping (first occurrence only).

    WHY needed?
      Capture names from a "reference" (CPU) run may differ from Neuron-run module
      names (e.g., router vs gate node name changes). We map those so that files
      line up by a canonical Neuron module path.

    Suffix handling:
      We preserve meaningful suffixes like "_output" because the mapping is applied
      only to the module path portion.

    Example:
      name='layers.0.mlp.gate_output',
      equiv={'mlp.gate':'mlp.router.linear_router'}
      -> 'layers.0.mlp.router.linear_router_output'
    """
    for src, dst in equiv.items():
        # exact segment or segment followed by suffix
        if (
            name == src
            or name.startswith(src + ".")
            or name.startswith(src + "_")
            or (("." + src + ".") in name)
        ):
            return name.replace(src, dst, 1)
        # fallback: plain first occurrence
        if src in name:
            return name.replace(src, dst, 1)
    return name


class TensorReplacementRegister:
    """
    Singleton registry that prepares per-module, per-step replacement tensors
    and binary masks to drive *tensor replacement* at runtime.

    Current scope/assumptions:
    • This registry is designed around **MoE router logits** replacement.
    • Neuron capture shapes are authoritative (we pad reference to match).
    • Steps come from the filename convention (ctx/cte/tkg phases).
    • `tr_map` declares, per step, which modules should be replaced.
    • For each (step, module) we store:
        - `packed_step[step][module]`: the replacement tensor (padded)
        - `masks_step[step][module]` : a `(1,)` bool tensor; True => replace

    Lifecycle:
    get_instance(ref_dir, neuron_dir, tr_map, ref_equiv_map?)
        -> scans dirs, validates shapes/ranks, aligns/pads reference,
        populates per-step/module tensors and masks
    step_args(step) -> ([tensors...], [masks...]) aligned with module_superset
    """

    # Singleton storage. Use get_instance() to construct; __init__ is disabled.
    _instance = None

    # ------------------------------------------------------------------------------
    # Singleton entry point
    # ------------------------------------------------------------------------------
    @classmethod
    def get_instance(
        cls,
        ref_dir: Optional[str] = None,
        neuron_dir: Optional[str] = None,
        tr_map: Optional[Dict[int, List[str]]] = None,
        config=None,
        ref_equiv_map: Optional[Dict[str, str]] = None,
    ) -> "TensorReplacementRegister":
        """
        First call constructs and configures the singleton.
        Subsequent calls return the already-constructed instance.

        Required on first call:
          ref_dir:     directory containing CPU/reference captures
          neuron_dir:  directory containing Neuron captures (authoritative shapes)
          tr_map:      { step:int -> [module_name:str, ...] } modules to replace
          ref_equiv_map (optional): name mapping to reconcile reference->neuron
        """
        if cls._instance is None:
            obj = super().__new__(cls)            # bypass __init__
            cls._instance = obj
            if any(x is None for x in (ref_dir, neuron_dir, tr_map, config)):
                raise ValueError(
                    "ref_dir, neuron_dir, and tr_map, config are required args when register has not been instantiated"
                )
            obj._configure(ref_dir, neuron_dir, tr_map, config, ref_equiv_map)
        return cls._instance

    # Prevent direct instantiation: enforce singleton usage
    def __init__(self):
        raise RuntimeError("Call TensorReplacementRegister.get_instance() instead")

    @classmethod
    def remove_hooks(cls):
        """
        Best-effort removal of any registered framework hooks.
        Safe to call even if no instance / hooks exist.
        """
        try:
            inst = cls.get_instance()
        except Exception:
            return
        if not hasattr(inst, "hooks") or not inst.hooks:
            return
        for h in inst.hooks:
            try:
                h.remove()
            except Exception:
                pass
        inst.hooks = []

    @classmethod
    def clear(cls):
        """
        Fully tear down the singleton and any side resources.
        (Useful in tests or dynamic reload workflows.)
        """
        if cls._instance is None:
            return
        cls.remove_hooks()
        cls._instance = None

    # ------------------------------------------------------------------------------
    # Internal configuration/build pipeline
    # ------------------------------------------------------------------------------
    def _configure(
        self,
        ref_dir: str,
        neuron_dir: str,
        tr_map: Dict[int, List[str]],
        config,
        ref_equiv_map: Optional[Dict[str, str]] = None,
    ):
        """
        Stash inputs and initialize state-holding containers, then build.

        Args:
          ref_dir: path to CPU/reference captured tensors
          neuron_dir: path to Neuron captured tensors
          tr_map: which modules to replace at each step (authoritative driver)
          ref_equiv_map: optional substring mapping to reconcile ref names
        """
        self.ref_dir = ref_dir
        self.neuron_dir = neuron_dir
        self.tr_map = tr_map
        self.ref_equiv_map = ref_equiv_map or {}

        # Derived / internal state
        # stable, deduped module list across all steps
        self.module_superset: List[str] = []
        # step -> module -> Shape
        self.step_shapes: Dict[int, Dict[str, torch.Size]] = {}
        # step -> module -> dtype (from Reference)
        self.ref_step_dtypes: Dict[int, Dict[str, torch.dtype]] = {}
        # step -> module -> replacement tensor (padded)
        self.packed_step: Dict[int, Dict[str, torch.Tensor]] = {}
        # step -> module -> Bool mask tensor (shape (1,))
        self.masks_step: Dict[int, Dict[str, torch.Tensor]] = {}
        # optional: place to store teardown handles
        self.hooks: List[Any] = []
        self.tensors_neuron = []
        self.tensors_ref = []
        self.config = config
        self._build()

    def _build(self):
        """
        High-level pipeline:
          1) Determine the *module superset* across all steps we want to replace.
          2) Scan Neuron dir (authoritative shapes) and Reference dir (to infer dtypes).
          3) Validate shapes/ranks consistency across steps.
          4) Normalize reference tensors (pad/right-align) to match Neuron shapes.
          5) Allocate `packed_step` (replacement tensors) and `masks_step` (replace flags).
        """
        # ---- 1. Gather all requested modules across all steps in a stable order
        steps = sorted(self.tr_map.keys())  # step ids: 1 for ctx/cte, 2.. for tkg, etc.
        if not steps:
            return

        # Stable, deduped module superset
        modules: List[str] = []
        seen: set = set()
        for s in steps:
            for m in self.tr_map[s]:
                if m not in seen:
                    seen.add(m)
                    modules.append(m)
        self.module_superset = modules

        # ---- 2. Scan directories (returns per-module per-step tensors)
        # NOTE: Neuron tensors carry the final compiled layout/shape; reference captures
        # may be smaller and require padding, so we always shape-match to Neuron.
        self.tensors_neuron = self._scan_dir(self.neuron_dir, source="neuron")
        self.tensors_ref = self._scan_dir(self.ref_dir, source="ref")

        # Capture the set of steps for which neuron tensors exist (authoritative)
        steps = sorted(
            {s for mod, step_map in self.tensors_neuron.items() for s in step_map.keys()}
        )

        # ---- 3. Infer and validate step-scoped shapes (from Neuron captures)
        self.step_shapes = {}
        # all tkg requested steps must share the same shape per module and same rank across ctx/tkg steps
        for s in steps:
            self.step_shapes.setdefault(s, {})
            for m in self.module_superset:
                if m not in self.tensors_neuron or s not in self.tensors_neuron[m]:
                    raise ValueError(f"Missing Neuron shape for module '{m}' at tkg step {s}.")
                shp = torch.Size(self.tensors_neuron[m][s].shape)
                self.step_shapes[s][m] = shp

        # Validate: each module must have consistent shape across all steps, and constant rank.
        for m in self.module_superset:
            module_shapes = [self.step_shapes[s][m] for s in steps]
            if not module_shapes or len(module_shapes) != len(steps):
                if module_shapes:
                    raise ValueError(
                        f"Got {len(module_shapes)} module shapes but expected {len(steps)}"
                    )
                raise ValueError(f"Expected non-zero tkg_shapes but got {module_shapes}")
            # All steps must agree on exact shape (common in TKG phases)
            if len({tuple(sz) for sz in module_shapes[1:]}) > 1:
                raise ValueError(
                    f"Shape mismatch across tkg steps for module '{m}': {[tuple(sz) for sz in module_shapes]}"
                )
            # rank constant across all steps
            ranks = {len(sz) for sz in module_shapes}
            if len(ranks) != 1:
                raise ValueError(
                    f"Rank mismatch across steps for '{m}': {[tuple(sz) for sz in module_shapes]}"
                )

        # ---- 4. Infer dtypes from reference captures
        # (We trust Neuron for shapes, but reference for the dtype we want to feed in.)
        for m in self.module_superset:
            dtypes: List[torch.dtype] = []
            for s in steps:
                self.ref_step_dtypes.setdefault(s, {})
                if m not in self.tensors_ref or s not in self.tensors_ref[m]:
                    raise ValueError(
                        f"Missing CPU tensor to infer dtype for step {s}, module '{m}'"
                    )
                t_ref = self.tensors_ref[m][s]
                dtypes.append(t_ref.dtype)
                self.ref_step_dtypes[s][m] = t_ref.dtype

            if not dtypes:
                raise ValueError(f"No CPU tensors found to infer dtype for module '{m}'")
            if len(set(dtypes)) != 1:
                raise ValueError(
                    f"Mixed dtypes across steps for module '{m}': "
                    f"{sorted(set(map(str, dtypes)))}"
                )

        # ---- 5. Allocate per-step replacement tensors and masks
        # packed_step[s][m] = normalized (padded) reference tensor ready to inject
        # masks_step[s][m]  = Bool flag (shape (1,)): True -> replace at runtime
        self.packed_step = {}
        self.masks_step = {}

        for s in steps:
            self.packed_step.setdefault(s, {})
            self.masks_step.setdefault(s, {})
            for m in self.module_superset:
                neu_t = self.tensors_neuron[m][s]          # final shape from Neuron
                dt = self.ref_step_dtypes[s][m]       # desired dtype from reference
                ref_t = self.tensors_ref[m][s]
                if self.config.strided_context_parallel_kernel_enabled and s == 1:
                    neu_t_unstrided = order_strided_tensor(neu_t, dim=1, stride=self.config.cp_degree)
                    ref_t = _ensure_rank_and_pad_ref_to_target(ref_t, neu_t_unstrided)
                    ref_t = stride_tensor(ref_t, dim=1, stride=self.config.cp_degree)
                else:
                    ref_t = _ensure_rank_and_pad_ref_to_target(ref_t, neu_t)
                # Defensive: dtype should agree with the inferred dtype map
                if dt != ref_t.dtype:
                    raise ValueError("dtype mismatch between cpu tensors")

                # Store ready-to-use replacement tensor
                self.packed_step[s][m] = ref_t

                # Construct the mask: shape (1,), True iff the (s,m) pair is requested in tr_map
                replace_flag = (
                    s in self.tr_map and m in self.tr_map[s]
                )
                self.masks_step[s][m] = torch.ones(
                    (1,), dtype=torch.bool
                ) if replace_flag else torch.zeros((1,), dtype=torch.bool)

    # --------------------------------------------------------------------------------
    # Public APIs to retrieve argument lists for a given step
    # --------------------------------------------------------------------------------
    def example_args(self, step: int):
        """
        Build *synthetic* zero-value args for a given step, with the correct shapes/dtypes typically used during tracing.

        Returns:
          (tr_list, mask_list)
            tr_list:  list of zero tensors with shapes == step_shapes[step][module]
            mask_list: list of zeros Bool tensors with shape (1,)
        """
        tr_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        for m in self.module_superset:
            if not hasattr(self, "step_shapes") or m not in self.step_shapes[step]:
                raise ValueError(f"shape missing for {step} module '{m}'")
            if not hasattr(self, "ref_step_dtypes") or m not in self.ref_step_dtypes[step]:
                raise ValueError(f"dtype missing for {step} module '{m}'")

            shp = self.step_shapes[step][m]
            dt = self.ref_step_dtypes[step][m]

            ''' Guard against difference in batch size defined in config vs tensors captured on neuron run
                Example: ctx_batch_size=1, batch_size=16, tkg_batch_size=16
                when run woth 16 prompts as done in logit validation tests
                router logits in ctx encoding phase will have shp = [16, S, H] although compiler expects [1, S, H]
            '''
            if step == 1:
                bs = self.config.ctx_batch_size or self.config.batch_size
            else:
                bs = self.config.tkg_batch_size or self.config.batch_size

            if shp[0] != bs:
                shp = torch.Size((bs, *shp[1:]))

            # Zero TF arg with static shape == compiled shape
            tr_list.append(torch.zeros(shp, dtype=dt))
            # Mask is a simple (1,) Bool flag for this module at this step
            mask_list.append(torch.zeros((1,), dtype=torch.bool))

        return (tr_list, mask_list)

    def step_args(self, step: int, divergence_idx=False):
        tr_list, mask_list = [], []

        if not divergence_idx:
            # Original behavior: directly return packed tensors for this token-generation step
            for m in self.module_superset:
                tr_list.append(self.packed_step[step][m])
                mask_list.append(self.masks_step[step][m])
            return tr_list, mask_list

        if 1 not in self.step_shapes:
            raise ValueError("Missing ctx (step=1) shapes; cannot infer Neuron bucket.")

        for m in self.module_superset:
            neuron_ctx_target = self.tensors_neuron[m][1]
            # Gather CPU segments from ctx..step (inclusive) for this module
            ref_step_map = self.tensors_ref.get(m, {})
            ref_steps = [s for s in sorted(ref_step_map.keys()) if 1 <= s <= step]
            if not ref_steps:
                raise ValueError(f"No CPU captures for module '{m}' in steps [1..{step}]")

            batch = neuron_ctx_target.shape[0]
            hidden = neuron_ctx_target.shape[-1]
            segments = []
            for s_idx, s_id in enumerate(ref_steps):
                seg = ref_step_map[s_id]

                if seg.dim() == 2:
                    if seg.shape[0] % batch != 0:
                        raise ValueError(
                            f"Cannot reshape {tuple(seg.shape)} into (B={batch}, S, H={hidden})"
                        )
                    seg = seg.reshape(batch, -1, hidden)
                else:
                    raise ValueError(f"Unsupported rank {seg.dim()} for module '{m}'")

                segments.append(seg)

            # Determine sequence dimension for concatenation
            # For 3D tensors (B, S, H), concatenate along sequence dimension (dim=1)
            # For 2D tensors (S, H), concatenate along sequence dimension (dim=0)
            seq_dim = 1 if segments[0].dim() >= 3 else 0
            try:
                combined_3d = torch.cat(segments, dim=seq_dim)
                combined = combined_3d.reshape(batch * combined_3d.shape[1], hidden)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Concatenation failed for module '{m}' along dim {seq_dim}. "
                    f"Check that non-seq dims match across steps."
                ) from e
            combined = _ensure_rank_and_pad_ref_to_target(combined, neuron_ctx_target)
            tr_list.append(combined)
            mask_list.append(self.masks_step[step][m])
        return tr_list, mask_list

    # --------------------------------------------------------------------------------
    # Directory scan helpers
    # --------------------------------------------------------------------------------
    def _scan_dir(
        self,
        root: Optional[str],
        source: str,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Scan a capture directory and collect tensors keyed by (module, step).

        Returns:
          out: { module_name -> { step:int -> tensor } }

        Notes:
          • For 'ref' source we apply `ref_equiv_map` to reconcile module names to
            the canonical Neuron names used at runtime.
          • Only modules in `module_superset` are retained (others are skipped).
          • If the same (module, step) appears twice with different shapes, we raise.
            (Multiple captures for the same logical point is ambiguous.)
        """
        out: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
        if not root or not os.path.isdir(root):
            return out

        for fn in os.listdir(root):
            m = _FN.match(fn)  # captured_tensors_<phase>_step_<N>_module_<name>.pt
            if not m:
                continue

            step = int(m.group(1))
            raw = m.group(2)

            # Reconcile module names for reference captures if necessary
            mod = raw
            if source == "ref" and self.ref_equiv_map:
                mod = _apply_ref_equiv(mod, self.ref_equiv_map)

            # Skip files for modules we don't plan to replace
            if mod not in self.module_superset:
                continue

            path = os.path.join(root, fn)
            try:
                t = torch.load(path, map_location="cpu")
            except Exception as exc:
                raise ValueError(f"Failed to read tensor from {path}") from exc

            prev = out[mod].get(step)
            if prev is not None:
                # Multiple tensors for the same (module, step) would be ambiguous
                raise ValueError(
                    f"Tensor for module {mod} and step {step} has already been recorded. "
                    f"This condition should not have been encountered"
                )

            out[mod][step] = t

        return out
