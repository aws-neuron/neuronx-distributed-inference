import os
import re
import torch
import torch.nn.functional as F
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
def _pad_to_shape_right(t: torch.Tensor, target: torch.Size) -> torch.Tensor:
    """
    Right-pad each dimension to reach `target` shape (NEVER truncates).
    - If any target dimension is smaller than source -> raises (we don't shrink).
    - Pads the *end* of each dimension, preserving "prefix" semantics.

    Args:
      t:       input tensor
      target:  desired shape

    Returns:
      Tensor with shape == target (same dtype/device as input).
    """
    if t.dim() != len(target):
        raise ValueError(f"rank mismatch: {tuple(t.shape)} vs {tuple(target)}")

    pads: List[int] = []
    # F.pad expects pairs reversed by dimension order:
    # (dim_n_right, dim_n_left, dim_{n-1}_right, ...)
    # We only add "right" pads, so each pair is (0, need) and we push them in reverse dim order.
    for d in reversed(range(t.dim())):
        need = target[d] - t.size(d)
        if need < 0:
            raise ValueError(
                f"cannot shrink tensor from {tuple(t.shape)} to {tuple(target)}"
            )
        pads.extend([0, need])

    return F.pad(t, pads) if any(pads) else t


def _ensure_rank_and_pad_ref_to_target(
    t: torch.Tensor,
    target: torch.Size,
    assume_batch_dim0: bool = True,
) -> torch.Tensor:
    """
    Normalize a *reference* (CPU/golden) capture to match a Neuron tensor shape.

    Steps:
      1) If the reference is missing a leading batch dimension (common in offline
         captures) and we allow it, unsqueeze at dim 0. (e.g., (S,H)->(1,S,H))
      2) Enforce same rank as target (we never guess missing middle dims).
      3) Right-pad each dimension to match the target exactly (never truncate).

    Rationale:
      Neuron shapes are authoritative. Reference captures may be smaller (e.g.,
      shorter sequence, missing batch dim). We pad *only to the right* to align.

    Args:
      t: reference tensor to normalize
      target: final shape to match (from Neuron capture)
      assume_batch_dim0: whether to auto-insert a missing leading batch dim

    Returns:
      Tensor with shape == target.
    """
    # Add missing batch dim (allowed only when target batch is 1)
    if t.dim() == len(target) - 1 and assume_batch_dim0:
        t = t.unsqueeze(0)  # e.g., (S,H) -> (1,S,H)

    # Rank must match now; we never attempt to infer middle dims
    if t.dim() != len(target):
        raise ValueError(
            f"After batch fix-up, rank mismatch: src={tuple(t.shape)} vs tgt={tuple(target)}"
        )

    return _pad_to_shape_right(t, target)


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
            obj = super().__new__(cls)  # bypass __init__
            cls._instance = obj
            if any(x is None for x in (ref_dir, neuron_dir, tr_map)):
                raise ValueError(
                    "ref_dir, neuron_dir, and tr_map are required args when register has not been instantiated"
                )
            # Populate internal state (dirs, maps, superset, shapes, tensors, masks)
            obj._configure(ref_dir, neuron_dir, tr_map, ref_equiv_map)
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
        tensors_neuron = self._scan_dir(self.neuron_dir, source="neuron")
        tensors_ref = self._scan_dir(self.ref_dir, source="ref")

        # Capture the set of steps for which neuron tensors exist (authoritative)
        steps = sorted(
            {s for mod, step_map in tensors_neuron.items() for s in step_map.keys()}
        )

        # ---- 3. Infer and validate step-scoped shapes (from Neuron captures)
        self.step_shapes = {}
        # all tkg requested steps must share the same shape per module and same rank across ctx/tkg steps
        for s in steps:
            self.step_shapes.setdefault(s, {})
            for m in self.module_superset:
                if m not in tensors_neuron or s not in tensors_neuron[m]:
                    raise ValueError(f"Missing Neuron shape for module '{m}' at tkg step {s}.")
                shp = torch.Size(tensors_neuron[m][s].shape)
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
                if m not in tensors_ref or s not in tensors_ref[m]:
                    raise ValueError(
                        f"Missing CPU tensor to infer dtype for step {s}, module '{m}'"
                    )
                t_ref = tensors_ref[m][s]
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
                shp = self.step_shapes[s][m]          # final shape from Neuron
                dt = self.ref_step_dtypes[s][m]       # desired dtype from reference
                ref_t = tensors_ref[m][s]

                # Align ref capture to neuron shape (right-pad, allow missing leading batch)
                ref_t = _ensure_rank_and_pad_ref_to_target(ref_t, shp)

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

            # Zero TF arg with static shape == compiled shape
            tr_list.append(torch.zeros(shp, dtype=dt))
            # Mask is a simple (1,) Bool flag for this module at this step
            mask_list.append(torch.zeros((1,), dtype=torch.bool))

        return (tr_list, mask_list)

    def step_args(self, step: int):
        """
        Retrieve the *real* per-step replacement tensors and masks aligned with module_superset.

        Returns:
          (tr_list, mask_list)
            tr_list:  [ packed_step[step][m] for m in module_superset ]
            mask_list:[ masks_step[step][m]  for m in module_superset ]
        """
        tr_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []

        for m in self.module_superset:
            tr_list.append(self.packed_step[step][m])
            mask_list.append(self.masks_step[step][m])

        return (tr_list, mask_list)

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
