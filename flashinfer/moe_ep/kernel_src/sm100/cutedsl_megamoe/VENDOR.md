# Vendoring record: cutedsl_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only; the drop-update *workflow* (what to
replace, what to audit) lives in `SKILL.md`.

## Upstream

The explicit Blackwell GenPhase prototype lives in the independent
[`../next_cutedsl_megamoe`](../next_cutedsl_megamoe/VENDOR.md) package. This
legacy package lazily re-exports its public API; it does not import Rubin or
replace its own device snapshot.

- **Vendored commit**: `fa662f345980f75776ae0c85d2e5c188918ded26`
- **Last synced**: 2026-09-29
- **Vendored subset**: the six kernel packages only (`common/`, `src/`,
  `moe_bf16_glu/`, `moe_mxfp8_glu/`, `moe_mxfp8_bf16_glu/`,
  `moe_nvfp4_swapab/`) under `src/` — no repo scaffolding (`ci/`, `tester/`,
  `tests/`, `scripts/`, `pyproject.toml`, …).

## Policy

- `src/` is a **verbatim** copy of the upstream drop: no injected files, no
  local edits. `diff -r src/<pkg> <upstream>/<pkg>` must come back clean.
- All adaptation lives in `shim/` (ours), re-exported through `__init__.py`;
  FlashInfer backends import the package `__init__` only, never `src/`.
- Local bug fixes go upstream first, then re-sync. If an emergency local edit
  is unavoidable, list it here as a pending-upstream diff until the next drop
  absorbs it.

## Pending local diffs vs upstream

- **Sparse decode (FlashInfer PR #6447 / #6565, 2026-10-14):** re-ported onto
  this drop in four device files. `src/src/token_comm.py` gives CTA0 ownership
  of small shared reset prefixes (EP1, prefixes above 16 KiB and NCU keep the
  distributed reset), preserves the grid counter phase when omitting a join,
  and, for capacities at most 1024, uses the last valid route extent to reduce
  the dispatch CTA count (`active_dispatch_warps` / `active_dispatch_threads`)
  while preserving physical metadata strides and full-grid high-word
  completion counts. `src/moe_nvfp4_swapab/topk_reduce.py` adds optional
  BF16 route masking (static and persistent paths); the NVFP4 and MXFP8
  `megamoe_kernel*.py` launchers pass the indices and expert count for BF16
  combine only and allocate the local extent/ready counter. Invalid-route
  payloads and scores are ignored. Reapply until upstream absorbs them; they
  do not change quantized/in-kernel reduction.
- **Persistent top-k reduction:** upstream MR !63 commit
  `4d51560753607a5b33ada5eb2ae913b0023dae3b` carries the one-cursor
  fixed-grid scheduler. It is integrated behind `topk_reduce_persistent` in
  `src/moe_nvfp4_swapab/megamoe_kernel.py` and
  `src/moe_mxfp8_glu/megamoe_kernel_mxfp8.py`, and
  `src/moe_mxfp8_bf16_glu/megamoe_kernel_mxfp8_bf16.py`; all integrations
  honor the vendored `skip_topk_reduce` path.
- **Activation controls:** `src/moe_nvfp4_swapab/{epilogue_refactor,kernel_fc12,megamoe_kernel}.py`
  retain FlashInfer's custom SwiGLU alpha/beta parameters and allow
  `situ_linear_beta=None`; upstream implements standard SwiGLU and requires
  both positive SiTU beta values.
- **Vendored-subset imports:** `src/moe_mxfp8_glu/runner_col_requant.py`
  uses package-local helpers and lazily imports its checker so importing the
  runner does not require omitted test scaffolding.

## Related trees

- `kernel_src/sm90/pull_style_cutedsl_megakernel/` is a **separate snapshot**
  of a fork of this repo (older common code, Hopper FP8 pull kernel). It is
  not merged into this directory on purpose: one kernel_src dir = one upstream
  commit. If upstream merges the SM90 kernel into the mother repo, fold that
  tree into this one on the next re-sync.

## Consumers

- `backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm100/mxfp8_mxfp8_bf16_cutedsl/`
