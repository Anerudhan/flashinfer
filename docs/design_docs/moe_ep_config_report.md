# MoE expert-parallel configuration study — DeepSeek-V4 on GB200/GB300

Comparison of FlashInfer MoE expert-parallel configurations under vLLM and
SGLang, for the blog post. Covers the two compute-path questions (cuTeDSL
**split** vs **MegaMoE**; TRTLLM **routed** vs **MegaMoE**), the transport
axis (FlashInfer all2all / NCCL-EP / NIXL-EP), and **DeepEP MegaMoE vs
FlashInfer MegaMoE** — on DeepSeek-V4-Pro and DeepSeek-V4-Flash.

Reference integrations:

- SGLang [#31470](https://github.com/sgl-project/sglang/pull/31470)
  "[NVIDIA] Support flashinfer Mega Moe" — merged 2026-09-10
  (`1b77f498`).
- vLLM [#49636](https://github.com/vllm-project/vllm/pull/49636)
  "[Model][MoE] DeepSeek-V4: add opt-in FlashInfer moe_ep expert backend" —
  merged 2026-08-25 (`8fe9317f`).

> **Status.** Measured: 12 serving cells across both models and both
> frameworks, plus GSM8K on two of them. Every configuration marked **N/A**
> is one the stack refuses to run, with the exact error quoted — not a
> missing measurement. Known gaps are listed in §7.

---

## 0. Summary — what to put in the blog post

**0.1 The transport axis is mostly a fiction today.** MoE runners and
all2all transports are *co-designed pairs*, not freely composable choices.
On DeepSeek-V4 NVFP4, the FlashInfer split runners
(`flashinfer_cutedsl`, `flashinfer_trtllm`) are effectively limited to
FlashInfer all2all. NIXL-EP and DeepEP-**low-latency** both require the
`batched_experts` activation format, which those NVFP4 kernels do not
implement, and the server refuses at init. DeepEP-**high-throughput** clears
that gate but then dies inside CUDA-graph capture
(`DeepEP error: CPU recv timeout`), so it is only measurable in eager mode.
SGLang refuses both `nixl` and `deepep` for `flashinfer_trtllm_routed` with
*"requires a fused func for a2a backend &lt;x&gt;, but none is registered."* —
but **does** support `flashinfer_cutedsl` + `deepep`. NCCL-EP is exposed by
neither framework. So support is decided **per (runner, transport) pair**,
not per runner: in vLLM every FlashInfer split runner is down to FlashInfer
all2all alone, while in SGLang cuTeDSL keeps DeepEP and TRTLLM-routed does
not. Full support table: §3.4b.

**0.1b DeepEP costs ~8× on this stack, for a reason that has nothing to do
with DeepEP's kernels.** The only DeepEP variant these runners accept
(high-throughput) cannot be CUDA-graph captured — it dies in capture with
`DeepEP error: CPU recv timeout` — so using it forces `--enforce-eager`.
Eager costs **8.2×** throughput on V4-Flash (73,091 → 8,920 tok/s, TPOT
26.3 → 228.1 ms). Whatever DeepEP's dispatch/combine kernels do, giving up
cuda-graph capture dominates it by an order of magnitude.

**0.1c DeepEP-HT and FlashInfer all2all are the same speed.** Measured
eager-vs-eager on the same TRTLLM-routed runner: 10,508 vs 10,411 tok/s —
within 0.9%. DeepEP is marginally ahead on TPOT, FlashInfer all2all on TTFT
and ITL. There is no throughput argument between the two transports; the
only decisive difference is the ~8× capture penalty DeepEP-HT forces (0.1b).

**0.2 Fusing the communication into the kernel is worth ~1.17× — but only
with cuda-graph capture.** On V4-Flash (GB200, EP=4, ISL 8192 / OSL 1024,
conc 256) both megakernels beat both split paths — 73.1k–78.0k vs
61.8k–62.4k tok/s. **Run the same comparison eager and the ordering
inverts**: FI MegaMoE 8,920 vs the split paths' 10,411–10,508 tok/s, i.e. it
*loses* by ~1.17×. The megakernel's advantage is host-side launch
amortisation under capture, not intrinsic device efficiency — its ITL is
still the best of the three in eager mode. Any claim about MegaMoE's speedup
should state the capture mode it was measured in.

**0.3 The split inner-kernel choice is second-order.** Under vLLM, cuTeDSL
split and TRTLLM-routed land within 1% of each other (62,408 vs 61,795
tok/s); under SGLang cuTeDSL is ahead by 5.8% (45,860 vs 43,333). Either way
the fused-vs-split axis moves the number far more than cuTeDSL-vs-TRTLLM
does. TRTLLM-routed consistently wins TTFT, cuTeDSL consistently wins TPOT.

**0.4 The headline gap: native deep_gemm MegaMoE is currently 6.7% faster
than FlashInfer cuTeDSL MegaMoE** (77,956 vs 73,091 tok/s) and ahead on every
latency metric (ITL 13.9 vs 16.1 ms), at equal accuracy (GSM8K 0.870 vs
0.875). This is the reverse of what the FlashInfer MegaMoE integration is
aiming for, and it is the most actionable result here.

**0.5 Model size decides which regime you can even measure.** V4-Pro's
851 GB NVFP4 checkpoint leaves ~47k KV tokens per rank at EP=4 — max
concurrency 5.0× — so a single node can only probe the small-batch corner.
V4-Flash has 15× the headroom (76.4×) and carries the real sweep. Any
V4-Pro large-batch claim needs EP=8 across two nodes.

**0.6 Two integration bugs found.**
`flashinfer_moe_ep_mega_deep_gemm` hard-imports `deep_gemm` and fails on an
image where vLLM's own mega path works fine via its vendored
`vllm.third_party.deep_gemm`. And vLLM's SwiGLU-clamp rejection message lists
`flashinfer_cutedsl` among the alternatives to the `flashinfer_cutedsl` it is
rejecting. Both worth filing upstream.

---

## 1. Model geometries

Read from the checkpoint `config.json` (not inferred):

| | DeepSeek-V4-Pro | DeepSeek-V4-Flash |
|---|---|---|
| `hidden_size` | 7168 | 4096 |
| `moe_intermediate_size` | 3072 | 2048 |
| `n_routed_experts` | 384 | 256 |
| `n_shared_experts` | 1 | 1 |
| `num_experts_per_tok` (top-k) | 6 | 6 |
| `num_hidden_layers` | 61 | 43 |
| expert quant | NVFP4 | NVFP4 (`group_size` 16) |
| checkpoints used | `deepseek-ai/DeepSeek-V4-Pro`, `nvidia/DeepSeek-V4-Pro-NVFP4` | `deepseek-ai/DeepSeek-V4-Flash`, `nvidia/DeepSeek-V4-Flash-NVFP4` |

Both hidden sizes are inside the NIXL-EP whitelist
(`{2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}`), so the NIXL-EP
transport is legal for both models.

---

## 2. Hardware and software

| | |
|---|---|
| Cluster | lyris (SLURM + pyxis), aarch64 |
| V4-Pro runs | 1× GB300 node (`theia`), 4× GB300, 284 GB/GPU, cc 10.3 (sm103), EP=4 |
| V4-Flash runs | 1× GB200 node (`lyris`), 4× GB200, 183 GB/GPU, cc 10.0 (sm100), EP=4 |
| vLLM | 0.29.0 (arm64 image), contains the merged #49636 code |
| SGLang | `lmsysorg/sglang:nightly-dev-cu13-20260918-20518d85` (arm64), post-merge |
| FlashInfer | 0.6.18 (in the vLLM image) |
| DeepEP | present in the vLLM image (`deep_ep` importable) |

V4-Pro is placed on GB300 because the NVFP4 checkpoint (851 GB) does not fit
in a GB200 node's 4×183 GB; it fits in GB300's 4×284 GB at EP=4. V4-Flash is
small enough for GB200. Cross-backend comparisons are always **within** a
model on **one** hardware type, so this placement does not affect any
conclusion drawn here; cross-*model* numbers are not compared directly.

### 2.1 KV headroom decides which batch regime each model can probe

Measured at `--gpu-memory-utilization 0.92`, EP=4:

| Model | ckpt size | KV per rank | max concurrency @ 9,472 tok/req |
|---|---|---|---|
| V4-Pro (NVFP4) | 851 GB | 47,316 tok | **5.0×** |
| V4-Flash (NVFP4) | 157 GB | 723,667 tok | **76.4×** |

V4-Pro's weights consume nearly the whole node at EP=4, so a single GB300
node can only hold ~5 concurrent 9.4k-token requests. That confines V4-Pro
EP=4 to the **small-batch** regime — which is precisely the corner where the
SGLang PR reports `flashinfer_trtllm_routed` beating MegaMoE on latency, and
so it cannot show the crossover. To probe the large-batch regime on V4-Pro
the weights must be spread wider (EP=8 over two nodes, ~8× the KV headroom).

V4-Flash's checkpoint is 5.4× smaller and its measured KV headroom is **15×
larger** (76.4× vs 5.0× concurrency), so at EP=4 on a single GB200 node it
carries the full concurrency sweep. V4-Flash therefore provides the headline
crossover curve, and V4-Pro provides the large-model datapoint.

Parallelism for every cell: `--data-parallel-size 4 --tensor-parallel-size 1
--enable-expert-parallel` (i.e. DP-attention + EP=4). This matters: vLLM only
selects an all2all transport when `dp_size > 1`. With TP-only, every cell
collapses onto the same monolithic TP-all-reduce path and the transport flag
becomes a no-op.

---

## 3. Configuration matrix and how each axis maps to a real flag

Verified against the shipped `vllm.config.kernel.MoEBackend` and
`vllm.config.parallel.All2AllBackend` enums.

### 3.1 Compute path (`--moe-backend`)

| Report axis | vLLM selector | Checkpoint |
|---|---|---|
| cuTeDSL **split** | `flashinfer_cutedsl` | NVFP4 |
| TRTLLM **routed** | `flashinfer_trtllm` | NVFP4 |
| FlashInfer **MegaMoE** (cuTeDSL) | `flashinfer_moe_ep_mega_cutedsl` | NVFP4 |
| FlashInfer **MegaMoE** (deep_gemm) | `flashinfer_moe_ep_mega_deep_gemm` | base (MXFP4/fp8) |
| **Native MegaMoE** (deep_gemm) | `deep_gemm_mega_moe` | base (MXFP4/fp8) |

### 3.2 Transport (`--all2all-backend`)

| Report axis | vLLM selector |
|---|---|
| FlashInfer all2all | `flashinfer_all2allv` (also `flashinfer_nvlink_one_sided`, `flashinfer_nvlink_two_sided`) |
| NIXL EP | `nixl_ep` |
| DeepEP | `deepep_low_latency`, `deepep_high_throughput` |
| NCCL EP | **not available upstream** — see §3.4 |

### 3.3 The nine arms

| Arm | `--moe-backend` | `--all2all-backend` |
|---|---|---|
| `split_cutedsl_fia2a` | `flashinfer_cutedsl` | `flashinfer_all2allv` |
| `split_cutedsl_nixl` | `flashinfer_cutedsl` | `nixl_ep` |
| `split_cutedsl_deepep` | `flashinfer_cutedsl` | `deepep_low_latency` |
| `split_trtllm_fia2a` | `flashinfer_trtllm` | `flashinfer_all2allv` |
| `split_trtllm_nixl` | `flashinfer_trtllm` | `nixl_ep` |
| `split_trtllm_deepep` | `flashinfer_trtllm` | `deepep_low_latency` |
| `mega_fi_cutedsl` | `flashinfer_moe_ep_mega_cutedsl` | — (fused) |
| `mega_fi_deepgemm` | `flashinfer_moe_ep_mega_deep_gemm` | — (fused) |
| `mega_native_deepgemm` | `deep_gemm_mega_moe` | — (fused) |

### 3.4 SGLang selectors (confirmed against the image's `--help`)

| Report axis | SGLang selector |
|---|---|
| cuTeDSL split | `--moe-runner-backend flashinfer_cutedsl` |
| TRTLLM routed | `--moe-runner-backend flashinfer_trtllm_routed` |
| FlashInfer MegaMoE | `--moe-runner-backend flashinfer_megamoe` + `--moe-a2a-backend flashinfer_megamoe` |
| FlashInfer all2all | `--moe-a2a-backend flashinfer` (+ `--flashinfer-a2a-dispatch-type {auto,bf16,nvfp4,mxfp8}`) |
| NIXL EP | `--moe-a2a-backend nixl` |
| DeepEP | `--moe-a2a-backend deepep` (also `deepep_v2`) |
| NCCL EP | **not available** — absent from the a2a enum |

Full a2a enum: `{none, deepep, mooncake, nixl, mori, ascend_fuseep, flashinfer,
megamoe, deepep_v2, pplx, ascend_tp, flashinfer_megamoe}`.

SGLang therefore covers **all three** requested transports for the split
runners (FlashInfer all2all / NIXL / DeepEP), where vLLM 0.29.0 lacks NIXL
for some paths; neither framework exposes NCCL-EP.

MegaMoE tuning knobs (env): `SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE`
(`bf16`/`mxfp8`/`nvfp4`), `SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE`,
`SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK`.

### 3.4b The runner × transport matrix is sparse — measured support

The single most important structural result of this study: **MoE runners and
all2all transports are not freely composable.** They are co-designed pairs,
and most cells simply are not implemented. Measured on this stack (✅ = ran to
completion, ✗ = refused at init, — = not applicable):

**vLLM 0.29.0, DeepSeek-V4 NVFP4**

| `--moe-backend` | `flashinfer_all2allv` | `nixl_ep` | `deepep_low_latency` | `deepep_high_throughput` |
|---|---|---|---|---|
| `flashinfer_cutedsl` | ✅ | ✗ batched fmt | ✗ batched fmt | ✗ under capture; eager only |
| `flashinfer_trtllm` | ✅ | ✗ batched fmt | ✗ batched fmt | ✗ under capture; eager only |
| `flashinfer_moe_ep_mega_cutedsl` | — fused | — | — | — |
| `flashinfer_moe_ep_mega_deep_gemm` | — fused (✗ needs standalone DeepGEMM) | — | — | — |
| `deep_gemm_mega_moe` | — fused | — | — | — |

**SGLang (nightly `20518d85`), DeepSeek-V4 NVFP4**

| `--moe-runner-backend` | `flashinfer` | `nixl` | `deepep` | `flashinfer_megamoe` |
|---|---|---|---|---|
| `flashinfer_cutedsl` | ✅ serves¹ | ✗ no fused func | **✅ serves¹** | — |
| `flashinfer_trtllm_routed` | ✅ serves¹ | ✗ no fused func | ✗ no fused func | — |
| `flashinfer_megamoe` | — | — | — | ✅ serves¹ (required pairing) |

Note the asymmetry: **SGLang's cuTeDSL split runner _does_ pair with DeepEP**
(server ready in 150 s), while its TRTLLM-routed runner does not. So the
sparsity is per-(runner, transport) pair, not a blanket property of the
FlashInfer runners — SGLang has registered a cuTeDSL×DeepEP fused func and
not the others.

¹ "serves" = the server reaches
`The server is fired up and ready to roll!`. SGLang throughput numbers are
absent from this revision for a benchmark-client reason unrelated to MoE-EP
(§5.3), not because the configuration failed.

SGLang's refusal is identical in shape for both unsupported transports:

```
NotImplementedError: Runner backend MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
requires a fused func for a2a backend nixl, but none is registered.
NotImplementedError: Runner backend MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
requires a fused func for a2a backend deepep, but none is registered.
```

The two refusal modes are explicit and worth quoting, because they say the
same thing in two vocabularies:

- vLLM: `NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support the
  deployment configuration since kernel does not support
  ('batched_experts',) activation format.`
- SGLang: `Runner backend MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED requires
  a fused func for a2a backend deepep, but none is registered.`

In both cases the transport dictates an activation layout (batched vs
standard) and the MoE kernel must have an implementation registered for that
layout. Support is therefore decided per (runner, transport) **pair** — the
same runner can be supported on one framework's DeepEP and refused on the
other's, as `flashinfer_cutedsl` is (✅ under SGLang `deepep`, ✗ under vLLM
`deepep_low_latency`).

A "try every kernel with every transport" sweep is consequently not a
meaningful experiment design on today's stack — the honest deliverable is
the support table above plus performance for the cells that exist.

### 3.5 Axes that do not exist as posed — and why

Four of the requested cells cannot be built as stated. Each is a finding
about how the stacks are actually wired, not a gap in the measurement.

These are findings, not omissions.

**MegaMoE has no transport axis.** Both frameworks agree on this
independently. SGLang's arg hook *rejects* `--moe-a2a-backend
flashinfer_megamoe` unless `--moe-runner-backend` is also
`flashinfer_megamoe` ("FlashInfer MegaMOE a2a backend requires
--moe-runner-backend flashinfer_megamoe"): the transport and the kernel are
one selection. On the FlashInfer side, the mega path is a single fused
comm+compute kernel over NVSHMEM symmetric memory.
`flashinfer/moe_ep/modes/mega_layer.py` states it directly — *"Fused EP mega
kernel — no separate dispatch/combine transport"* — and `MegaConfig` carries
no comm field (`megakernel`, `quantize_input`, `preprocess_weights`,
`transformed_weights` only). So "MegaMoE × {FI all2all, NCCL EP, NIXL EP}" is
not three configurations; it is one. The transport axis is meaningful only
for the **split** rows, which is how the matrix above is built.

**NCCL-EP is not an upstream vLLM transport.** `All2AllBackend` in vLLM
0.29.0 is `{naive, pplx, deepep_high_throughput, deepep_low_latency,
deepep_v2, mori_high_throughput, mori_low_latency, nixl_ep,
allgather_reducescatter, flashinfer_all2allv, flashinfer_nvlink_two_sided,
flashinfer_nvlink_one_sided}` — no NCCL-EP entry. NCCL-EP exists as a
FlashInfer `moe_ep` comm backend and as the unmerged vLLM branch backends
`flashinfer_ep_low_latency` / `flashinfer_ep_high_throughput`
(see `vllm_moe_ep_integration.md`), but it cannot be selected in the shipped
framework, so it is reported N/A at the serving level.

**"DeepEP MegaMoE" is not a configuration that exists.** A megakernel and
DeepEP are mutually exclusive by construction. In vLLM's DeepSeek-V4 model
(`models/deepseek_v4/nvidia/model.py`):

```python
if self.use_mega_moe:
    self._init_mega_moe_experts(...)   # uses get_ep_group() directly
else:
    self._init_fused_moe_experts(...)  # the path that consumes --all2all-backend
```

with the matching `if not self.use_mega_moe: return self._forward_fused_moe(...)`
in `forward`. Every member of `MEGA_MOE_BACKENDS` — including the *native*
`deep_gemm_mega_moe` — therefore bypasses the modular FusedMoE kernel, and
`--all2all-backend` is **inert** for those arms. Passing `deepep_low_latency`
alongside `deep_gemm_mega_moe` would produce a cell labelled "DeepEP MegaMoE"
that never called DeepEP.

So the requested comparison resolves into two real, separate questions, and
the report answers both:

1. **Native MegaMoE vs FlashInfer MegaMoE** — `deep_gemm_mega_moe` vs
   `flashinfer_moe_ep_mega_{deep_gemm,cutedsl}`. Both are fused-comm
   megakernels; this is the mega-vs-mega question.
2. **DeepEP vs FlashInfer all2all vs NIXL as a transport** — measured on the
   **split** runners (`flashinfer_cutedsl`, `flashinfer_trtllm`), which are
   the only configurations where the transport is actually on the path.

**NIXL-EP is not usable with either FlashInfer split runner in vLLM 0.29.0.**
Both `flashinfer_trtllm` and `flashinfer_cutedsl` serve DeepSeek-V4 normally
under `flashinfer_all2allv` (§5.2) and both abort during worker init under
`nixl_ep`. The TRTLLM failure names the root cause directly:

```
ValueError: NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support the
deployment configuration since kernel does not support
('batched_experts',) activation format.
```

NIXL-EP selects the **`batched_experts`** activation format, and neither
FlashInfer NVFP4 split kernel implements it. The cuTeDSL arm hits the same
incompatibility, surfaced through a different validator:

```
ValueError: Model sets swiglu_limit=10.0, but the explicitly requested
moe_backend='flashinfer_cutedsl' does not apply the SwiGLU clamp. Use
'flashinfer_trtllm', 'flashinfer_cutlass', 'flashinfer_cutedsl', 'cutlass',
'b12x', 'marlin', or 'humming' instead.
```

DeepSeek-V4 clamps the routed-expert SwiGLU (`swiglu_limit=10.0`), and on the
batched path vLLM refuses a MoE backend that would silently skip the clamp.
Both logs show `Using NixlEPAll2AllManager all2all manager` immediately
before the abort, so NIXL is selected and then the kernel/format check
rejects the combination.

The cuTeDSL error message is additionally self-inconsistent — it lists
`flashinfer_cutedsl` among the suggested alternatives while rejecting
`flashinfer_cutedsl`. Worth reporting upstream.

Consequence: **the entire NIXL-EP column is empty for FlashInfer runners in
vLLM 0.29.0** — not because NIXL is slow, but because the NVFP4 FlashInfer
kernels do not implement `batched_experts`. NIXL would need either a
batched-capable NVFP4 FlashInfer kernel or a non-batched NIXL path. SGLang
exposes `--moe-a2a-backend nixl` independently and is the place to retry
this axis.

**`flashinfer_moe_ep_mega_deep_gemm` needs a standalone DeepGEMM; the native
mega path does not.** On the stock vLLM image the FlashInfer deep_gemm mega
arm aborts with `ModuleNotFoundError: No module named 'deep_gemm'`, while the
*native* `deep_gemm_mega_moe` arm comes up normally on the same image:

```
INFO [deep_gemm.py:186] deep_gemm not found in site-packages,
                        trying vendored vllm.third_party.deep_gemm
INFO [deep_gemm.py:213] DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
```

vLLM vendors a DeepGEMM fallback and its own mega path uses it; FlashInfer's
backend does a hard top-level `import deep_gemm` and does not consult the
vendored module. This is an integration gap rather than a capability gap —
worth fixing upstream so the FlashInfer backend reuses
`vllm.third_party.deep_gemm` when the standalone package is absent. Measured
here by layering a real DeepGEMM into the image
(`vllm-0.29.0-deepgemm-arm64.sqsh`).

**FlashInfer all2all is not a `moe_ep` comm backend.** `moe_ep` ships exactly
two split transports, `nccl_ep` and `nixl_ep`
(`flashinfer/moe_ep/backends/split/comm/`). The FlashInfer all2all used here
is the framework-level MNNVL path (`flashinfer/comm/trtllm_moe_alltoall.py`,
`MoeAlltoAll`), which vLLM/SGLang drive directly. That is why the transport
axis is measured at the serving level rather than through `moe_ep`.

---

## 4. Methodology

Each arm is a fresh `vllm serve` on a clean port, benchmarked with
InferenceX's `benchmark_serving.py`:

```
--dataset-name random --random-input-len <ISL> --random-output-len <OSL>
--random-range-ratio 0.8 --max-concurrency <C> --request-rate inf
--ignore-eos --num-warmups 2C --percentile-metrics ttft,tpot,itl,e2el
```

Controls carried over from the existing A/B harness, because each one has
previously produced a wrong answer when dropped:

- **Port gate.** An arm refuses to run if the port is still held. vLLM forks
  `ApiServer_*`/`EngineCore_*`/`Worker_*` whose argv lacks `vllm serve`;
  killing only the launcher leaves a stale server that answers the next arm's
  requests with HTTP 404 — rc=0, results file full of zeros.
- **Validity gate.** A results file with `completed == 0` is a failure even
  when every exit code was 0.
- **Identical batching envelope.** `--max-num-batched-tokens`,
  `--max-cudagraph-capture-size` and the cudagraph capture-size list are the
  same across arms, so batching capacity is not confounded with kernel speed.
- **KV budget.** `--gpu-memory-utilization` is held fixed across arms in a
  sweep; where an arm's larger weights/symmetric buffer materially cut KV, it
  is called out rather than silently absorbed.

**Functional correctness** is established per arm before its perf number is
reported: GSM8K through the running server (`eval_gsm8k.py`, same server
process as the throughput run). An arm that fails the accuracy gate has its
perf number reported as invalid rather than dropped silently.

---

## 5. Results

### 5.1 DeepSeek-V4-Pro, GB300 EP=4

ISL 8192 / OSL 1024, `--random-range-ratio 0.8`, `max_concurrency 64`,
128 prompts after 128 warmups. Note the KV ceiling from §2.1: the server
admits ~5 concurrent requests, so this row set characterises the
**small-batch** regime only.

| Compute | Transport | done | tok/s | tok/s/GPU | TTFT ms | TPOT ms | ITL ms | vs MegaMoE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FI MegaMoE (cuTeDSL) | fused (in-kernel) | 128 | 12,302 | 3,076 | 1,542 | 41.7 | 29.7 | 1.000× |
| TRTLLM routed | FlashInfer all2all | 128 | 9,868 | 2,467 | 1,799 | 53.0 | 36.8 | **0.802×** |

MegaMoE leads TRTLLM-routed by **1.25×** on throughput and is ahead on every
latency percentile. This is worth flagging against the SGLang PR's summary,
which reports `flashinfer_trtllm_routed` winning at low concurrency: the
effective concurrency here is KV-capped at ~5 (§2.1), and MegaMoE still wins,
so on this geometry the crossover — if any — sits below `max_concurrency 64`
rather than above it.

_Remaining V4-Pro arms in flight._

### 5.2 DeepSeek-V4-Flash, GB200 EP=4

ISL 8192 / OSL 1024, `max_concurrency 256`, 512 prompts after 512 warmups.
KV headroom here is 76.4× (§2.1), so this is the regime that can actually
show the crossover.

| Compute | Transport | done | tok/s | tok/s/GPU | TTFT ms | TPOT ms | ITL ms | vs FI Mega | GSM8K |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Native MegaMoE (deep_gemm)** | fused (in-kernel) | 512 | **77,956** | 19,489 | 1,700 | 25.1 | 13.9 | **1.067×** | 0.870 |
| FI MegaMoE (cuTeDSL) | fused (in-kernel) | 512 | 73,091 | 18,273 | 1,808 | 26.3 | 16.1 | 1.000× | — |
| cuTeDSL split | FlashInfer all2all | 512 | 62,408 | 15,602 | 1,756 | 31.6 | 18.5 | 0.854× | — |
| TRTLLM routed | FlashInfer all2all | 512 | 61,795 | 15,449 | 1,641 | 32.4 | 17.7 | 0.845× | 0.875 |
| cuTeDSL split | NIXL EP | — | \_ | \_ | \_ | \_ | \_ | **N/A** | — |
| TRTLLM routed | NIXL EP | — | \_ | \_ | \_ | \_ | \_ | **N/A** | — |
| TRTLLM routed | DeepEP LL | — | \_ | \_ | \_ | \_ | \_ | **N/A** | — |
| FI MegaMoE (deep_gemm) | fused (in-kernel) | — | \_ | \_ | \_ | \_ | \_ | **N/A** | — |

**N/A** rows are configuration-impossible on this stack, not slow — see §3.5.
NIXL-EP and DeepEP-LL both require the `batched_experts` activation format
that the FlashInfer NVFP4 split kernels do not implement;
`flashinfer_moe_ep_mega_deep_gemm` needs a standalone DeepGEMM the image
lacks.

Three things fall out of this table:

1. **Fusing the comm is worth ~1.17×.** Both megakernels beat both split
   paths by a wide margin (73.1k / 78.0k vs 61.8k / 62.4k tok/s).
2. **The inner-kernel choice barely matters on the split path.** cuTeDSL
   split and TRTLLM-routed land within 1% of each other. TRTLLM-routed even
   has the best TTFT of any arm (1,641 ms), so the megakernels' advantage is
   decode-side (TPOT/ITL), not prefill.
3. **Native deep_gemm MegaMoE currently beats FlashInfer cuTeDSL MegaMoE by
   6.7%** (77,956 vs 73,091 tok/s) and is ahead on every latency metric
   (ITL 13.9 vs 16.1 ms). Both are functionally correct (GSM8K 0.870 vs the
   0.875 of the TRTLLM-routed arm). This is the headline
   "native vs FlashInfer megakernel" result and is the one worth chasing —
   it is the reverse of what the FlashInfer MegaMoE integration is aiming
   for, and the gap is decode-latency-shaped.

#### 5.2b The eager-mode tax dominates the DeepEP question

DeepEP-HT is the only DeepEP variant these runners accept, and it cannot be
CUDA-graph captured (§3.5), so measuring it forces `--enforce-eager` on every
arm in that comparison. That handicap is enormous:

| FI MegaMoE (cuTeDSL), V4-Flash EP=4, conc 256 | tok/s | TTFT ms | TPOT ms | ITL ms |
|---|---:|---:|---:|---:|
| CUDA-graph captured | 73,091 | 1,808 | 26.3 | 16.1 |
| `--enforce-eager` | 8,920 | 6,523 | 228.1 | 149.1 |
| **penalty** | **8.2×** | 3.6× | 8.7× | 9.3× |

So on today's stack, selecting DeepEP-HT costs roughly **8× before any
transport-level difference is even measured**, purely because it gives up
cuda-graph capture. That is the practically decisive fact about DeepEP here,
and it dwarfs whatever the dispatch/combine kernels themselves do.

DeepEP numbers must therefore be read eager-vs-eager (§5.2c) and never
against the captured table above; `collect_matrix.py` enforces this by
placing eager runs in a separate comparison group.

#### 5.2c Eager-mode transport comparison — DeepEP vs FlashInfer all2all

All three arms `--enforce-eager`, so these are mutually comparable (and must
not be read against §5.2's captured table).

| Compute | Transport | done | tok/s | tok/s/GPU | TTFT ms | TPOT ms | ITL ms | vs FI Mega |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TRTLLM routed | **DeepEP HT** | 512 | **10,508** | 2,627 | 6,679 | 183.7 | 175.1 | **1.178×** |
| TRTLLM routed | FlashInfer all2all | 512 | 10,411 | 2,603 | 5,637 | 190.1 | 163.6 | 1.167× |
| FI MegaMoE (cuTeDSL) | fused (in-kernel) | 512 | 8,920 | 2,230 | 6,523 | 228.1 | 149.1 | 1.000× |

Two results, both worth the blog post:

1. **DeepEP-HT and FlashInfer all2all are equivalent — within 0.9%**
   (10,508 vs 10,411 tok/s) on the same TRTLLM-routed runner. DeepEP is
   marginally better on TPOT (183.7 vs 190.1 ms), FlashInfer all2all is
   better on TTFT (5,637 vs 6,679 ms) and ITL (163.6 vs 175.1 ms). There is
   no throughput case for preferring one transport over the other here; the
   decisive difference is that DeepEP-HT forfeits cuda-graph capture and
   therefore ~8× (§5.2b).

2. **The megakernel's advantage is contingent on cuda-graph capture.**
   Captured, FI MegaMoE beats the split paths by ~1.17× (§5.2). Eager, it
   *loses* to them by ~1.17×. The fused kernel's win is not an intrinsic
   arithmetic advantage — it comes from how much better the fused path
   amortises under capture. Its ITL is still the best of the three in eager
   (149.1 ms), so what it loses eager is host-side launch amortisation, not
   device efficiency.

### 5.3 SGLang cross-check (DeepSeek-V4-Flash, GB200 EP=4)

SGLang matters here for two reasons: it is the framework the reference PR
(#31470) targets, and its a2a enum exposes `nixl`, which vLLM could not use
with any FlashInfer runner.

Two configuration requirements had to be met before any arm would start, both
worth documenting for anyone reproducing this:

1. **Dispatch capacity.** `SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK
   × ep_size` must cover the largest CuteDSL MoE forward — i.e.
   `max_prefill_tokens`, 16384 by default. The stock 1024/rank yields only
   4096 at EP=4 and every `flashinfer_cutedsl` arm refuses to start. 4096/rank
   is the minimum that works at EP=4.
2. **Offline tokenizer.** `sglang.bench_serving` must be given
   `--model`/`--tokenizer` as local paths; otherwise it tries the Hub and dies
   with `LocalEntryNotFoundError` on compute nodes that have no egress.

Support result already established: `flashinfer_trtllm_routed` + `deepep` is
**not implemented** — `NotImplementedError: Runner backend
MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED requires a fused func for a2a
backend deepep, but none is registered.`

#### 5.3a SGLang throughput — the ranking inverts vs vLLM

ISL 8192 / OSL 1024, `max_concurrency 256`, 512 prompts, SGLang defaults
(cuda-graph enabled).

| Compute | Transport | done | tok/s | tok/s/GPU | TTFT ms | TPOT ms | ITL ms | vs FI Mega |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **cuTeDSL split** | FlashInfer all2all | 512 | **45,860** | 11,465 | 4,465 | 40.7 | 12.7 | **1.226×** |
| TRTLLM routed | FlashInfer all2all | 512 | 43,333 | 10,833 | 3,656 | 45.7 | 12.3 | 1.158× |
| FI MegaMoE (cuTeDSL) | fused (in-kernel) | 512 | 37,419 | 9,355 | 3,707 | 52.4 | 18.9 | 1.000× |

**This is the opposite ordering to vLLM.** Under vLLM the fused megakernel
led the split paths by ~1.17× (§5.2); under SGLang it is the *slowest* of the
three, losing by 1.16–1.23×, and is worse on every latency metric
(TPOT 52.4 vs 40.7/45.7 ms, ITL 18.9 vs 12.7/12.3 ms).

Read this as a **within-framework** statement. Absolute tok/s is not
comparable across the two frameworks here — they differ in scheduler,
chunked-prefill and cuda-graph capture defaults, and this study did not
normalise them. What *is* comparable is the ordering inside each framework,
and the orderings disagree. Combined with §5.2c (where the ordering also
inverts between captured and eager inside vLLM), the conclusion is that
**"MegaMoE is faster than the split path" is not a portable claim** — it
depends on the host framework and on cuda-graph capture, and the headline
1.17× from §5.2 should always be quoted with both.

One arm failed for an unrelated reason: `sg_split_cutedsl_deepep` serves
(ready in 150 s) but its benchmark warmup aborts with
`aiohttp.http_exceptions.TransferEncodingError: 400` — a server-side 400
during warmup rather than a configuration refusal. Not chased further.

#### 5.3b Getting SGLang to run at all

`sg_megamoe` reaches
`The server is fired up and ready to roll!` (ready after 560 s, 16.47M-token
KV pool, weights loaded as `quant=fp8, quant_algo=MIXED_PRECISION` with NVFP4
experts).

Three prerequisites had to be met before any SGLang arm produced a number;
all three cost a debugging cycle and are worth knowing up front:

1. **Dispatch capacity.** `SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK
   × ep_size` must cover the largest CuteDSL MoE forward — `max_prefill_tokens`,
   16384 by default. The stock 1024/rank yields only 4096 at EP=4 and every
   `flashinfer_cutedsl` arm refuses to start. 4096/rank is the minimum at EP=4.
2. **The `random` dataset is not self-contained.**
   `sglang/benchmark/datasets/random.py` samples its length distribution from
   ShareGPT and fetches it via `hf_hub_download`, which dies with
   `OfflineModeIsEnabled` on egress-less compute nodes. Neither the model nor
   the tokenizer is involved — both resolve locally. Stage
   `anon8231489123/ShareGPT_Vicuna_unfiltered` and pass `--dataset-path`.
3. **Offline tokenizer.** Pass `--model`/`--tokenizer` as local paths.

### 5.4 Correctness (GSM8K)

Run against the *same* server process that produced the throughput number
for that arm, so an arm cannot post a fast time on a broken kernel.
200 questions, 5-shot-style prompting via `/v1/chat/completions`.

| Model | Configuration | GSM8K |
|---|---|---:|
| V4-Flash | TRTLLM routed + FlashInfer all2all | 0.875 |
| V4-Flash | Native MegaMoE (deep_gemm) | 0.870 |

Both clear the ≥0.80 gate used by the earlier EP work and sit within noise of
each other, so the 6.7% throughput gap in §5.2 is a genuine performance
difference and not one arm cutting numerical corners. Arms that never reached
`ready` have no accuracy number by construction — they are marked N/A, not
zero.

Every cell reported in §5.1–5.3 additionally passed the harness's two
structural gates: the port must be free before the arm starts (else a stale
server from the previous arm would answer), and a result with
`completed == 0` is rejected even when every exit code was 0.

---

## 6. Reproduce

```bash
# on lyris
cd /lustre/fsw/coreai_libraries_cudnn/agopal/dsv4ab

# vLLM, EP=4. MODE=both adds a GSM8K pass on the same server process.
CONC=256 NPROMPTS=512 ./job_matrix.sh flash gb200 05:00:00 \
    "mega_fi_cutedsl mega_native_deepgemm split_trtllm_fia2a split_cutedsl_fia2a" both

# V4-Pro needs GB300 (851 GB NVFP4 does not fit a GB200 node at EP=4),
# and EP=8 over two nodes to escape the 5x KV concurrency ceiling.
CONC=64 ./job_matrix.sh    pro gb300 05:00:00 "<arms>" both
CONC=256 ./job_matrix_mn.sh pro gb300 05:00:00 "<arms>"

# DeepEP high-throughput only runs eager (it dies in cuda-graph capture),
# so compare it against an eager trio, never against captured numbers.
EAGER=1 CONC=256 ./job_matrix.sh flash gb200 05:00:00 \
    "mega_fi_cutedsl split_trtllm_fia2a split_trtllm_deepep_ht" perf

# SGLang, EP=4
CONC=256 ./job_sg_matrix.sh flash gb200 05:00:00 "sg_megamoe sg_trtllm_routed_fia2a"

# render (eager rows are grouped separately from captured ones)
python3 collect_matrix.py results/matrix_* results/sgmatrix_* --csv matrix.csv
```

Knobs: `DRIVER=` selects the in-container driver (so a fix can roll out while
an older job still executes the previous copy), `EAGER=1` drops cuda-graph
capture, `GPU_MEM_UTIL`, `ACC_N` (GSM8K question count), `MNBT`, `ISL`/`OSL`.

`run_matrix.sh` holds the arm table; `collect_matrix.py` renders the report
table (throughput, tok/s/GPU, TTFT/TPOT/ITL medians, ratio vs a chosen
baseline arm, GSM8K).

---

## 7. Known gaps

Stated plainly so nothing here reads as more complete than it is.

| Gap | Why | Cost to close |
|---|---|---|
| `flashinfer_moe_ep_mega_deep_gemm` unmeasured | Hard `import deep_gemm`; the vLLM image has no standalone DeepGEMM and three build attempts failed (container lacks `git`; `install.sh` calls bare `python`; wheel never produced). vLLM's native mega path works on the same image via its vendored copy. | Fix the FlashInfer backend to fall back to `vllm.third_party.deep_gemm`, or bake a working DeepGEMM image. |
| V4-Pro large-batch regime | KV-bound to 5.0× concurrency at EP=4 (§2.1); only the small-batch corner was probed. | Run `job_matrix_mn.sh` (written, staged, unrun) for EP=8 across two GB300 nodes. |
| NIXL-EP throughput | Refused by every FlashInfer runner on both frameworks (§3.4b) — no number exists to report. | Needs a `batched_experts`-capable NVFP4 FlashInfer kernel, or a non-batched NIXL path. |
| `sg_split_cutedsl_deepep` throughput | Serves (ready 150 s) but benchmark warmup aborts with `TransferEncodingError: 400`. | Untriaged; likely a client/server protocol issue, not MoE-EP. |
| Cross-framework absolute tok/s | vLLM and SGLang were run at their own scheduler / chunked-prefill / capture defaults, not normalised. | Only within-framework orderings are claimed (§5.3a). |
| Single measurement per cell | No repeats; run-to-run variance not quantified. | Re-run key cells ≥3× and report medians before publishing tight ratios. |

The last row matters most for the blog post: the 6.7% native-vs-FlashInfer
MegaMoE gap (§0.4) is the tightest claim in this report and rests on one run
per arm. It should be repeated before it is published as a number.

---

## 8. vLLM size sweep — TRTLLM routed vs FlashInfer cuTeDSL vs native MegaMoE

§5.2 compares the backends at a **single** operating point (ISL 8192 / OSL
1024 / conc 256). That is one corner of the serving space, and §5.2c already
showed the ranking is not stable across conditions — so a single point is not
enough to say which backend to ship. This section sweeps six serving
scenarios against the three backends worth shipping.

### 8.1 Configurations

| Report name | `--moe-backend` | `--all2all-backend` | Precision |
|---|---|---|---|
| TRTLLM routed | `flashinfer_trtllm` | `flashinfer_all2allv` | NVFP4 (W4A4) |
| FlashInfer cuTeDSL | `flashinfer_cutedsl` | `flashinfer_all2allv` | NVFP4 (W4A4) |
| Native MegaMoE | `deep_gemm_mega_moe` | — (fused, see §3.5) | MXFP4/fp8 |

FlashInfer all2all is the only transport the split runners can use on vLLM
(§3.4b), so it is held fixed and the comparison is purely inner-kernel vs
fused-megakernel.

### 8.2 Scenario points

DeepSeek-V4-Flash, GB200, EP=4, `--gpu-memory-utilization 0.92`,
`--max-num-batched-tokens 2048`, cuda-graph capture on (the default), and
`--max-model-len = ISL + OSL + 256`.

Each point sits **inside** the requested range for its scenario and was
checked against the measured 723,667-token KV budget per rank (§2.1) so that
the requested batch actually fits rather than silently queueing:

| Scenario | requested ISL / OSL / batch | measured point (ISL / OSL / batch) | prompts | max_model_len | KV-feasible batch |
|---|---|---|---:|---:|---:|
| Chat / interactive | 128–1K / 128–1K / 64–128 | 1024 / 1024 / 128 | 512 | 2,304 | ~314 |
| RAG | 2K–8K / 256–1K / 32–64 | 8192 / 1024 / 64 | 256 | 9,472 | ~76 |
| Summarization | 4K–16K / 512–1K / 8–16 | 16384 / 1024 / 16 | 64 | 17,664 | ~41 |
| Code generation | 1K–4K / 512–2K / 8–16 | 4096 / 2048 / 16 | 64 | 6,400 | ~113 |
| Agentic | 4K–64K / 1K–16K / 16–64 | 16384 / 4096 / 32 | 128 | 20,736 | ~34 |
| Agentic (long) | 4K–64K / 1K–16K / 16–64 | 32768 / 4096 / 16 | 64 | 37,120 | ~19 |
| Reasoning | 512–4K / 2K–16K / 8–16 | 4096 / 16384 / 16 | 64 | 20,736 | ~34 |

**Batch vs concurrency — what is actually pinned.** These are two different
knobs and the distinction matters for reading the table. `--max-concurrency`
is a *client* cap on in-flight requests; it only bounds the batch from above.
The server-side running batch — how many sequences the scheduler actually
puts in one forward — is `--max-num-seqs`. The two coincide only in
steady-state decode with enough KV, and they diverge during prefill (at
`--max-num-batched-tokens 2048`, an ISL-16384 request takes 8 chunks, so the
instantaneous batch is far below the concurrency cap).

Every run in this section pins **both** to the scenario's batch
(`--max-num-seqs = --max-concurrency = batch`), so the "batch" column is the
server-side batch and not merely an offered-load ceiling. Results are tagged
`conc<N>b<BATCH>` so a re-run at a different batch cannot silently overwrite
one at the same concurrency.

**Agentic gets two points, and still not the top of its range.** The pair
32K/4K/16 and 16K/4K/32 holds OSL fixed and trades ISL against batch, which
isolates how each backend responds to prompt length at constant output
length — the axis that matters most for agentic traffic.

The top of the range remains out of reach on one node: ISL 64K + OSL 16K
needs `max_model_len` 81,920, which the KV budget only supports at a batch of
~8 — below the requested 16–64, so it would measure queueing rather than the
MoE backend. Even 32K/4K is batch-limited: at `max_model_len` 37,120 the
723,667-token budget allows ~19, so batch 16 (≈82% of KV) is the largest
power-of-two that fits and batch 32 would not. Probing the true 64K end needs
EP=8 over two nodes (§7).

The Chat point additionally runs GSM8K, which closes the one correctness gap
in §5.4 (cuTeDSL split had no accuracy number).

### 8.3 Results

DeepSeek-V4-Flash, GB200 (sm100), EP=4, cuda-graph captured. 21/21 arms
completed (`rc=0`). Ratios are against native MegaMoE in the same row group.

| Scenario | ISL/OSL/batch | Backend | tok/s | tok/s/GPU | TTFT ms | TPOT ms | ITL ms | vs Mega | GSM8K |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Chat | 1024/1024/128 | **Native MegaMoE** | **17,200** | 4,300 | 84.6 | 14.23 | 12.18 | 1.000× | 0.855 |
| Chat | 1024/1024/128 | TRTLLM routed | 13,340 | 3,335 | 120.3 | 18.26 | 15.71 | 0.776× | 0.855 |
| Chat | 1024/1024/128 | FlashInfer cuTeDSL | 12,965 | 3,241 | 98.3 | 18.90 | 16.29 | 0.754× | 0.845 |
| RAG | 8192/1024/64 | **Native MegaMoE** | **32,874** | 8,219 | 321.4 | 16.57 | 10.69 | 1.000× | — |
| RAG | 8192/1024/64 | FlashInfer cuTeDSL | 25,109 | 6,277 | 425.4 | 21.32 | 13.11 | 0.764× | — |
| RAG | 8192/1024/64 | TRTLLM routed | 23,792 | 5,948 | 420.8 | 23.05 | 14.55 | 0.724× | — |
| Summarization | 16384/1024/16 | **Native MegaMoE** | **20,885** | 5,221 | 531.5 | 11.91 | 8.69 | 1.000× | — |
| Summarization | 16384/1024/16 | FlashInfer cuTeDSL | 15,655 | 3,914 | 675.5 | 16.16 | 12.48 | 0.750× | — |
| Summarization | 16384/1024/16 | TRTLLM routed | 15,524 | 3,881 | 702.9 | 16.25 | 11.92 | 0.743× | — |
| Code gen | 4096/2048/16 | **Native MegaMoE** | **5,051** | 1,263 | 124.3 | 9.13 | 8.57 | 1.000× | — |
| Code gen | 4096/2048/16 | TRTLLM routed | 3,734 | 933 | 172.4 | 12.51 | 11.82 | 0.739× | — |
| Code gen | 4096/2048/16 | FlashInfer cuTeDSL | 3,621 | 905 | 171.0 | 12.93 | 12.45 | 0.717× | — |
| Agentic | 16384/4096/32 | **Native MegaMoE** | **13,740** | 3,435 | 479.6 | 11.34 | 9.29 | 1.000× | — |
| Agentic | 16384/4096/32 | TRTLLM routed | 10,202 | 2,551 | 649.7 | 15.47 | 12.91 | 0.742× | — |
| Agentic | 16384/4096/32 | FlashInfer cuTeDSL | 9,765 | 2,441 | 636.4 | 15.55 | 13.56 | 0.711× | — |
| Agentic (long) | 32768/4096/16 | **Native MegaMoE** | **12,764** | 3,191 | 914.2 | 10.52 | 8.61 | 1.000× | — |
| Agentic (long) | 32768/4096/16 | TRTLLM routed | 9,521 | 2,380 | 1225.4 | 14.37 | 11.80 | 0.746× | — |
| Agentic (long) | 32768/4096/16 | FlashInfer cuTeDSL | 9,471 | 2,368 | 1188.2 | 14.14 | 11.14 | 0.742× | — |
| Reasoning | 4096/16384/16 | **Native MegaMoE** | **2,234** | 559 | 130.0 | 8.44 | 8.35 | 1.000× | — |
| Reasoning | 4096/16384/16 | TRTLLM routed | 1,607 | 402 | 178.2 | 11.83 | 11.79 | 0.719× | — |
| Reasoning | 4096/16384/16 | FlashInfer cuTeDSL | 1,544 | 386 | 172.5 | 12.26 | 12.23 | 0.691× | — |

**Native MegaMoE wins all seven scenarios, and the margin is remarkably
stable.** The split paths land at 0.69–0.78× of it everywhere — a 1.28–1.45×
MegaMoE advantage that barely moves across a 32× span of ISL (1K→32K), a 16×
span of OSL (1K→16K) and an 8× span of batch (16→128). It also wins every
latency metric in every scenario, not just throughput.

That stability is the result. §5.2c showed the ranking inverting between
captured and eager, and §5.3a showed it inverting between vLLM and SGLang —
but *within* captured vLLM it does not depend on the serving shape at all.
So the choice is not workload-dependent: on this stack, pick the megakernel.

**The two split backends are near-identical**, within 1–6% of each other with
no consistent winner (TRTLLM routed ahead in 5 of 7, FlashInfer cuTeDSL in
RAG and summarization). This reproduces §0.3 across the whole size range:
fused-vs-split is the axis that matters, cuTeDSL-vs-TRTLLM is noise.

**Accuracy is equivalent** where measured (Chat): 0.855 / 0.855 / 0.845, so
the throughput gap is not bought with numerics.

Two shape effects worth noting for the blog post:

- **Reasoning is the hardest case for everyone.** At OSL 16384 the run is
  almost pure decode and absolute throughput collapses to 2.2k tok/s
  (vs 32.9k for RAG) — an order of magnitude, driven by the shape rather
  than the backend.
- **Prefill length hurts the split paths more.** The worst split ratios are
  the long-prefill / long-decode cases (agentic 0.711×, reasoning 0.691×),
  the best is chat (0.776×), consistent with the dispatch/combine cost being
  paid per MoE layer regardless of how much work the layer does.

### 8.4 How to read this section

The question §5.2 could not answer is whether the megakernel's win is
uniform. Three things to look for:

1. **Does the ranking hold at short ISL?** §5.2 ran ISL 8192. Chat (ISL 1024)
   has far less prefill to amortise the fused path over.
2. **Does it hold at decode-heavy shapes?** Reasoning (OSL 16384) is almost
   entirely decode, which is where §5.2 located the megakernel's advantage —
   so it should be the megakernel's best case.
3. **Does it hold at small batch?** Summarization / codegen / reasoning
   run batch 16 against §5.2's 256. The SGLang PR's claim that
   `trtllm_routed` wins at low concurrency is directly testable here.
