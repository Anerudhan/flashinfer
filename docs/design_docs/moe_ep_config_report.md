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

> **Status: in progress.** Environment and configuration mapping are settled
> and verified against the shipped code (below). Result tables are populated
> as runs land; every empty cell is a run that has not completed yet, not a
> measured zero.

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
| **DeepEP MegaMoE** (native) | `deep_gemm_mega_moe` | base (MXFP4/fp8) |

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
| `mega_deepep_native` | `deep_gemm_mega_moe` | `deepep_low_latency` |

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

### 3.5 Two axes that do not exist as asked — and why

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

_Pending — runs in flight._

### 5.2 DeepSeek-V4-Flash, GB200 EP=4

_Pending — checkpoint staging in flight._

### 5.3 SGLang cross-check

_Pending._

### 5.4 Correctness (GSM8K)

_Pending._

---

## 6. Reproduce

```bash
# on lyris
cd /lustre/fsw/coreai_libraries_cudnn/agopal/dsv4ab
CONC=256 NPROMPTS=512 ./job_matrix.sh pro   gb300 04:00:00 "<arms>" both
CONC=256 NPROMPTS=512 ./job_matrix.sh flash gb200 04:00:00 "<arms>" both
python3 collect_matrix.py results/matrix_* --csv matrix.csv
```

`run_matrix.sh` holds the arm table; `collect_matrix.py` renders the report
table (throughput, tok/s/GPU, TTFT/TPOT/ITL medians, ratio vs a chosen
baseline arm, GSM8K).
