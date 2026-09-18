# MoE-EP configuration matrix harness

End-to-end serving harness for the DeepSeek-V4 MoE expert-parallel
configuration study. Full methodology, measured results and the structural
findings live in
[`docs/design_docs/moe_ep_config_report.md`](../../docs/design_docs/moe_ep_config_report.md).

## Scripts

| Script | Role |
|---|---|
| `run_matrix.sh` | in-container arm driver, single node, EP=4 (vLLM) |
| `run_matrix_mn.sh` | in-container arm driver, two nodes, EP=8 (vLLM) |
| `run_sg_matrix.sh` | in-container arm driver, single node, EP=4 (SGLang) |
| `job_matrix.sh` / `job_matrix_mn.sh` / `job_sg_matrix.sh` | SLURM/pyxis submitters |
| `collect_matrix.py` | renders the report table from both frameworks' outputs |

## Usage

```bash
# vLLM, one node, EP=4 — perf + GSM8K for the listed arms
CONC=256 NPROMPTS=512 ./job_matrix.sh flash gb200 04:00:00 \
    "split_cutedsl_fia2a split_cutedsl_nixl split_cutedsl_deepep" both

# vLLM, two nodes, EP=8 — needed for V4-Pro's large-batch regime
CONC=256 ./job_matrix_mn.sh pro gb300 04:00:00 "mega_fi_cutedsl"

# SGLang, one node, EP=4
CONC=256 ./job_sg_matrix.sh flash gb200 04:00:00 "sg_megamoe sg_trtllm_routed_fia2a"

# render
python3 collect_matrix.py results/matrix_* results/sgmatrix_* --csv matrix.csv
```

## Arms

vLLM (`run_matrix.sh`, `run_matrix_mn.sh`):

| Arm | `--moe-backend` | `--all2all-backend` |
|---|---|---|
| `split_cutedsl_{fia2a,nixl,deepep}` | `flashinfer_cutedsl` | `flashinfer_all2allv` / `nixl_ep` / `deepep_low_latency` |
| `split_trtllm_{fia2a,nixl,deepep}` | `flashinfer_trtllm` | same three |
| `mega_fi_cutedsl` | `flashinfer_moe_ep_mega_cutedsl` | — |
| `mega_fi_deepgemm` | `flashinfer_moe_ep_mega_deep_gemm` | — |
| `mega_native_deepgemm` | `deep_gemm_mega_moe` | — |

SGLang (`run_sg_matrix.sh`): `sg_split_cutedsl_*`, `sg_trtllm_routed_*`
(`flashinfer` / `nixl` / `deepep`), and `sg_megamoe{,_ikr,_cmb_nvfp4,_cmb_mxfp8}`.

## Two things that will bite you

**Mega backends ignore `--all2all-backend`.** vLLM's DeepSeek-V4 model
branches on `use_mega_moe` into `_init_mega_moe_experts`, which uses
`get_ep_group()` directly and never enters the modular FusedMoE path. Passing
a transport flag to a mega arm produces a silently mislabelled cell — the
transport axis is only real for the `split_*` arms.

**Kill every vLLM child between arms.** vLLM forks
`ApiServer_*` / `EngineCore_*` / `Worker_*` processes whose argv does not
contain `vllm serve`; killing only the launcher leaves a server holding the
port, and the next arm's client then talks to a stale server serving the
previous model — HTTP 404 on every request, exit code 0, and a results file
full of zeros. Both drivers gate on the port being free before an arm runs
and reject a result with `completed == 0`.
