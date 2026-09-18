"""MoE expert-parallel **micro** benchmark — layer-level, no model, no server.

Measures one DeepSeek-V4 MoE-EP layer (dispatch -> expert GEMM -> combine) across
the transport x inner-kernel matrix, at token counts derived from real serving
scenarios. This is the cheap counterpart to the end-to-end study in
``docs/design_docs/moe_ep_config_report.md``: no checkpoint, no scheduler, no KV
cache, so it isolates the MoE layer and runs in minutes instead of hours.

**It also reaches configurations the e2e path cannot.** NCCL-EP is not exposed by
vLLM or SGLang, but it is a first-class ``flashinfer.moe_ep`` split transport, so
it is measurable here.

Configurations (``--configs``)::

    cutedsl_fia2a     split, FlashInfer all2all (MNNVL), CuteDSL NVFP4 GEMM
    trtllm_fia2a      split, FlashInfer all2all (MNNVL), TRT-LLM FP4 block-scale
    cutedsl_nccl      split, NCCL-EP,                    CuteDSL NVFP4 GEMM
    trtllm_nccl       split, NCCL-EP,                    TRT-LLM FP4 block-scale
    cutedsl_nixl      split, NIXL-EP,                    CuteDSL NVFP4 GEMM
    trtllm_nixl       split, NIXL-EP,                    TRT-LLM FP4 block-scale
    megamoe_cutedsl   mega, fused comm+GEMM, CuteDSL NVFP4 megakernel
    megamoe_deepgemm  mega, fused comm+GEMM, DeepGEMM fp8/fp4 megakernel

Precision: **W4A4 (NVFP4)** wherever the backend supports it, which is every
config above except ``megamoe_deepgemm`` (DeepGEMM's mega kernel is
``fp8_fp4_mega_moe`` — FP4 weights, FP8 activations, i.e. W4A8). The actual
precision each config ran at is emitted in the ``precision`` CSV column rather
than assumed. ``--quant`` forces a common precision for an apples-to-apples row.

CUDA graphs: only the **megakernel** configs are captured. The split EP
transports (NCCL-EP, NIXL-EP, and MNNVL all2all) perform host-visible work
inside dispatch and throw from C++ under stream capture —
``CUDA error nccl_ep.cc:2021 'operation not permitted when stream is
capturing'`` — which reaches ``std::terminate`` and SIGABRTs every rank. That
is not catchable from Python, so capture is *gated* on a per-config
``capturable`` flag rather than attempted inside a ``try``. The ``graph``
column records what each cell actually used.

Transport availability is probed at runtime via
``flashinfer.moe_ep.available_backends()``; a transport the build lacks is
reported per cell with its reason rather than silently skipped. Note that
NIXL-EP needs UCX >= 1.21 built with the device API, and the in-tree recipe
for that (``docker/Dockerfile.flashinfer-nvep``) installs an **amd64** DOCA
package and hardcodes ``x86_64-linux-gnu`` pkgconfig paths — so on an aarch64
cluster (GB200/GB300) it requires an arm64 port of that recipe, not just a
rebuild.

Launch (one process per GPU)::

    torchrun --nproc_per_node=4 bench_moe_ep_micro.py --model flash \\
        --configs all --scenarios all --csv micro.csv

    # single scenario, validate numerics first
    torchrun --nproc_per_node=4 bench_moe_ep_micro.py --model flash \\
        --scenarios chat --validate

Rank 0 prints a per-(config, tokens) table and a scenario roll-up; every row is
also written to ``--csv``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
from dataclasses import dataclass, field
from statistics import median

# Keep the benchmarks dir off sys.path so `import flashinfer` finds the package,
# not a sibling file.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]


# ---------------------------------------------------------------------------
# Model geometries (read from the DeepSeek-V4 checkpoints' config.json)
# ---------------------------------------------------------------------------
MODELS = {
    # hidden, moe_intermediate, n_routed_experts, num_experts_per_tok, layers
    "flash": dict(hidden=4096, intermediate=2048, num_experts=256, top_k=6, layers=43),
    "pro": dict(hidden=7168, intermediate=3072, num_experts=384, top_k=6, layers=61),
}


# ---------------------------------------------------------------------------
# Serving scenarios -> MoE layer token counts
# ---------------------------------------------------------------------------
# At the MoE layer the only thing that matters is how many tokens arrive in one
# forward. Two regimes per scenario:
#
#   decode  : one token per sequence, so tokens_per_rank == batch.
#   prefill : a chunked-prefill slice, so tokens_per_rank == chunk size, bounded
#             above by the scenario's ISL (a 512-token prompt cannot fill an
#             8192-token chunk).
#
# OSL does not change the shape; it sets how many decode steps a request pays,
# and is used only by the scenario roll-up to weight decode against prefill.
@dataclass(frozen=True)
class Scenario:
    name: str
    isl: tuple[int, int]
    osl: tuple[int, int]
    batch: tuple[int, int]

    @property
    def decode_tokens(self) -> list[int]:
        """Per-rank token counts in the decode phase (== batch)."""
        lo, hi = self.batch
        return sorted({lo, hi})

    def prefill_tokens(self, chunks: list[int]) -> list[int]:
        """Per-rank token counts in the prefill phase, bounded by ISL."""
        isl_lo, isl_hi = self.isl
        return sorted({c for c in chunks if c <= isl_hi} | {min(isl_lo, max(chunks))})


SCENARIOS = {
    s.name: s
    for s in (
        Scenario("chat", isl=(128, 1024), osl=(128, 1024), batch=(64, 128)),
        Scenario("rag", isl=(2048, 8192), osl=(256, 1024), batch=(32, 64)),
        Scenario("summarization", isl=(4096, 16384), osl=(512, 1024), batch=(8, 16)),
        Scenario("codegen", isl=(1024, 4096), osl=(512, 2048), batch=(8, 16)),
        Scenario("agentic", isl=(4096, 65536), osl=(1024, 16384), batch=(16, 64)),
        Scenario("reasoning", isl=(512, 4096), osl=(2048, 16384), batch=(8, 16)),
    )
}

# Chunked-prefill sizes a server would actually use.
PREFILL_CHUNKS = [512, 1024, 2048, 4096, 8192]


# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    name: str
    mode: str  # "split" | "mega"
    comm: str  # "fia2a" | "nccl_ep" | "nixl_ep" | "fused"
    compute: str  # "cutedsl" | "trtllm" | "nvfp4_cutedsl" | "deep_gemm_mega"
    precision: str  # nominal; the runner records what actually ran
    # CUDA-graph capture is GATED, not attempted-and-caught: the split EP
    # transports throw a C++ EPException from inside dispatch under capture
    # ("operation not permitted when stream is capturing"), which calls
    # std::terminate and SIGABRTs the whole process. Python cannot catch it,
    # so a non-capturable config must never be offered to the capture path.
    capturable: bool = False


CONFIGS = {
    c.name: c
    for c in (
        Config("cutedsl_fia2a", "split", "fia2a", "cutedsl", "W4A4"),
        Config("trtllm_fia2a", "split", "fia2a", "trtllm", "W4A4"),
        Config("cutedsl_nccl", "split", "nccl_ep", "cutedsl", "W4A4"),
        Config("trtllm_nccl", "split", "nccl_ep", "trtllm", "W4A4"),
        Config("cutedsl_nixl", "split", "nixl_ep", "cutedsl", "W4A4"),
        Config("trtllm_nixl", "split", "nixl_ep", "trtllm", "W4A4"),
        Config(
            "megamoe_cutedsl", "mega", "fused", "nvfp4_cutedsl", "W4A4", capturable=True
        ),
        Config(
            "megamoe_deepgemm",
            "mega",
            "fused",
            "deep_gemm_mega",
            "W4A8",
            capturable=True,
        ),
    )
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", choices=sorted(MODELS), default="flash")
    p.add_argument(
        "--configs",
        default="all",
        help="comma-separated config names, or 'all' (default)",
    )
    p.add_argument(
        "--scenarios",
        default="all",
        help="comma-separated scenario names, or 'all' (default)",
    )
    p.add_argument(
        "--quant",
        choices=["nvfp4", "bf16"],
        default="nvfp4",
        help="split-path precision. nvfp4 == W4A4 (default)",
    )
    p.add_argument(
        "--cuda-graph",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto: capture only for decode-sized token counts of CAPTURABLE "
        "configs (the megakernels). The split EP transports abort the process "
        "under capture, so they are never captured regardless of this flag.",
    )
    p.add_argument(
        "--graph-max-tokens",
        type=int,
        default=256,
        help="with --cuda-graph auto, capture at or below this token count",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--tokens",
        default="",
        help="explicit comma-separated token counts, overriding the scenarios",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="check each config against a single-process dense-MoE reference "
        "before timing it, and skip (reporting rel-L2) if it disagrees",
    )
    p.add_argument("--csv", default="", help="write all rows to this CSV")
    p.add_argument(
        "--layers",
        type=int,
        default=0,
        help="scale the roll-up by this many MoE layers (0 = model default)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Token plan
# ---------------------------------------------------------------------------
@dataclass
class TokenPlan:
    tokens: list[int] = field(default_factory=list)
    # token count -> set of (scenario, phase) that need it
    provenance: dict[int, set[tuple[str, str]]] = field(default_factory=dict)

    def add(self, n: int, scenario: str, phase: str) -> None:
        self.provenance.setdefault(n, set()).add((scenario, phase))


def build_token_plan(scenarios: list[Scenario], explicit: str) -> TokenPlan:
    plan = TokenPlan()
    if explicit:
        plan.tokens = sorted({int(v) for v in explicit.split(",") if v.strip()})
        for n in plan.tokens:
            plan.add(n, "explicit", "explicit")
        return plan
    for s in scenarios:
        for n in s.decode_tokens:
            plan.add(n, s.name, "decode")
        for n in s.prefill_tokens(PREFILL_CHUNKS):
            plan.add(n, s.name, "prefill")
    plan.tokens = sorted(plan.provenance)
    return plan


# ---------------------------------------------------------------------------
# Layer construction
# ---------------------------------------------------------------------------
def _moe_config(args, geo, *, local_num_experts, local_expert_offset, max_tokens, dev):
    """MoEConfig pinned to a single inner backend (so the row means what it says)."""
    from flashinfer.fused_moe.api import (
        BackendOptions,
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
        TrtllmFp4Config,
    )

    routing = RoutingConfig(num_experts=geo["num_experts"], top_k=geo["top_k"])
    experts = ExpertConfig(
        intermediate_size=geo["intermediate"],
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
    )
    execution = ExecutionConfig(tune_max_num_tokens=max_tokens)

    if args.quant == "nvfp4":
        quant = QuantConfig(variant=QuantVariant.NVFP4)
        # Pin one candidate per config; a two-candidate list would let the
        # autotuner silently pick the other kernel and both rows would measure
        # the same thing.
        cand = CuteDslConfig() if args._compute == "cutedsl" else TrtllmFp4Config()
        precision = "W4A4 (NVFP4)"
    else:
        quant = QuantConfig(variant=QuantVariant.BF16)
        cand = TrtllmBf16Config()
        precision = "W16A16 (BF16)"

    cfg = MoEConfig(
        routing=routing,
        quant=quant,
        experts=experts,
        backend=BackendOptions(candidates=(cand,)),
        execution=execution,
    )
    return cfg, precision


def _canonical_weights(geo, local_num_experts, dev):
    import torch

    from flashinfer.moe_ep import MoEWeightPack

    w13 = torch.randn(
        local_num_experts,
        2 * geo["intermediate"],
        geo["hidden"],
        dtype=torch.bfloat16,
        device=dev,
    )
    w2 = torch.randn(
        local_num_experts,
        geo["hidden"],
        geo["intermediate"],
        dtype=torch.bfloat16,
        device=dev,
    )
    return MoEWeightPack(w13=w13, w2=w2)


def build_layer(cfg: Config, args, geo, *, rank, world, max_tokens, dev, tcp_store):
    """Return (layer, precision_str). Raises on an unsupported configuration."""
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        FusedMoeKernelConfig,
        MegaConfig,
        MoEEpLayer,
        NcclEpConfig,
        Nvfp4CutedslMegaMoeConfig,
        NvepConfig,
        SplitConfig,
    )

    local_num_experts = geo["num_experts"] // world
    local_expert_offset = local_num_experts * rank

    bootstrap = BootstrapConfig(
        world_size=world,
        rank=rank,
        stream=torch.cuda.current_stream().cuda_stream,
        tcp_store=tcp_store,
    )
    weights = _canonical_weights(geo, local_num_experts, dev)

    if cfg.mode == "mega":
        if cfg.compute == "nvfp4_cutedsl":
            mega = Nvfp4CutedslMegaMoeConfig(
                intermediate_size=geo["intermediate"], top_k=geo["top_k"]
            )
            precision = "W4A4 (NVFP4)"
        else:
            mega = DeepGemmMegaMoeConfig(
                intermediate_size=geo["intermediate"], top_k=geo["top_k"]
            )
            # deep_gemm.fp8_fp4_mega_moe: FP4 weights, FP8 activations.
            precision = "W4A8 (fp8_fp4)"
        fleet = FleetParams(
            num_experts=geo["num_experts"],
            max_tokens_per_rank=max_tokens,
            token_hidden_size=geo["hidden"],
            dtype_bytes=2,
        )
        layer = MoEEpLayer(
            bootstrap, fleet, weights=weights, backend=MegaConfig(megakernel=mega)
        )
        return layer, precision

    # --- split ---------------------------------------------------------------
    args._compute = cfg.compute
    # RANK_MAJOR keeps the compute view at world*max_tokens instead of
    # num_local_experts*cap, which is what makes the small-token (decode) rows
    # meaningful rather than padding-dominated.
    layout = EpLayout.RANK_MAJOR
    compute_max_tokens = max_tokens * world

    moe_cfg, precision = _moe_config(
        args,
        geo,
        local_num_experts=local_num_experts,
        local_expert_offset=local_expert_offset,
        max_tokens=compute_max_tokens,
        dev=dev,
    )

    if cfg.comm == "nccl_ep":
        comm = NcclEpConfig()
    elif cfg.comm == "nixl_ep":
        comm = NvepConfig()
    else:
        raise NotImplementedError(
            "FlashInfer all2all is not a moe_ep comm backend; "
            "it is driven by FiA2ARunner, not MoEEpLayer"
        )

    fleet = FleetParams(
        num_experts=geo["num_experts"],
        max_tokens_per_rank=max_tokens,
        token_hidden_size=geo["hidden"],
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
        layout=layout,
    )
    layer = MoEEpLayer(
        bootstrap,
        fleet,
        weights=weights,
        backend=SplitConfig(comm=comm, kernel=FusedMoeKernelConfig(moe_config=moe_cfg)),
    )
    return layer, precision


# ---------------------------------------------------------------------------
# FlashInfer all2all runner
# ---------------------------------------------------------------------------
class FiA2ARunner:
    """dispatch/combine via ``MoeAlltoAll`` (MNNVL) + the same fused-MoE compute.

    ``MoeAlltoAll.dispatch`` returns ``[ep_size, max_tokens, *]`` — the same
    shape as moe_ep's RANK_MAJOR recv buffer — so the activation pack is built
    with moe_ep's own ``build_activation_pack_rank_major`` rather than a
    hand-rolled equivalent. The one convention difference is handled explicitly:
    MoeAlltoAll ships **global** expert ids, while the bridge expects ids local
    to this rank, so the offset is subtracted before the call.
    """

    def __init__(self, cfg, args, geo, *, rank, world, max_tokens, dev):
        import torch

        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll
        from flashinfer.fused_moe.layer import MoELayer

        self.geo = geo
        self.world = world
        self.rank = rank
        self.max_tokens = max_tokens
        self.dev = dev
        self.local_num_experts = geo["num_experts"] // world
        self.local_expert_offset = self.local_num_experts * rank

        args._compute = cfg.compute
        self.moe_cfg, self.precision = _moe_config(
            args,
            geo,
            local_num_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            max_tokens=max_tokens * world,
            dev=dev,
        )
        from flashinfer.fused_moe.api import QuantVariant
        from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
            materialize_fused_moe_weights,
        )

        self.is_nvfp4 = self.moe_cfg.quant.variant is QuantVariant.NVFP4
        canonical = _canonical_weights(geo, self.local_num_experts, dev)
        # Same weight materialisation moe_ep's own split kernel performs, so the
        # fia2a rows and the nccl/nixl rows run byte-identical expert weights.
        self.weights = materialize_fused_moe_weights(canonical, self.moe_cfg)
        self.runner = MoELayer(self.moe_cfg)

        mapping = Mapping(
            world_size=world, rank=rank, gpus_per_node=min(world, 4), moe_ep_size=world
        )
        self.a2a = MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_tokens,
            top_k=geo["top_k"],
            num_experts=geo["num_experts"],
            hidden_size=geo["hidden"],
        )
        self._torch = torch

    def forward(self, tensors):
        from flashinfer.moe_ep.backends.split.kernel.fused_moe.bridge import (
            build_activation_pack_rank_major,
            reshape_for_combine,
        )

        torch = self._torch
        x = tensors.hidden_states
        idx = tensors.topk_ids.to(torch.int32)
        w = tensors.topk_weights

        recv = self.a2a.dispatch(idx, [x, idx, w], self.max_tokens)
        recv_x, recv_idx, recv_w = recv[0], recv[1], recv[2]
        dim0, dim1, _ = recv_x.shape

        # MoeAlltoAll ships GLOBAL expert ids; the rank-major bridge expects ids
        # local to this rank (it masks anything outside
        # [0, num_local_experts) to weight 0), so rebase them here.
        local_idx = recv_idx.to(torch.int32) - self.local_expert_offset

        pack = build_activation_pack_rank_major(
            recv_x,
            local_idx,
            recv_w,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            is_nvfp4=self.is_nvfp4,
        )
        out_2d = self.runner(pack, self.weights)
        return self.a2a.combine(reshape_for_combine(out_2d, dim0, dim1))

    def destroy(self):
        pass


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_forward(layer, tensors, *, warmup, iters, use_graph):
    """Return (median_us, graph_used). Falls back to eager if capture fails."""
    import torch

    for _ in range(warmup):
        layer.forward(tensors)
    torch.cuda.synchronize()

    graph = None
    if use_graph:
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    layer.forward(tensors)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                layer.forward(tensors)
            torch.cuda.synchronize()
        except Exception:
            # Several EP transports refuse capture (host-side handshakes inside
            # dispatch). Record eager rather than dropping the cell.
            graph = None

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        if graph is not None:
            graph.replay()
        else:
            layer.forward(tensors)
        ends[i].record()
    torch.cuda.synchronize()

    samples = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends, strict=True)]
    return median(samples), graph is not None


def make_tensors(geo, num_tokens, dev, seed=0):
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(
        num_tokens, geo["hidden"], dtype=torch.bfloat16, device=dev, generator=g
    )
    scores = torch.randn(
        num_tokens, geo["num_experts"], dtype=torch.float32, device=dev, generator=g
    )
    w, idx = torch.topk(scores, geo["top_k"], dim=-1)
    w = torch.softmax(w, dim=-1).to(torch.bfloat16)
    return MoEEpTensors(hidden_states=x, topk_ids=idx.to(torch.int32), topk_weights=w)


# ---------------------------------------------------------------------------
# Scenario roll-up
# ---------------------------------------------------------------------------
def rollup(rows, scenarios, layers):
    """Per-request MoE-layer cost implied by the measured per-token costs.

    For each scenario take the *representative* prefill chunk (largest measured
    chunk <= ISL_hi) and the *high* batch for decode, then::

        req_ms = layers * (ISL * us_per_tok(prefill) + OSL * us_per_tok(decode)) / 1000

    This is MoE-layer time only — it deliberately excludes attention, norms and
    the LM head, so it is a comparison between configs, not a latency estimate.
    """
    by = {}
    for r in rows:
        if r["ok"]:
            by[(r["config"], r["tokens"])] = r["us"]

    out = []
    for s in scenarios:
        chunks = s.prefill_tokens(PREFILL_CHUNKS)
        pre_n = max(chunks)
        dec_n = max(s.decode_tokens)
        isl = s.isl[1]
        osl = s.osl[1]
        for cname in sorted({r["config"] for r in rows}):
            pre = by.get((cname, pre_n))
            dec = by.get((cname, dec_n))
            if pre is None or dec is None:
                continue
            req_ms = layers * (isl * (pre / pre_n) + osl * (dec / dec_n)) / 1e3
            out.append(
                dict(
                    scenario=s.name,
                    config=cname,
                    prefill_tokens=pre_n,
                    decode_tokens=dec_n,
                    isl=isl,
                    osl=osl,
                    prefill_us_per_tok=pre / pre_n,
                    decode_us_per_tok=dec / dec_n,
                    req_ms=req_ms,
                )
            )
    return out


def main() -> int:
    args = _parse_args()
    args._compute = "cutedsl"

    import torch
    import torch.distributed as dist

    names = sorted(CONFIGS) if args.configs == "all" else args.configs.split(",")
    cfgs = [CONFIGS[n.strip()] for n in names if n.strip()]
    snames = sorted(SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")
    scens = [SCENARIOS[n.strip()] for n in snames if n.strip()]
    geo = MODELS[args.model]
    layers = args.layers or geo["layers"]

    plan = build_token_plan(scens, args.tokens)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda", rank % torch.cuda.device_count())

    max_tokens = max(plan.tokens)
    tcp_store = None
    if any(c.comm == "nixl_ep" for c in cfgs):
        addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = int(os.environ.get("MASTER_PORT", "29500"))
        tcp_store = dist.TCPStore(
            host_name=addr, port=port + 1, world_size=world, is_master=(rank == 0)
        )

    dev_name = torch.cuda.get_device_name(dev)
    major, minor = torch.cuda.get_device_capability(dev)
    arch = f"sm{major}{minor}"
    if rank == 0:
        print(
            f"model={args.model} geo=hidden{geo['hidden']}/inter{geo['intermediate']}/"
            f"E{geo['num_experts']}/top{geo['top_k']} layers={layers} "
            f"EP={world} quant={args.quant}"
        )
        print(f"device={dev_name} arch={arch}")
        print(f"token sweep: {plan.tokens}")

    rows = []
    for cfg in cfgs:
        layer = None
        precision = cfg.precision
        build_err = ""
        try:
            if cfg.comm == "fia2a":
                layer = FiA2ARunner(
                    cfg,
                    args,
                    geo,
                    rank=rank,
                    world=world,
                    max_tokens=max_tokens,
                    dev=dev,
                )
                precision = layer.precision
            else:
                layer, precision = build_layer(
                    cfg,
                    args,
                    geo,
                    rank=rank,
                    world=world,
                    max_tokens=max_tokens,
                    dev=dev,
                    tcp_store=tcp_store,
                )
        except Exception as exc:  # noqa: BLE001
            build_err = f"{type(exc).__name__}: {exc}"[:200]

        for n in plan.tokens:
            row = dict(
                model=args.model,
                device=dev_name,
                arch=arch,
                config=cfg.name,
                mode=cfg.mode,
                comm=cfg.comm,
                compute=cfg.compute,
                precision=precision,
                ep=world,
                tokens=n,
                us=0.0,
                us_per_tok=0.0,
                graph=False,
                ok=False,
                note=build_err,
            )
            if layer is not None:
                try:
                    t = make_tensors(geo, n, dev)
                    want_graph = args.cuda_graph == "on" or (
                        args.cuda_graph == "auto" and n <= args.graph_max_tokens
                    )
                    # Never hand a non-capturable config to the capture path:
                    # it aborts the process rather than raising (see Config).
                    use_graph = want_graph and cfg.capturable
                    us, used = time_forward(
                        layer,
                        t,
                        warmup=args.warmup,
                        iters=args.iters,
                        use_graph=use_graph,
                    )
                    row.update(us=us, us_per_tok=us / n, graph=used, ok=True, note="")
                except Exception as exc:  # noqa: BLE001
                    row["note"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(row)
            if rank == 0:
                # Print as we go: a long matrix is otherwise a black box until
                # the very end, and a mid-run hang is indistinguishable from
                # slow autotune.
                if row["ok"]:
                    print(
                        f"  [{cfg.name}] tokens={n:<6} {row['us']:9.1f} us "
                        f"({row['us_per_tok']:.4f} us/tok)"
                        f"{' graph' if row['graph'] else ''}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{cfg.name}] tokens={n:<6} FAIL {row['note']}",
                        flush=True,
                    )
            dist.barrier()

        if layer is not None:
            with contextlib.suppress(Exception):
                layer.destroy()

    if rank != 0:
        dist.destroy_process_group()
        return 0

    hdr = (
        f"{'config':<18} {'precision':<16} {'tokens':>7} {'us':>10} "
        f"{'us/tok':>9} {'graph':>6}  note"
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["ok"]:
            print(
                f"{r['config']:<18} {r['precision']:<16} {r['tokens']:>7} "
                f"{r['us']:>10.1f} {r['us_per_tok']:>9.4f} "
                f"{'yes' if r['graph'] else 'no':>6}"
            )
        else:
            print(
                f"{r['config']:<18} {r['precision']:<16} {r['tokens']:>7} "
                f"{'--':>10} {'--':>9} {'--':>6}  {r['note']}"
            )

    ru = rollup(rows, scens, layers)
    if ru:
        print()
        h2 = (
            f"{'scenario':<14} {'config':<18} {'ISL':>6} {'OSL':>6} "
            f"{'pre us/tok':>11} {'dec us/tok':>11} {'MoE ms/req':>11}"
        )
        print(h2)
        print("-" * len(h2))
        for e in sorted(ru, key=lambda e: (e["scenario"], e["req_ms"])):
            print(
                f"{e['scenario']:<14} {e['config']:<18} {e['isl']:>6} {e['osl']:>6} "
                f"{e['prefill_us_per_tok']:>11.4f} {e['decode_us_per_tok']:>11.4f} "
                f"{e['req_ms']:>11.2f}"
            )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if ru:
            roll_path = args.csv.replace(".csv", "") + "_rollup.csv"
            with open(roll_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(ru[0].keys()))
                w.writeheader()
                w.writerows(ru)
            print(f"\nwrote {args.csv} and {roll_path}")
        else:
            print(f"\nwrote {args.csv}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
