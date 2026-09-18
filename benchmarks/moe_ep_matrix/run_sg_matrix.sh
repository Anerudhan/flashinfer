#!/bin/bash
# In-container single-node EP4 MoE-EP matrix driver for SGLang (DeepSeek-V4).
# Mirrors run_matrix.sh (vLLM) so the two frameworks' arms line up.
#
# Backend surface comes from sglang PR #31470:
#   --moe-runner-backend {flashinfer_cutedsl, flashinfer_trtllm_routed, flashinfer_megamoe}
#   --moe-a2a-backend    {flashinfer, deepep, flashinfer_megamoe}
#   SGLANG_FLASHINFER_MEGAMOE_{COMBINE_DTYPE,IN_KERNEL_FC2_REDUCE,MAX_TOKENS_PER_RANK}
#
# Reads: ROOT OUT MODEL ARMS CONC NPROMPTS ISL OSL MODE EP
set -uo pipefail
: "${ROOT:?}" "${OUT:?}"
MODEL=${MODEL:-pro}
CONC=${CONC:-256}
NPROMPTS=${NPROMPTS:-512}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
MODE=${MODE:-perf}
PORT=${PORT:-8899}
EP=${EP:-4}

export HF_HOME=$ROOT/cache/huggingface HF_HUB_CACHE=$ROOT/cache/huggingface/hub HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer-ws
export TRITON_CACHE_DIR=$ROOT/cache/triton
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sg-pycache

# The FlashInfer A2A dispatch buffer is sized as
# SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK * ep_size and must cover
# the largest CuteDSL MoE forward, which is max_prefill_tokens (16384 by
# default). The stock 1024/rank only yields 4096 at EP=4 and the cutedsl arms
# refuse to start. 4096/rank * 4 = 16384 exactly covers it.
export SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-4096}

case $MODEL in
  pro)   CKPT_NVFP4=$ROOT/ckpt/deepseek-v4-pro-nvfp4; CKPT_BASE=$ROOT/ckpt/deepseek-v4-pro ;;
  flash) CKPT_NVFP4=$ROOT/ckpt/deepseek-v4-flash-nvfp4; CKPT_BASE=$ROOT/ckpt/deepseek-v4-flash ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac

# arm -> "runner a2a ckpt_kind extra_env"
arm_spec () {
  case $1 in
    sg_split_cutedsl_fia2a)  echo "flashinfer_cutedsl flashinfer nvfp4 -" ;;
    sg_split_cutedsl_nixl)   echo "flashinfer_cutedsl nixl nvfp4 -" ;;
    sg_split_cutedsl_deepep) echo "flashinfer_cutedsl deepep nvfp4 -" ;;
    sg_trtllm_routed_fia2a)  echo "flashinfer_trtllm_routed flashinfer nvfp4 -" ;;
    sg_trtllm_routed_nixl)   echo "flashinfer_trtllm_routed nixl nvfp4 -" ;;
    sg_trtllm_routed_deepep) echo "flashinfer_trtllm_routed deepep nvfp4 -" ;;
    sg_megamoe)              echo "flashinfer_megamoe flashinfer_megamoe nvfp4 -" ;;
    sg_megamoe_ikr)          echo "flashinfer_megamoe flashinfer_megamoe nvfp4 SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE=1" ;;
    sg_megamoe_cmb_nvfp4)    echo "flashinfer_megamoe flashinfer_megamoe nvfp4 SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE=nvfp4" ;;
    sg_megamoe_cmb_mxfp8)    echo "flashinfer_megamoe flashinfer_megamoe nvfp4 SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE=mxfp8" ;;
    *) echo "" ;;
  esac
}

port_state () {
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", int(sys.argv[1]))); print("FREE")
except OSError:
    print("BUSY")
finally:
    s.close()
PY
}

cleanup_sg () {
  for p in '[s]glang.launch_server' '[s]glang::' '[S]cheduler' '[D]etokenizer' '[T]okenizerManager'; do
    pkill -9 -f "$p" 2>/dev/null
  done
  sleep 10
  for i in $(seq 1 60); do
    [ "$(port_state $PORT)" = "FREE" ] && return 0
    sleep 5
  done
  echo "  ERROR: port $PORT still BUSY"; return 1
}

run_arm () {
  local ARM=$1
  local spec; spec=$(arm_spec "$ARM")
  [ -z "$spec" ] && { echo "!! unknown arm $ARM"; return 2; }
  read -r RUNNER A2A CKIND EXTRA <<< "$spec"
  local MPATH; [ "$CKIND" = nvfp4 ] && MPATH=$CKPT_NVFP4 || MPATH=$CKPT_BASE
  local TAG="sg_${MODEL}_${ARM}_ep${EP}_conc${CONC}"

  echo
  echo "########## ARM=$ARM model=$MODEL runner=$RUNNER a2a=$A2A extra=$EXTRA"
  if [ ! -f "$MPATH/config.json" ]; then
    echo "  SKIP $ARM: checkpoint not staged at $MPATH"; return 0
  fi
  if [ "$(port_state $PORT)" != "FREE" ]; then
    echo "  FATAL: port busy before arm -- refusing to measure"; return 9
  fi

  # Per-arm env knob (megamoe variants only).
  if [ "$EXTRA" != "-" ]; then export "${EXTRA?}"; fi

  python3 -m sglang.launch_server --model-path "$MPATH" \
      --tp-size $EP --dp-size $EP --enable-dp-attention --ep-size $EP \
      --moe-runner-backend "$RUNNER" --moe-a2a-backend "$A2A" \
      --trust-remote-code --host 0.0.0.0 --port $PORT \
      --mem-fraction-static ${MEM_FRAC:-0.9} \
      --context-length $((ISL+OSL+256)) \
      > "$OUT/${TAG}.server.log" 2>&1 &
  local HPID=$! ready=0
  for i in $(seq 1 360); do
    sleep 10
    kill -0 $HPID 2>/dev/null || { echo "  server exited early"; break; }
    curl -sf "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null && { ready=1; echo "  ready after $((i*10))s"; break; }
  done

  local rc=0
  if [ "$ready" = 1 ]; then
    # --model/--tokenizer must be the LOCAL checkpoint path: the compute nodes
    # run with HF_HUB_OFFLINE=1 and no egress, so letting bench_serving resolve
    # the tokenizer from the Hub fails with LocalEntryNotFoundError.
    python3 -m sglang.bench_serving --backend sglang \
        --host 127.0.0.1 --port $PORT \
        --model "$MPATH" --tokenizer "$MPATH" \
        --dataset-name random --random-input-len $ISL --random-output-len $OSL \
        --random-range-ratio 0.8 --num-prompts $NPROMPTS --max-concurrency $CONC \
        --output-file "$OUT/${TAG}.jsonl" > "$OUT/${TAG}.bench.log" 2>&1 || rc=1
    # Keep the whole client log: piping through `tail` truncates the traceback
    # that explains a non-zero exit, which cost a debugging cycle already.
    tail -25 "$OUT/${TAG}.bench.log"
  else
    echo "  ARM $ARM NEVER READY"; tail -30 "$OUT/${TAG}.server.log"; rc=7
  fi

  kill -TERM $HPID 2>/dev/null; sleep 15
  cleanup_sg
  [ "$EXTRA" != "-" ] && unset "${EXTRA%%=*}"
  echo "--- ARM $ARM rc=$rc ---"
  return $rc
}

echo "=== preflight ==="
python3 - <<'PY'
import torch, sglang
from sglang.srt.arg_groups import choices as C
print("sglang", sglang.__version__, "| cc", torch.cuda.get_device_capability(),
      "| gpus", torch.cuda.device_count())
for name in dir(C):
    if "MOE" in name and "CHOICES" in name:
        print(" ", name, "=", getattr(C, name))
PY
[ $? -eq 0 ] || { echo "PREFLIGHT FAILED"; exit 3; }

cleanup_sg
mkdir -p "$OUT"
FAIL=0
for a in ${ARMS:?}; do run_arm "$a" || FAIL=1; done
echo "=== sglang matrix done, FAIL=$FAIL ==="
exit $FAIL
