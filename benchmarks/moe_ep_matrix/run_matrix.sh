#!/bin/bash
# In-container single-node EP4 MoE-EP matrix driver (DeepSeek-V4 Pro / Flash).
# One vLLM server per arm; arms differ only in --moe-backend / --all2all-backend.
#
# Reads: ROOT OUT MODEL ARMS CONC MNBT NPROMPTS ISL OSL MODE EP
set -uo pipefail
: "${ROOT:?}" "${OUT:?}"
MODEL=${MODEL:-pro}
MNBT=${MNBT:-2048}
CONC=${CONC:-256}
NPROMPTS=${NPROMPTS:-512}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
MODE=${MODE:-perf}          # perf | acc | both
PORT=8888
EP=${EP:-4}

export PYTHONPATH=$ROOT/pysite:${PYTHONPATH:-}
export HF_HOME=$ROOT/cache/huggingface HF_HUB_CACHE=$ROOT/cache/huggingface/hub HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer-ws
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export NVSHMEM_MAX_TEAMS=${NVSHMEM_MAX_TEAMS:-1024}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/ix-pycache

case $MODEL in
  pro)   CKPT_NVFP4=$ROOT/ckpt/deepseek-v4-pro-nvfp4; CKPT_BASE=$ROOT/ckpt/deepseek-v4-pro ;;
  flash) CKPT_NVFP4=$ROOT/ckpt/deepseek-v4-flash-nvfp4; CKPT_BASE=$ROOT/ckpt/deepseek-v4-flash-nvfp4 ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac

case $MNBT in
  2048) SIZES='[1,2,4,8,16,32,64,128,256,512,1024,2048]';;
  4096) SIZES='[1,2,4,8,16,32,64,128,256,512,1024,2048,4096]';;
  *)    SIZES='[1,2,4,8,16,32,64,128,256,512,1024,2048]';;
esac

# arm name -> "moe_backend all2all ckpt_kind"
arm_spec () {
  case $1 in
    split_cutedsl_fia2a)  echo "flashinfer_cutedsl flashinfer_all2allv nvfp4" ;;
    split_cutedsl_nixl)   echo "flashinfer_cutedsl nixl_ep nvfp4" ;;
    split_cutedsl_deepep) echo "flashinfer_cutedsl deepep_low_latency nvfp4" ;;
    split_trtllm_fia2a)   echo "flashinfer_trtllm flashinfer_all2allv nvfp4" ;;
    split_trtllm_nixl)    echo "flashinfer_trtllm nixl_ep nvfp4" ;;
    split_trtllm_deepep)  echo "flashinfer_trtllm deepep_low_latency nvfp4" ;;
    mega_fi_cutedsl)      echo "flashinfer_moe_ep_mega_cutedsl - nvfp4" ;;
    mega_fi_deepgemm)     echo "flashinfer_moe_ep_mega_deep_gemm - base" ;;
    mega_deepep_native)   echo "deep_gemm_mega_moe deepep_low_latency base" ;;
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

cleanup_vllm () {
  for p in '[v]llm serve' '[V]LLM::' '[E]ngineCore' '[A]piServer' '[V]LLM_DP_Coordinator' '[d]eepseek-v4'; do
    pkill -9 -f "$p" 2>/dev/null
  done
  sleep 15
  for i in $(seq 1 60); do
    [ "$(port_state $PORT)" = "FREE" ] && return 0
    pkill -9 -f '[A]piServer' 2>/dev/null; sleep 5
  done
  echo "  ERROR: port $PORT still BUSY"; return 1
}

run_arm () {
  local ARM=$1
  local spec; spec=$(arm_spec "$ARM")
  [ -z "$spec" ] && { echo "!! unknown arm $ARM"; return 2; }
  read -r BACKEND A2A CKIND <<< "$spec"
  local MPATH; [ "$CKIND" = nvfp4 ] && MPATH=$CKPT_NVFP4 || MPATH=$CKPT_BASE
  local TAG="${MODEL}_${ARM}_ep${EP}_conc${CONC}_mnbt${MNBT}"

  echo
  echo "########## ARM=$ARM model=$MODEL backend=$BACKEND all2all=${A2A} ckpt=$(basename $MPATH)"
  if [ "$(port_state $PORT)" != "FREE" ]; then
    echo "  FATAL: port busy before arm -- refusing to measure"; return 9
  fi

  local A2A_ARGS=()
  [ "$A2A" != "-" ] && A2A_ARGS=(--all2all-backend "$A2A")

  local COMMON=(--served-model-name "$MPATH"
    --data-parallel-size $EP --tensor-parallel-size 1 --enable-expert-parallel
    --moe-backend "$BACKEND" "${A2A_ARGS[@]}"
    --kv-cache-dtype fp8 --block-size 256 --no-enable-prefix-caching --trust-remote-code
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice
    --reasoning-parser deepseek_v4 --attention_config.use_fp4_indexer_cache True
    --max-model-len $((ISL+OSL+256)) --max-num-batched-tokens $MNBT
    --max-cudagraph-capture-size $MNBT
    --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":$SIZES}"
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.92})

  vllm serve "$MPATH" "${COMMON[@]}" --host 0.0.0.0 --port $PORT \
      > "$OUT/${TAG}.server.log" 2>&1 &
  local HPID=$! ready=0
  for i in $(seq 1 360); do
    sleep 10
    kill -0 $HPID 2>/dev/null || { echo "  server exited early"; break; }
    curl -sf "http://127.0.0.1:$PORT/v1/models" -o /tmp/m.json 2>/dev/null && { ready=1; echo "  ready after $((i*10))s"; break; }
  done

  local rc=0
  if [ "$ready" = 1 ]; then
    grep -aE "Available KV cache memory|GPU KV cache size" "$OUT/${TAG}.server.log" | tail -2
    if [ "$MODE" = perf ] || [ "$MODE" = both ]; then
      python3 "$ROOT/InferenceX/utils/bench_serving/benchmark_serving.py" \
        --model "$MPATH" --backend vllm --base-url "http://127.0.0.1:$PORT" \
        --dataset-name random --random-input-len $ISL --random-output-len $OSL \
        --random-range-ratio 0.8 --num-prompts $NPROMPTS --max-concurrency $CONC \
        --request-rate inf --ignore-eos --num-warmups $((CONC*2)) \
        --percentile-metrics 'ttft,tpot,itl,e2el' --save-result \
        --result-dir "$OUT" --result-filename "${TAG}.json" \
        --trust-remote-code 2>&1 | tail -18 || rc=1
    fi
    if [ "$MODE" = acc ] || [ "$MODE" = both ]; then
      python3 "$ROOT/eval_gsm8k.py" --parquet "$ROOT/gsm8k_test.jsonl" \
        --url "http://127.0.0.1:$PORT/v1/chat/completions" \
        --model "$MPATH" --n ${ACC_N:-200} --tag "$ARM" \
        --out "$OUT/${TAG}.gsm8k.json" \
        > "$OUT/${TAG}.gsm8k.txt" 2>&1 || rc=1
      tail -3 "$OUT/${TAG}.gsm8k.txt"
    fi
  else
    echo "  ARM $ARM NEVER READY"; tail -30 "$OUT/${TAG}.server.log"; rc=7
  fi

  kill -TERM $HPID 2>/dev/null; sleep 20
  cleanup_vllm

  if [ "$MODE" != acc ] && [ -f "$OUT/${TAG}.json" ]; then
    local n; n=$(python3 -c "import json;print(json.load(open('$OUT/${TAG}.json')).get('completed',0))" 2>/dev/null || echo 0)
    echo "  completed requests: $n"
    [ "${n:-0}" -lt 1 ] && { echo "  ARM $ARM INVALID (0 completed)"; rc=8; }
  fi
  echo "--- ARM $ARM rc=$rc ---"
  return $rc
}

echo "=== preflight ==="
python3 - <<'PY'
import torch, vllm, flashinfer
from vllm.config.kernel import MoEBackend, MEGA_MOE_BACKENDS
from vllm.config.parallel import All2AllBackend
import typing
print("vllm", vllm.__version__, "| flashinfer", flashinfer.__version__,
      "| cc", torch.cuda.get_device_capability(), "| gpus", torch.cuda.device_count())
print("moe backends ok:", {"flashinfer_cutedsl","flashinfer_trtllm",
      "flashinfer_moe_ep_mega_cutedsl","flashinfer_moe_ep_mega_deep_gemm",
      "deep_gemm_mega_moe"} <= set(typing.get_args(MoEBackend)))
print("a2a backends ok:", {"flashinfer_all2allv","nixl_ep","deepep_low_latency"}
      <= set(typing.get_args(All2AllBackend)))
PY
[ $? -eq 0 ] || { echo "PREFLIGHT FAILED"; exit 3; }

cleanup_vllm
mkdir -p "$OUT"
FAIL=0
for a in ${ARMS:?}; do run_arm "$a" || FAIL=1; done
echo "=== matrix done, FAIL=$FAIL ==="
exit $FAIL
