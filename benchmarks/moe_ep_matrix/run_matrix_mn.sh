#!/bin/bash
# Two-node EP=8 MoE-EP matrix driver (node-role script, run under srun -N2).
#   node 0: head vLLM (API server) + benchmark client
#   node 1: headless vLLM engines (DP ranks 4-7)
# Coordinated through sentinel files on Lustre.
#
# Why this exists: at EP=4 a single node holds the whole DeepSeek-V4-Pro NVFP4
# checkpoint (851 GB), leaving ~47k KV tokens per rank -- max concurrency 5x at
# 9.4k tok/req. Spreading the experts over 8 ranks frees most of that memory and
# lets the large-batch regime actually be measured.
#
# Reads: ROOT OUT MODEL ARMS CONC MNBT NPROMPTS ISL OSL HEAD_IP
set -uo pipefail
: "${ROOT:?}" "${OUT:?}" "${HEAD_IP:?}"
MODEL=${MODEL:-pro}
MNBT=${MNBT:-2048}
CONC=${CONC:-256}
NPROMPTS=${NPROMPTS:-512}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
PORT=8888
RPC=13345
EP=8
NODE=${SLURM_NODEID:-0}

export PYTHONPATH=$ROOT/pysite:${PYTHONPATH:-}
export HF_HOME=$ROOT/cache/huggingface HF_HUB_CACHE=$ROOT/cache/huggingface/hub HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer-ws
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export NVSHMEM_MAX_TEAMS=${NVSHMEM_MAX_TEAMS:-1024}

case $MODEL in
  pro)   CKPT_NVFP4=$ROOT/ckpt/deepseek-v4-pro-nvfp4; CKPT_BASE=$ROOT/ckpt/deepseek-v4-pro ;;
  flash) CKPT_NVFP4=$ROOT/ckpt/deepseek-v4-flash-nvfp4; CKPT_BASE=$ROOT/ckpt/deepseek-v4-flash ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac

case $MNBT in
  2048) SIZES='[1,2,4,8,16,32,64,128,256,512,1024,2048]';;
  4096) SIZES='[1,2,4,8,16,32,64,128,256,512,1024,2048,4096]';;
  8192) SIZES='[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]';;
  *)    SIZES='[1,2,4,8,16,32,64,128,256,512,1024,2048]';;
esac

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

SENT=$OUT/.sentinel; mkdir -p "$SENT"

run_arm () {
  local ARM=$1
  local spec; spec=$(arm_spec "$ARM")
  [ -z "$spec" ] && { echo "!! unknown arm $ARM"; return 2; }
  read -r BACKEND A2A CKIND <<< "$spec"
  local MPATH; [ "$CKIND" = nvfp4 ] && MPATH=$CKPT_NVFP4 || MPATH=$CKPT_BASE
  local TAG="${MODEL}_${ARM}_ep${EP}_conc${CONC}_mnbt${MNBT}"

  if [ ! -f "$MPATH/config.json" ]; then
    echo "[node $NODE] SKIP $ARM: checkpoint not staged at $MPATH"
    touch "$SENT/done_$ARM"
    return 0
  fi

  local A2A_ARGS=()
  [ "$A2A" != "-" ] && A2A_ARGS=(--all2all-backend "$A2A")

  local COMMON=(--served-model-name "$MPATH"
    --data-parallel-size $EP --data-parallel-size-local 4
    --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port $RPC
    --tensor-parallel-size 1 --enable-expert-parallel
    --moe-backend "$BACKEND" "${A2A_ARGS[@]}"
    --kv-cache-dtype fp8 --block-size 256 --no-enable-prefix-caching --trust-remote-code
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice
    --reasoning-parser deepseek_v4 --attention_config.use_fp4_indexer_cache True
    --max-model-len $((ISL+OSL+256)) --max-num-batched-tokens $MNBT
    --max-cudagraph-capture-size $MNBT
    --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":$SIZES}"
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.95})

  if [ "$NODE" = 0 ]; then
    echo
    echo "########## ARM=$ARM model=$MODEL backend=$BACKEND all2all=$A2A ep=$EP"
    vllm serve "$MPATH" "${COMMON[@]}" --host 0.0.0.0 --port $PORT \
        > "$OUT/${TAG}.head.log" 2>&1 &
    local HPID=$! ready=0
    for i in $(seq 1 360); do
      sleep 10
      kill -0 $HPID 2>/dev/null || { echo "  head exited early"; break; }
      curl -sf "http://127.0.0.1:$PORT/v1/models" -o /dev/null 2>/dev/null && { ready=1; echo "  ready after $((i*10))s"; break; }
    done
    if [ "$ready" = 1 ]; then
      grep -aE "Available KV cache memory|GPU KV cache size" "$OUT/${TAG}.head.log" | tail -2
      python3 "$ROOT/InferenceX/utils/bench_serving/benchmark_serving.py" \
        --model "$MPATH" --backend vllm --base-url "http://127.0.0.1:$PORT" \
        --dataset-name random --random-input-len $ISL --random-output-len $OSL \
        --random-range-ratio 0.8 --num-prompts $NPROMPTS --max-concurrency $CONC \
        --request-rate inf --ignore-eos --num-warmups $((CONC*2)) \
        --percentile-metrics 'ttft,tpot,itl,e2el' --save-result \
        --result-dir "$OUT" --result-filename "${TAG}.json" \
        --trust-remote-code 2>&1 | tail -18
      local n
      n=$(python3 -c "import json;print(json.load(open('$OUT/${TAG}.json')).get('completed',0))" 2>/dev/null || echo 0)
      echo "  completed requests: $n"
      [ "${n:-0}" -lt 1 ] && echo "  ARM $ARM INVALID (0 completed)"
    else
      echo "  ARM $ARM NEVER READY"; tail -30 "$OUT/${TAG}.head.log"
    fi
    kill -TERM $HPID 2>/dev/null; sleep 20
    touch "$SENT/done_$ARM"
  else
    echo "[worker] ARM=$ARM headless ranks 4-7"
    vllm serve "$MPATH" "${COMMON[@]}" --headless --data-parallel-start-rank 4 \
        > "$OUT/${TAG}.worker.log" 2>&1 &
    local WPID=$!
    for i in $(seq 1 420); do
      [ -f "$SENT/done_$ARM" ] && { echo "[worker] head done for $ARM"; break; }
      kill -0 $WPID 2>/dev/null || { echo "[worker] engines exited early"; tail -15 "$OUT/${TAG}.worker.log"; break; }
      sleep 10
    done
    kill -TERM $WPID 2>/dev/null; sleep 15
  fi

  for p in '[v]llm serve' '[V]LLM::' '[E]ngineCore' '[A]piServer' '[V]LLM_DP_Coordinator' '[d]eepseek-v4'; do
    pkill -9 -f "$p" 2>/dev/null
  done
  sleep 30
  echo "--- ARM $ARM done (node $NODE) ---"
}

for a in ${ARMS:?}; do run_arm "$a"; done

for p in '[v]llm serve' '[V]LLM::' '[E]ngineCore' '[A]piServer' '[d]eepseek-v4'; do pkill -9 -f "$p" 2>/dev/null; done
echo "[node $NODE] matrix done"
