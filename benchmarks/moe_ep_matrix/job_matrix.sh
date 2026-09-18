#!/bin/bash
# SLURM submitter for the single-node EP4 MoE-EP matrix.
#   ./job_matrix.sh <model:pro|flash> <partition> <time> "<arms>" [mode]
# Example:
#   ./job_matrix.sh pro gb300 04:00:00 "mega_fi_cutedsl" perf
set -euo pipefail
MODEL=${1:?model}
PART=${2:?partition}
TLIM=${3:-04:00:00}
ARMS=${4:?arms}
MODE=${5:-perf}

ROOT=/lustre/fsw/coreai_libraries_cudnn/agopal/dsv4ab
IMG=$ROOT/img/vllm-0.29.0-arm64.sqsh
# Overridable so a corrected driver can be rolled out while an older job is
# still mid-execution of the previous copy (bash reads scripts incrementally,
# so overwriting a running driver in place can corrupt it).
DRIVER=${DRIVER:-run_matrix.sh}
# $$ disambiguates two submissions inside the same second, which otherwise
# share an output directory.
STAMP=$(date +%Y%m%d-%H%M%S)-$$
OUT=$ROOT/results/matrix_${MODEL}_${STAMP}
mkdir -p "$OUT" "$ROOT/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH -A coreai_libraries_cudnn
#SBATCH -p ${PART}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=${TLIM}
#SBATCH --job-name=coreai_libraries_cudnn-flashinfer.moeep_${MODEL}
#SBATCH --output=${ROOT}/logs/matrix_${MODEL}_${STAMP}.%j.out
set -x
srun --container-image=${IMG} \
     --container-mounts=${ROOT}:${ROOT} \
     --container-workdir=${ROOT}/InferenceX \
     bash -lc 'ROOT=${ROOT} OUT=${OUT} MODEL=${MODEL} ARMS="${ARMS}" MODE=${MODE} \
        CONC=\${CONC:-256} MNBT=\${MNBT:-2048} NPROMPTS=\${NPROMPTS:-512} \
        EAGER=\${EAGER:-0} GPU_MEM_UTIL=\${GPU_MEM_UTIL:-0.92} ACC_N=\${ACC_N:-200} \
        ISL=\${ISL:-8192} OSL=\${OSL:-1024} BATCH=\${BATCH:-} \
        bash ${ROOT}/${DRIVER}'
echo "RESULTS: ${OUT}"
EOF
echo "OUT=$OUT"
