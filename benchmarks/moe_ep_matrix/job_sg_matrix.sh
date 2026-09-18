#!/bin/bash
# SLURM submitter for the single-node EP4 SGLang MoE-EP matrix.
#   ./job_sg_matrix.sh <model:pro|flash> <partition> <time> "<arms>"
# Example:
#   ./job_sg_matrix.sh flash gb200 04:00:00 "sg_megamoe sg_split_cutedsl_fia2a"
set -euo pipefail
MODEL=${1:?model}
PART=${2:?partition}
TLIM=${3:-04:00:00}
ARMS=${4:?arms}

ROOT=/lustre/fsw/coreai_libraries_cudnn/agopal/dsv4ab
IMG=$ROOT/img/sglang-nightly-20260918-arm64.sqsh
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$ROOT/results/sgmatrix_${MODEL}_${STAMP}
mkdir -p "$OUT" "$ROOT/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH -A coreai_libraries_cudnn
#SBATCH -p ${PART}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=${TLIM}
#SBATCH --job-name=coreai_libraries_cudnn-flashinfer.sgmoeep_${MODEL}
#SBATCH --output=${ROOT}/logs/sgmatrix_${MODEL}_${STAMP}.%j.out
set -x
srun --container-image=${IMG} \
     --container-mounts=${ROOT}:${ROOT} \
     --container-workdir=${ROOT} \
     bash -lc 'ROOT=${ROOT} OUT=${OUT} MODEL=${MODEL} ARMS="${ARMS}" \
        CONC=\${CONC:-256} NPROMPTS=\${NPROMPTS:-512} \
        bash ${ROOT}/run_sg_matrix.sh'
echo "RESULTS: ${OUT}"
EOF
echo "OUT=$OUT"
