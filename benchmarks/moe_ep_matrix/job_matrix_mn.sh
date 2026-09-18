#!/bin/bash
# SLURM submitter for the two-node EP=8 MoE-EP matrix.
#   ./job_matrix_mn.sh <model:pro|flash> <partition> <time> "<arms>"
# Example:
#   ./job_matrix_mn.sh pro gb300 04:00:00 "mega_fi_cutedsl split_trtllm_fia2a"
set -euo pipefail
MODEL=${1:?model}
PART=${2:?partition}
TLIM=${3:-04:00:00}
ARMS=${4:?arms}

ROOT=/lustre/fsw/coreai_libraries_cudnn/agopal/dsv4ab
IMG=$ROOT/img/vllm-0.29.0-arm64.sqsh
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$ROOT/results/mnmatrix_${MODEL}_${STAMP}
mkdir -p "$OUT" "$ROOT/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH -A coreai_libraries_cudnn
#SBATCH -p ${PART}
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=${TLIM}
#SBATCH --job-name=coreai_libraries_cudnn-flashinfer.mnmoeep_${MODEL}
#SBATCH --output=${ROOT}/logs/mnmatrix_${MODEL}_${STAMP}.%j.out
set -uo pipefail
ROOT=${ROOT}
OUT=${OUT}
WS=\$ROOT/ws/\${SLURM_JOB_ID}
mkdir -p "\$OUT" "\$WS"
HEAD=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -1)
HEAD_IP=\$(getent ahostsv4 "\$HEAD" | awk '{print \$1; exit}')
export ROOT OUT HEAD_IP
export MODEL=${MODEL} ARMS="${ARMS}"
export CONC=\${CONC:-256} MNBT=\${MNBT:-2048} NPROMPTS=\${NPROMPTS:-512}
echo "###### \$(date -Is) EP8 nodes=\$SLURM_JOB_NODELIST head=\$HEAD_IP arms=\$ARMS"

srun --ntasks=2 --ntasks-per-node=1 --container-image=${IMG} \
     --container-mounts="\$ROOT:\$ROOT,\$WS:/workspace" \
     --container-workdir="\$ROOT/InferenceX" \
     --no-container-entrypoint --export=ALL \
     bash "\$ROOT/run_matrix_mn.sh"
echo "###### EP8 rc=\$?  results in \$OUT"
EOF
echo "OUT=$OUT"
