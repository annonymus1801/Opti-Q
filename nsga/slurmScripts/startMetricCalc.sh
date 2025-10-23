#!/bin/bash
#SBATCH --job-name=Metrics_QoA
#SBATCH --output=logs_SQA/metrics_%A_%a.out
#SBATCH --error=logs_SQA/metrics_%A_%a.err
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --constraint=rtx_6000
#SBATCH --gres=gpu:1
#SBATCH --mail-user=bhargvb1@umbc.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=1
#SBATCH --partition=gpu-general



echo "Job start: $(date)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

module load Anaconda3
conda init
eval "$(conda shell.bash hook)"
conda activate miniConda

pip install -r ../requirement.txt

# Find all subdirectories (excluding logs_SQA)
mapfile -t DIRS < <(find . -maxdepth 1 -type d ! -name '.' ! -name 'logs_SQA' | sort)

if (( SLURM_ARRAY_TASK_ID >= ${#DIRS[@]} )); then
  echo "No directory for task $SLURM_ARRAY_TASK_ID; exiting."
  exit 0
fi

WORKDIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"
COLLECT_CSV="$WORKDIR/collected_results.csv"
METRICS_CSV="$WORKDIR/metrics.csv"
QoA_CSV="$WORKDIR/qoa.csv"

if [[ ! -f "$COLLECT_CSV" ]]; then
  echo "Missing $COLLECT_CSV; skipping."
  exit 0
fi

echo "Processing: $WORKDIR"
echo "  Collected: $COLLECT_CSV"
echo "  Metrics out: $METRICS_CSV"
echo "  QoA out:     $QoA_CSV"

# 1) Compute metrics
python3 metrics.py \
  --input "$COLLECT_CSV" \
  --output "$METRICS_CSV"

echo "Doing QoA now"

# 2) Compute QoA
python3 qoa.py \
  -i "$METRICS_CSV" \
  -o "$QoA_CSV"

echo "Task completed: $(date)"

