#!/bin/bash
#SBATCH --job-name=KcollectLVL
#SBATCH --output=klog/collect_%A_%a.out
#SBATCH --error=klog/collect_%A_%a.err
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --constraint=rtx_6000
#SBATCH --gres=gpu:4
#SBATCH --array=0-24
#SBATCH --mail-user=bhargvb1@umbc.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu-general

echo "SLURM job started at $(date)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

module load Anaconda3

conda init
eval "$(conda shell.bash hook)"
conda activate miniConda

# Install dependencies once per job
python -m pip install --no-user --no-cache-dir -r ../../requirement.txt

export TMPDIR=/umbc/rs/ryus/users/bhargvb1/conda_envs/miniConda
export HF_HOME=$TMPDIR

# Set up Ollama on a unique port per task
PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_TMPDIR=/nfs/rs/ryus/users/bhargvb1/nsga_exp/testOllama/ollama_temp
export OLLAMA_HOST="0.0.0.0:${PORT}"

echo "Starting Ollama server on ${OLLAMA_HOST}..."
ollama serve &

# Give it a moment to start
sleep 10

# Pre-pull your models
# for M in llama3-chatqa:8b qwen2.5:14b mistral:7b gemma2:27b phi4:14b mistral:instruct; do
#   ollama pull $M
# done

ollama pull llama3-chatqa:8b
ollama pull qwen2.5:14b
ollama pull mistral:7b
ollama pull gemma2:27b
ollama pull phi4:14b
ollama pull gemma3:27b

# Build list of subdirectories (exclude hidden and current dir)
mapfile -t DIRS < <(find . -maxdepth 1 -type d ! -name '.' ! -name 'clog' ! -name 'results_all' ! -name 'updated_final_metrics_l4_with_tok' ! -name 'nsga_results' | sort)

# Guard: make sure the array size matches
if (( SLURM_ARRAY_TASK_ID >= ${#DIRS[@]} )); then
  echo "No directory for task $SLURM_ARRAY_TASK_ID; exiting."
  exit 0
fi

WORKDIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"
INPUT_CSV="$WORKDIR/merged_output.csv"
OUTPUT_CSV="$WORKDIR/collected_results.csv"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "No merged_output.csv in '$WORKDIR'; skipping."
  exit 0
fi

echo "Processing directory: $WORKDIR"
echo "Input CSV: $INPUT_CSV"
echo "Output CSV: $OUTPUT_CSV"

# Run your collection script (checkpoint every 100 rows)
python3 collection_with_checkpoint.py \
  -i "$INPUT_CSV" \
  -o "$OUTPUT_CSV" \
  --checkpoint-every 100



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



# Tear down Ollama
echo "Shutting down Ollama on port ${PORT}..."
lsof -ti tcp:${PORT} | xargs -r kill -9

echo "SLURM job completed at $(date)"

