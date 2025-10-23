#!/bin/bash
#
#SBATCH --cluster=chip-cpu
#SBATCH --job-name=impL
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --array=0-10
#SBATCH --output=nlogs/nsga_%A_%a.out
#SBATCH --error=nlogs/nsga_%A_%a.err
#SBATCH --mail-user=bhargvb1@umbc.edu
#SBATCH --mail-type=END,FAIL,ALL

module load Anaconda3

conda init
eval "$(conda shell.bash hook)"
conda activate miniConda

# conda activate miniConda
pip install -r ../../requirement.txt
pip install Levenshtein

CONFIG_FILES=(missing_configs/*.json)
CFG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Task $SLURM_ARRAY_TASK_ID using $CFG"
python gnsaga.py --config "$CFG" 
echo "[$(date)] Done task $SLURM_ARRAY_TASK_ID"
