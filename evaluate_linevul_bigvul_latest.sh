#!/bin/bash
#SBATCH -p a100
#SBATCH --nodes 1
#SBATCH -c 20
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=1-1
#SBATCH --err="_logs_linevul_latest/bigvul_results_10_%a.err"
#SBATCH --output="_logs_linevul_latest/bigvul_results_10_%a.out"
#SBATCH --job-name="Big_Latest"

## Setup Python Environment
module purge
module use /apps/skl/modules/all
module load CUDA/11.2.0
module load cuDNN/CUDA-11.2/8.1.1.33
module load Anaconda3/2020.07
module load Java/1.8.0_191
module load Singularity/3.7.4
module load git/2.32.0

source activate myenv
conda deactivate
source activate myenv

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

## Read inputs
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p evaluate_linevul_clean.csv`
python3 -u evaluate_linevul_bigvul_latest.py "${par[0]}"