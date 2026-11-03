#!/bin/bash
#SBATCH --job-name=rss-hanson-baseline
#SBATCH --account=ehpc-dev-2026d08-170-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/hanson-baseline-20261103/robosimstudio/outputs/hanson_baseline_20261103/slurm-%j.log
set -euo pipefail
source /nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/newton/outputs/stage1_setup/activate.sh
cd /nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/hanson-baseline-20261103/robosimstudio
export PYTHONPATH="/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/hanson-baseline-20261103/robosimstudio:/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/hanson-baseline-20261103/newton"
python -u outputs/hanson_baseline_20261103/run_baseline.py
