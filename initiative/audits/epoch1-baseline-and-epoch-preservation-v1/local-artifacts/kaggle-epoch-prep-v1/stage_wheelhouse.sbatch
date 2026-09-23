#!/bin/bash
#SBATCH --job-name=rsna-kaggle-wheelhouse-stage
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3.12 /home/sa305415/rsna-knee/kaggle/preparation-stage-v1/stage_wheelhouse.py
