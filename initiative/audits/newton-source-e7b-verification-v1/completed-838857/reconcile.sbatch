#!/bin/bash
#SBATCH --job-name=rsna-source-reconcile-e7b
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home/sa305415/rsna-knee/code-staging-control/7aa4978f141e4832832f95967f456f1d/batch.stdout
#SBATCH --error=/home/sa305415/rsna-knee/code-staging-control/7aa4978f141e4832832f95967f456f1d/batch.stderr
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3.12 /home/sa305415/rsna-knee/code-staging-control/7aa4978f141e4832832f95967f456f1d/reconcile.py
