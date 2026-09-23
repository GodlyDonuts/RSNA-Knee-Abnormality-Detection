#!/bin/bash
#SBATCH --job-name=rsna-hf-publisher-env
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=evc42
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
/usr/bin/python3.12 /home/sa305415/rsna-knee/hf-epoch-publisher-v1/build_env.py
