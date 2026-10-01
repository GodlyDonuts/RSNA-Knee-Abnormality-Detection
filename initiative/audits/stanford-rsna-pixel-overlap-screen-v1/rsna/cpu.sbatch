#!/bin/bash
#SBATCH --job-name=rsna-private-rsna-pixel-index
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --nodelist=evc31
#SBATCH --time=02:00:00
#SBATCH --output=/home/sa305415/rsna-knee/private-rsna-pixel-index-20261001-v1/stdout.txt
#SBATCH --error=/home/sa305415/rsna-knee/private-rsna-pixel-index-20261001-v1/stderr.txt
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/nice -n 10 /usr/bin/ionice -c 3 /usr/bin/python3.12 -B /home/sa305415/rsna-knee/private-rsna-pixel-index-20261001-v1/launch_pixel_index.py --control /home/sa305415/rsna-knee/private-rsna-pixel-index-20261001-v1
