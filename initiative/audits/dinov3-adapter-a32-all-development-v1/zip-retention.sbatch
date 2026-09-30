#!/bin/bash
#SBATCH --job-name=rsna-old-zip-storage
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
/usr/bin/python3.12 /home/sa305415/rsna-knee/all-development-storage-v1/retain_old_zips.py
lfs quota -u sa305415 /home/sa305415
