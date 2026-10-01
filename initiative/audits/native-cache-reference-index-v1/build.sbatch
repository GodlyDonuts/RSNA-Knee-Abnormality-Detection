#!/bin/bash
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=rsna-reference-index-v1
#SBATCH --output=/home/sa305415/rsna-knee/native-cache-reference-index-v1/%j.out
#SBATCH --error=/home/sa305415/rsna-knee/native-cache-reference-index-v1/%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3.12 /home/sa305415/rsna-knee/native-cache-reference-index-v1/native_cache_reference_index.py --archive /home/sa305415/rsna-knee/cache/cache256-v3-official-recovery-v1.tar --manifest /home/sa305415/rsna-knee/native-cache-reference-index-v1/manifest.json --output /home/sa305415/rsna-knee/native-cache-reference-index-v1/reference-index.json
