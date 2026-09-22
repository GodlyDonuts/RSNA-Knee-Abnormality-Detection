#!/bin/bash
#SBATCH --job-name=rsna-python-identities-v1
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --nodelist=evc22,evc37
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=/home/sa305415/rsna-knee/python-node-identity-v1/batch.stdout
#SBATCH --error=/home/sa305415/rsna-knee/python-node-identity-v1/batch.stderr
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 --gres=none /usr/bin/python3.12 /home/sa305415/rsna-knee/python-node-identity-v1/inspect.py
