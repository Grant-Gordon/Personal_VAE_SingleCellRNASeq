#!/bin/bash
#SBATCH --job-name=Train_DAE
#SBATCH --output=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs/%x_%j.out
#SBATCH --error=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=bigmem
#SBATCH --mem=16G

WORKING_DIR="/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq"

cd "$WORKING_DIR" || exit 1
source personalVAEenv/bin/activate

export PYTHONPATH="$WORKING_DIR:$PYTHONPATH"
python3 -m DAE.train_DAE > DAE/train_DAE.log 2>&1



