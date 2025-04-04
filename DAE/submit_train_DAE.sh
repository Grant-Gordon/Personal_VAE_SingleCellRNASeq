#!/bin/bash
#SBATCH --job-name=Train_DAE
#SBATCH --output=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs/%x_%j.out
#SBATCH --error=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G

WORKING_DIR="/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq"
DATA_DIR="/mnt/projects/debruinz_project/july2024_census_data/full"
OUTPUT_DIR="/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs"

EPOCHS=1
BATCH_SIZE=3
SPECIES="human"

cd "$WORKING_DIR" || exit 1
source personalVAEenv/bin/activate

export PYTHONPATH="$WORKING_DIR:$PYTHONPATH"
python3 -m DAE.train_DAE --data_dir=$DATA_DIR --epochs=$EPOCHS --batch_size=$BATCH_SIZE --output_dir=$OUTPUT_DIR --species=$SPECIES 



