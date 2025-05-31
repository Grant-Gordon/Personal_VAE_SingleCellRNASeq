#!/bin/bash
#SBATCH --job-name=100_epoch
#SBATCH --output=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/mheaded_c_vae/JobOutputs/Job_%j-%x/%x.out
#SBATCH --error=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/mheaded_c_vae/JobOutputs/Job_%j-%x/%x.err
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G

GIT_ROOT_DIR="/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq"
PROJECT_ROOT_DIR="${GIT_ROOT_DIR}/mheaded_c_vae"
DATA_DIR="/mnt/projects/debruinz_project/july2024_census_data/subset" #Use "july2024_census_data/subset" for 3_mil samples use "july2024_census_data/full" for full 16 mill census data
OUTPUT_DIR="${PROJECT_ROOT_DIR}/JobOutputs/Job_${SLURM_JOB_ID}-${SLURM_JOB_NAME}"
CONFIG_PATH="${PROJECT_ROOT_DIR}/configs/train_human_cvae.yaml"


cd "$GIT_ROOT_DIR" || exit 1
source personalVAEenv/bin/activate
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PROJECT_ROOT_DIR:$PYTHONPATH"
cd $PROJECT_ROOT_DIR ||exit 1

# Step 1: Build vocab (skips if already exists)
python3 ./scripts/vocab_builder.py --config $CONFIG_PATH training.output_dir=$OUTPUT_DIR

# Step 2: Train model
python3 ./training/train.py --config $CONFIG_PATH training.output_dir=$OUTPUT_DIR

cp $CONFIG_PATH $OUTPUT_DIR
