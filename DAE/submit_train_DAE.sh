#!/bin/bash
#SBATCH --job-name=Train_DAE_sub_updated_logs
#SBATCH --output=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs/Job_%j-%x/%x.out
#SBATCH --error=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs/Job_%j-%x/%x.err
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=2
#SBATCH --mem=32G

#Define environment variables 
PROJECT_ROOT_DIR="/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq"
DATA_DIR="/mnt/projects/debruinz_project/july2024_census_data/subset" #Use "july2024_census_data/subset" for 3_mil samples use "july2024_census_data/full" for full 16 mill census data
OUTPUT_DIR="${PROJECT_ROOT_DIR}/JobOutputs/Job_${SLURM_JOB_ID}-${SLURM_JOB_NAME}"



#Model Paramters 
EPOCHS=1
BATCH_SIZE=128
SPECIES="human"
CHUNKS_PRELOADED=1
NUM_PRELOADER_THREADS=1

#Set up environment 
cd "$PROJECT_ROOT_DIR" || exit 1
source personalVAEenv/bin/activate
mkdir -p $OUTPUT_DIR


#Train Model 
export PYTHONPATH="$PROJECT_ROOT_DIR:$PYTHONPATH"
python3 -m DAE.train_DAE --data_dir=$DATA_DIR --epochs=$EPOCHS --batch_size=$BATCH_SIZE --output_dir=$OUTPUT_DIR --species=$SPECIES --chunks_preloaded=$CHUNKS_PRELOADED --num_preloader_threads=$NUM_PRELOADER_THREADS

#clean up output 
#TODO mv dcgm-gpu-stats* $OUTPUT_DIR


