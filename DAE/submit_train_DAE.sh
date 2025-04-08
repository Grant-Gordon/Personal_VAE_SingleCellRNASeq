#!/bin/bash
#SBATCH --job-name=Train_DAE
#SBATCH --output=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs/%x_%j.out
#SBATCH --error=/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=2
#SBATCH --mem=32G

#Define environment variables 
PROJECT_ROOT_DIR="/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq"
DATA_DIR="/mnt/projects/debruinz_project/july2024_census_data/subset" #Use data/subset for 3_mil samples use data/full for full 16 mill census data
OUTPUTS_DIR="/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/JobOutputs"
RUN_OUTPUT_DIR="${OUTPUTS_DIR}/job_${SLURM_JOB_ID}-${SLURM_JOB_NAME}"
mkdir -p $RUN_OUTPUT_DIR



#Model Paramters 
EPOCHS=1
BATCH_SIZE=128
SPECIES="human"
CHUNKS_PRELOADED=2
NUM_PRELOADER_THREADS=2

#Set up environment 
cd "$WORKING_DIR" || exit 1
source personalVAEenv/bin/activate

#Train Model 
export PYTHONPATH="$WORKING_DIR:$PYTHONPATH"
python3 -m DAE.train_DAE --data_dir=$DATA_DIR --epochs=$EPOCHS --batch_size=$BATCH_SIZE --output_dir=$OUTPUT_DIR --species=$SPECIES --chunks_preloaded=$CHUNKS_PRELOADED --num_preloader_threads=$NUM_PRELOADER_THREADS
#clean up output 
mv *.log $RUN_OUTPUT_DIR
mv *.out $RUN_OUTPUT_DIR
mv *.csv $RUN_OUTPUT_DIR
mv *.err $RUN_OUTPUT_DIR
mv tensorboard_logs $RUN_OUTPUT_DIR


