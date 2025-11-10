#!/bin/bash
#SBATCH --job-name=W2V2_PRETRAIN
#SBATCH --account=def-joelzy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=11:20:00
#SBATCH --output=logs_maps/%j-out.txt
#SBATCH --error=logs_maps/%j-error.txt
#SBATCH --partition=compute


# Random sleep to avoid simultaneous job start
sleep $((RANDOM % 60))

# Change to temporary SLURM directory
cd $SLURM_TMPDIR
mkdir -p $SLURM_TMPDIR/brain2vec

# Copy source files
cp /home/mengkris/links/scratch/wav2vec2_pret.py \
   /home/mengkris/links/scratch/old_config.py \
   /home/mengkris/links/scratch/modeling_outputs.py \
   $SLURM_TMPDIR/brain2vec/

cd $SLURM_TMPDIR/brain2vec

# =========================
#   Modules and Environment
# =========================
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load python/3.11
module load cuda/12.2
module load cudacore/.12.2.2
module load cudnn/8.9.5.29

# Set HuggingFace / matplotlib / cache directories
mkdir -p $SLURM_TMPDIR/huggingface_cache $SLURM_TMPDIR/matplotlib
export TRANSFORMERS_CACHE=$SLURM_TMPDIR/huggingface_cache
export HF_HOME=$SLURM_TMPDIR/huggingface_cache
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib


# Create and activate virtual environment
python -m venv $SLURM_TMPDIR/brain2vec
source $SLURM_TMPDIR/brain2vec/bin/activate

# Install packages
pip install --no-index --upgrade pip setuptools wheel
pip install --no-index torch torchvision torchaudio transformers==4.36 accelerate scikit-learn tqdm numpy matplotlib debugpy pandas pathlib

# Arguments
patient_id=$1
TRAIN_PATH="/home/mengkris/links/scratch/pd_data/all_but_one_patient_zscored_train_lfp/${1}.npy"
TRAIN_LABEL_PATH="/home/mengkris/links/scratch/pd_data/all_but_one_patient_zscored_train_label/${1}.npy"
TEST_PATH="/home/mengkris/links/scratch/pd_data/one_patient_zscored_val_lfp/${1}.npy"
TEST_LABEL_PATH="/home/mengkris/links/scratch/pd_data/one_patient_zscored_val_label/${1}.npy"
OUTPUT_PRE_DIR="/home/mengkris/links/scratch/pretrained_wav2vec2_${1}"
OUTPUT_DIR="/home/mengkris/links/scratch/finetuned_models/wav2vec2_${1}"
VERSION_TAG="wav2vec2_$1"

echo "TRAIN_PATH: $TRAIN_PATH"
echo "TRAIN_LABEL_PATH: $TRAIN_LABEL_PATH"
echo "TEST_PATH: $TEST_PATH"
echo "TEST_LABEL_PATH: $TEST_LABEL_PATH"
echo "OUTPUT_PRE_DIR: $OUTPUT_PRE_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "VERSION_TAG: $VERSION_TAG"
# =========================
#   Run Pretraining 200000  32000 500
# =========================
accelerate launch wav2vec2_pret.py \
  --model_name_or_path "/home/mengkris/links/scratch/patrick_vonplaten_wav2vec2_model" \
  --pretrained_output_dir $OUTPUT_PRE_DIR \
  --output_dir $OUTPUT_DIR \
  --train_path $TRAIN_PATH \
  --train_label_path $TRAIN_LABEL_PATH \
  --test_path $TEST_PATH \
  --test_label_path $TEST_LABEL_PATH \
  --vers $VERSION_TAG \
  --max_train_steps 200000 \
  --num_warmup_steps 32000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 0.0001 \
  --weight_decay 0.01 \
  --max_duration_in_seconds 30.0 \
  --min_duration_in_seconds 2.0 \
  --logging_steps 1 \
  --saving_steps 500 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-06 \
  --gradient_checkpointing True \
  --mask_time_prob 0.65 \
  --mask_time_length 10 \
  --num_train_epochs 3 \
  --validation_split_percentage 1 \
  --audio_column_name "eeg" \
  --seed 42 \
  --push_to_hub False

echo "=============================="
echo "Finished Wav2Vec2 pretraining job"
echo "Output directory: /Users/kristal/Desktop/wave2vec2_pretrained_1_stan"
echo "=============================="
